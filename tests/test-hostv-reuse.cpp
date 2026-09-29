// LOCAL (hostv): model-free end-to-end exactness + host-time probe for the host-side speculative-decoding changes
// (LLAMA_GRAPH_REUSE_MULTI, LLAMA_KQ_MASK_FAST, LLAMA_SYNC_SKIP_IDLE, GGML_SYCL_GRAPH_RECORD_AHEAD / _SIG_MEMO).
//
// Builds a small random QWEN35 model in memory with the production layer layout (64 layers, full attention every
// 4th, + 1 MTP nextn block), so the target graph has the production node count and host build/split/alloc cost,
// but tiny tensors. Two independent context pairs (target + MTP draft) run the SAME server-like call sequence:
//   drafts: 0..7 single-token MTP decodes (token + hidden row), each read back (logits + nextn row)
//   verify: 1..8 tokens, all logits, llama_synchronize(tgt)
//   catch-up: MTP decode of the verify batch, NOT synchronized (the server reads the target outputs meanwhile)
//   read:   all target logits + nextn rows (after the catch-up was launched, as the server does)
//   accept: random prefix, rollback of the rest (n_rs_seq = 8 target snapshots, KV seq_rm on both)
// pair A has LLAMA_GRAPH_REUSE_MULTI=0, pair B LLAMA_GRAPH_REUSE_MULTI=<multi_b>; every row of A and B is compared
// bit for bit, and a checksum over all of pair A's outputs is printed so that runs with different global switches
// (the other envs) can be compared against each other.
//
// usage: test-hostv-reuse [steps=400] [multi_b=1] [n_layer=64] [n_ctx=4096] [seed=1] [n_vocab=4096] [mtp=1] [n_prompt=300]
// [draftreplay] env HOSTV_DFT_KV=q8_0: quantized MTP draft KV (server -ctkd/-ctvd q8_0; turns on the KV Hadamard rotation);
//   env HOSTV_N_HEAD=24 HOSTV_N_HEAD_KV=4: attention heads of dim 256 as in the house model (default 2 / 1 heads of 128)
// [verifystep] env HOSTV_N_EMBD / HOSTV_N_FF: model width / FFN size (default 256 / 384); HOSTV_QTYPE=q4_K: every 2D
//   weight whose rows are a multiple of 256 is q4_K (output.weight q8_0), filled with quantized random values, so the
//   quantized SYCL paths (XMX q4_K, split-K small rows, fusions) run; the model is built twice (pass 1 lists the shapes)
#include "common.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-ext.h"
#include "../src/llama-model-saver.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
static void set_env(const char * k, const char * v) { _putenv_s(k, v); }
#else
static void set_env(const char * k, const char * v) { setenv(k, v, 1); }
#endif

struct hostv_tinfo { std::string name; int64_t ne[4]; };
static std::vector<hostv_tinfo> g_hostv_seen;   // [verifystep] pass 1 of HOSTV_QTYPE

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    const size_t seed0 = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    std::mt19937 gen(seed0 ^ hasher(tensor->name));
    std::normal_distribution<float> dis(0.0f, 0.1f);
    const bool is_ssm_a = strstr(tensor->name, "ssm_a") != nullptr;
    const int64_t ne = ggml_nelements(tensor);
    g_hostv_seen.push_back({ tensor->name, { tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3] } });
    if (ggml_is_quantized(tensor->type)) {   // [verifystep] HOSTV_QTYPE
        std::vector<float> tmp(ne);
        for (auto & v : tmp) { v = dis(gen); }
        std::vector<uint8_t> q(ggml_nbytes(tensor));
        ggml_quantize_chunk(tensor->type, tmp.data(), q.data(), 0, ggml_nrows(tensor), tensor->ne[0], nullptr);
        ggml_backend_tensor_set(tensor, q.data(), 0, q.size());
    } else if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (auto & v : tmp) { const float x = dis(gen); v = is_ssm_a ? -fabsf(x) : x; }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (auto & v : tmp) { const float x = dis(gen); v = ggml_fp32_to_fp16(is_ssm_a ? -fabsf(x) : x); }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("unexpected tensor type %s for %s", ggml_type_name(tensor->type), tensor->name);
    }
}

static gguf_context_ptr make_qwen35(uint32_t n_layer, uint32_t n_vocab, bool mtp) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_QWEN35, ret.get());
    const uint32_t n_embd = getenv("HOSTV_N_EMBD") ? atoi(getenv("HOSTV_N_EMBD")) : 256, n_head = 2;
    const uint32_t n_ff   = getenv("HOSTV_N_FF") ? atoi(getenv("HOSTV_N_FF")) : 384;
    const uint32_t n_head_attn = getenv("HOSTV_N_HEAD") ? atoi(getenv("HOSTV_N_HEAD")) : 0;
    const uint32_t n_head_kv   = getenv("HOSTV_N_HEAD_KV") ? atoi(getenv("HOSTV_N_HEAD_KV")) : 1;
    const uint32_t n_embd_head = n_head_attn > 0 ? 256 : n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,        llm_arch_name(LLM_ARCH_QWEN35));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                  n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,              uint32_t(262144));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,            n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                 n_layer + (mtp ? 1u : 0u));
    if (mtp) {
        ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,    uint32_t(1));
    }
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,         n_ff);
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,     uint32_t(4));
    if (n_head_attn > 0) {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,    n_head_attn);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_kv);
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,    n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,  n_embd_head);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,    n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, uint32_t(1));
    }
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,     std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,              uint32_t(256));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,             uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,              uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,          n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,             uint32_t(2));
    {
        std::vector<std::string> list(n_vocab);
        std::vector<float>       scores(n_vocab, 0.0f);
        ms.add_kv(LLM_KV_TOKENIZER_MODEL, "test");
        for (uint32_t i = 0; i < n_vocab; i++) {
            list[i] = "tok_" + std::to_string(i);
        }
        ms.add_kv(LLM_KV_TOKENIZER_LIST,   list);
        ms.add_kv(LLM_KV_TOKENIZER_SCORES, scores);
    }
    return ret;
}

struct timing {
    std::map<std::string, std::pair<int64_t, double>> acc; // key -> (n, ms)
    void add(const std::string & k, int64_t t0) {
        auto & a = acc[k];
        a.first++;
        a.second += (ggml_time_us() - t0) / 1000.0;
    }
};

struct pair_ctx {
    llama_context_ptr  tgt;
    llama_context_ptr  dft;
    timing             tm;
    std::vector<float> pending_h;
    // outputs of the current step, compared between the pairs
    std::vector<float> out;
};

static int      n_mismatch = 0;
static int      n_compared = 0;
static uint64_t checksum   = 1469598103934665603ull;

static void hash_bytes(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++) {
        checksum = (checksum ^ b[i]) * 1099511628211ull;
    }
}

int main(int argc, char ** argv) {
    const int      steps    = argc > 1 ? atoi(argv[1]) : 400;
    const int      multi_b  = argc > 2 ? atoi(argv[2]) : 1;
    const uint32_t n_layer  = argc > 3 ? atoi(argv[3]) : 64;
    const uint32_t n_ctx    = argc > 4 ? atoi(argv[4]) : 4096;
    const size_t   seed     = argc > 5 ? atoll(argv[5]) : 1;
    const uint32_t n_vocab  = argc > 6 ? atoi(argv[6]) : 4096;
    const bool     mtp      = argc > 7 ? atoi(argv[7]) != 0 : true;
    const int      n_prompt = argc > 8 ? atoi(argv[8]) : 300;

    llama_backend_init();

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            dev = ggml_backend_dev_get(i);
            break;
        }
    }
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    fprintf(stderr, "device: %s  steps=%d multi_b=%d n_layer=%u n_ctx=%u seed=%zu n_vocab=%u mtp=%d n_prompt=%d\n",
            ggml_backend_dev_description(dev), steps, multi_b, n_layer, n_ctx, seed, n_vocab, (int) mtp, n_prompt);

    auto gguf = make_qwen35(n_layer, n_vocab, mtp);
    llama_model_params mp = llama_model_default_params();
    std::vector<ggml_backend_dev_t> devs = { dev, nullptr };
    mp.devices = devs.data();
    mp.load_mtp = mtp;
    size_t wseed = seed;
    const char * qtype_env = getenv("HOSTV_QTYPE");
    if (qtype_env && std::string(qtype_env) == "q4_K") {   // [verifystep] pass 1: list the tensors, then declare their types
        {
            llama_model_ptr m0(llama_model_init_from_user(gguf.get(), set_tensor_data, &wseed, mp));
            GGML_ASSERT(m0);
        }
        int n_q = 0;
        for (const auto & ti : g_hostv_seen) {
            const bool two_d = ti.ne[1] > 1 && ti.ne[2] == 1 && ti.ne[3] == 1;
            const bool is_out = ti.name == "output.weight";
            const ggml_type t = is_out ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_K;
            if (!two_d || ti.ne[0] % 256 != 0 || ti.name.find(".weight") == std::string::npos || ti.name.find("norm") != std::string::npos ||
                gguf_find_tensor(gguf.get(), ti.name.c_str()) != -1) {
                continue;
            }
            ggml_tensor tt;
            memset(&tt, 0, sizeof(tt));
            tt.type = t;
            for (int d = 0; d < 4; d++) { tt.ne[d] = ti.ne[d]; }
            tt.nb[0] = ggml_type_size(t);
            tt.nb[1] = tt.nb[0] * (tt.ne[0] / ggml_blck_size(t));
            tt.nb[2] = tt.nb[1] * tt.ne[1];
            tt.nb[3] = tt.nb[2] * tt.ne[2];
            ggml_set_name(&tt, ti.name.c_str());
            gguf_add_tensor(gguf.get(), &tt);
            n_q++;
        }
        fprintf(stderr, "HOSTV_QTYPE q4_K: %d quantized weights\n", n_q);
        g_hostv_seen.clear();
    }
    llama_model_ptr model(llama_model_init_from_user(gguf.get(), set_tensor_data, &wseed, mp));
    if (!model) {
        fprintf(stderr, "failed to create model\n");
        return 1;
    }
    const int n_embd      = llama_model_n_embd(model.get());
    const int n_vocab_out = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));

    auto make_pair = [&](int multi) {
        pair_ctx p;
        set_env("LLAMA_GRAPH_REUSE_MULTI", multi ? "1" : "0");
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx      = n_ctx;
        cp.n_batch    = 512;
        cp.n_ubatch   = 512;
        cp.n_seq_max  = 1;
        cp.kv_unified = true;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.n_rs_seq   = 8;
        cp.n_threads  = cp.n_threads_batch = 4;
        p.tgt.reset(llama_init_from_model(model.get(), cp));
        GGML_ASSERT(p.tgt);
        if (mtp) {
            cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            cp.n_rs_seq = 0;
            if (getenv("HOSTV_DFT_KV") && std::string(getenv("HOSTV_DFT_KV")) == "q8_0") {
                cp.type_k = GGML_TYPE_Q8_0;
                cp.type_v = GGML_TYPE_Q8_0;
            }
            p.dft.reset(llama_init_from_model(model.get(), cp));
            GGML_ASSERT(p.dft);
            llama_set_embeddings_nextn(p.tgt.get(), true, false);
            llama_set_embeddings_nextn(p.dft.get(), true, true);
        }
        p.pending_h.assign(n_embd, 0.0f);
        return p;
    };
    pair_ctx A = make_pair(0);
    pair_ctx B = make_pair(multi_b);
    pair_ctx * pairs[2] = { &A, &B };

    std::mt19937 rng(seed * 7919 + 1);
    auto rnd = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };

    llama_batch batch  = llama_batch_init(512, 0, 1);
    llama_batch batchd = llama_batch_init(512, n_embd, 1); // token + embd (MTP)
    std::vector<llama_token> tok_store(512);
    batchd.token = tok_store.data(); // llama_batch_init allocates either token or embd; MTP batches carry both

    auto push = [&](pair_ctx & p, const float * v, size_t n) {
        p.out.insert(p.out.end(), v, v + n);
    };
    auto compare_step = [&](int step, const char * what) {
        n_compared++;
        hash_bytes(A.out.data(), A.out.size() * sizeof(float));
        if (A.out.size() != B.out.size() || memcmp(A.out.data(), B.out.data(), A.out.size() * sizeof(float)) != 0) {
            if (n_mismatch < 20) {
                size_t i = 0;
                while (i < A.out.size() && i < B.out.size() && memcmp(&A.out[i], &B.out[i], sizeof(float)) == 0) {
                    i++;
                }
                fprintf(stderr, "MISMATCH step %d %s at %zu/%zu\n", step, what, i, A.out.size());
            }
            n_mismatch++;
        }
        A.out.clear();
        B.out.clear();
    };

    // --- prompt
    std::vector<llama_token> hist;
    for (int i = 0; i < n_prompt; i++) {
        hist.push_back(rnd(0, n_vocab - 1));
    }
    for (pair_ctx * p : pairs) {
        for (int i0 = 0; i0 < n_prompt; i0 += 512) {
            const int n = std::min(512, n_prompt - i0);
            common_batch_clear(batch);
            for (int i = i0; i < i0 + n; i++) {
                common_batch_add(batch, hist[i], i, { 0 }, i == n_prompt - 1);
            }
            GGML_ASSERT(llama_decode(p->tgt.get(), batch) == 0);
            if (mtp) {
                // shapes matter here, not semantics: zero hidden rows
                batchd.n_tokens = n;
                for (int k = 0; k < n; k++) {
                    batchd.token[k] = hist[i0 + k]; batchd.pos[k] = i0 + k; batchd.n_seq_id[k] = 1; batchd.seq_id[k][0] = 0; batchd.logits[k] = 0;
                    memset(batchd.embd + (size_t) k * n_embd, 0, n_embd * sizeof(float));
                }
                GGML_ASSERT(llama_decode(p->dft.get(), batchd) == 0);
            }
        }
        llama_synchronize(p->tgt.get());
        if (mtp) {
            llama_synchronize(p->dft.get());
        }
    }
    llama_pos pos = n_prompt;
    fprintf(stderr, "prompt done (%d tokens)\n", n_prompt);

    llama_token last = hist.back();
    int n_verify_tokens = 0;
    for (int step = 0; step < steps && pos + 16 < (llama_pos) n_ctx; step++) {
        // the random choices of this step are shared by both pairs
        const int n_draft = rnd(0, 7);
        std::vector<llama_token> draft;
        for (int d = 0; d < n_draft; d++) {
            draft.push_back(rnd(0, n_vocab - 1));
        }
        const int n_acc = rnd(1, n_draft + 1);
        const llama_token next = rnd(0, n_vocab - 1);

        std::vector<llama_token> vt = { last };
        vt.insert(vt.end(), draft.begin(), draft.end());

        for (pair_ctx * p : pairs) {
            // --- drafts (the MTP KV region >= pos was dropped at the end of the previous step)
            if (mtp) {
                std::vector<float> h = p->pending_h;
                for (int d = 0; d < n_draft; d++) {
                    batchd.n_tokens = 1;
                    batchd.token[0] = d == 0 ? last : draft[d - 1];
                    batchd.pos[0] = pos + d; batchd.n_seq_id[0] = 1; batchd.seq_id[0][0] = 0; batchd.logits[0] = 1;
                    memcpy(batchd.embd, h.data(), n_embd * sizeof(float));
                    int64_t t0 = ggml_time_us();
                    GGML_ASSERT(llama_decode(p->dft.get(), batchd) == 0);
                    p->tm.add("dft.draft.n1", t0);
                    t0 = ggml_time_us();
                    const float * lg = llama_get_logits_ith(p->dft.get(), 0); // the sampler's synchronize
                    p->tm.add("dft.draft.sync+read", t0);
                    push(*p, lg, n_vocab_out);
                    const float * hr = llama_get_embeddings_nextn_ith(p->dft.get(), 0);
                    memcpy(h.data(), hr, n_embd * sizeof(float));
                    push(*p, hr, n_embd);
                }
                const int64_t t0 = ggml_time_us();
                llama_memory_seq_rm(llama_get_memory(p->dft.get()), 0, pos, -1);
                p->tm.add("dft.seq_rm", t0);
            }

            // --- verify + synchronize (server decode())
            common_batch_clear(batch);
            for (int k = 0; k < (int) vt.size(); k++) {
                common_batch_add(batch, vt[k], pos + k, { 0 }, true);
            }
            int64_t t0 = ggml_time_us();
            GGML_ASSERT(llama_decode(p->tgt.get(), batch) == 0);
            p->tm.add("tgt.verify.n" + std::to_string(vt.size()), t0);
            llama_synchronize(p->tgt.get());

            // --- MTP catch-up of the whole verify batch, left in flight (server spec_process)
            if (mtp) {
                const float * h_tgt = llama_get_embeddings_nextn(p->tgt.get());
                batchd.n_tokens = (int) vt.size();
                for (int k = 0; k < (int) vt.size(); k++) {
                    batchd.token[k] = vt[k]; batchd.pos[k] = pos + k; batchd.n_seq_id[k] = 1; batchd.seq_id[k][0] = 0; batchd.logits[k] = 0;
                    const float * src = k == 0 ? p->pending_h.data() : h_tgt + (size_t) (k - 1) * n_embd;
                    memcpy(batchd.embd + (size_t) k * n_embd, src, n_embd * sizeof(float));
                }
                t0 = ggml_time_us();
                GGML_ASSERT(llama_decode(p->dft.get(), batchd) == 0);
                p->tm.add("dft.catchup.n" + std::to_string(vt.size()), t0);
            }

            // --- read the target outputs while the catch-up may still run (process_h_rows + sample_accept)
            t0 = ggml_time_us();
            for (int k = 0; k < (int) vt.size(); k++) {
                push(*p, llama_get_logits_ith(p->tgt.get(), k), n_vocab_out);
                if (mtp) {
                    push(*p, llama_get_embeddings_nextn_ith(p->tgt.get(), k), n_embd);
                }
            }
            p->tm.add("tgt.read_outputs", t0);

            // --- accept n_acc of the verify tokens, roll back the rest
            if (mtp) {
                memcpy(p->pending_h.data(), llama_get_embeddings_nextn_ith(p->tgt.get(), n_acc - 1), n_embd * sizeof(float));
            }
            t0 = ggml_time_us();
            if (n_acc < (int) vt.size() && !llama_memory_seq_rm(llama_get_memory(p->tgt.get()), 0, pos + n_acc, -1)) {
                fprintf(stderr, "tgt seq_rm failed (rollback %d)\n", (int) vt.size() - n_acc);
                return 1;
            }
            p->tm.add("tgt.seq_rm", t0);
            if (mtp) {
                t0 = ggml_time_us();
                llama_memory_seq_rm(llama_get_memory(p->dft.get()), 0, pos + n_acc, -1);
                p->tm.add("dft.seq_rm", t0);
            }
        }
        compare_step(step, "step outputs");

        n_verify_tokens += vt.size();
        pos += n_acc;
        last = next;

        // every 100 steps: a 40-token turn (bigger than LLAMA_GRAPH_REUSE_MULTI_TOKENS -> the gf_res_prev path and a
        // different allocation), then generation continues on the cached small graphs
        if ((step + 1) % 100 == 0 && pos + 60 < (llama_pos) n_ctx) {
            std::vector<llama_token> turn;
            for (int i = 0; i < 40; i++) {
                turn.push_back(rnd(0, n_vocab - 1));
            }
            turn[0] = last;
            for (pair_ctx * p : pairs) {
                common_batch_clear(batch);
                for (int i = 0; i < 40; i++) {
                    common_batch_add(batch, turn[i], pos + i, { 0 }, i == 39);
                }
                GGML_ASSERT(llama_decode(p->tgt.get(), batch) == 0);
                push(*p, llama_get_logits_ith(p->tgt.get(), -1), n_vocab_out);
                if (mtp) {
                    const float * h_tgt = llama_get_embeddings_nextn(p->tgt.get());
                    batchd.n_tokens = 40;
                    for (int k = 0; k < 40; k++) {
                        batchd.token[k] = turn[k]; batchd.pos[k] = pos + k; batchd.n_seq_id[k] = 1; batchd.seq_id[k][0] = 0; batchd.logits[k] = 0;
                        const float * src = k == 0 ? p->pending_h.data() : h_tgt + (size_t) (k - 1) * n_embd;
                        memcpy(batchd.embd + (size_t) k * n_embd, src, n_embd * sizeof(float));
                    }
                    memcpy(p->pending_h.data(), h_tgt + (size_t) 39 * n_embd, n_embd * sizeof(float));
                    GGML_ASSERT(llama_decode(p->dft.get(), batchd) == 0);
                    llama_synchronize(p->dft.get());
                }
            }
            compare_step(step, "turn outputs");
            pos += 40;
            last = rnd(0, n_vocab - 1);
        }
        if ((step + 1) % 100 == 0) {
            fprintf(stderr, "step %d pos %d compared %d mismatches %d\n", step + 1, pos, n_compared, n_mismatch);
        }
    }
    for (pair_ctx * p : pairs) {
        llama_synchronize(p->tgt.get());
        if (mtp) {
            llama_synchronize(p->dft.get());
        }
    }

    for (pair_ctx * p : pairs) {
        const char * name = p == &A ? "A(ref)" : (multi_b ? "B(multi)" : "B(ref2)");
        fprintf(stderr, "== %s: tgt graphs reused %d", name, llama_perf_context(p->tgt.get()).n_reused);
        if (mtp) {
            fprintf(stderr, ", dft graphs reused %d", llama_perf_context(p->dft.get()).n_reused);
        }
        fprintf(stderr, "\n");
        for (auto & kv : p->tm.acc) {
            fprintf(stderr, "   %-22s n=%6lld avg=%8.3f ms (host)\n", kv.first.c_str(),
                    (long long) kv.second.first, kv.second.second / kv.second.first);
        }
    }
    fprintf(stderr, "HOSTV-REUSE %s: steps_compared=%d mismatches=%d final_pos=%d verify_tokens=%d checksum=%016llx\n",
            n_mismatch == 0 ? "PASS" : "FAIL", n_compared, n_mismatch, pos, n_verify_tokens, (unsigned long long) checksum);

    batchd.token = nullptr;
    llama_batch_free(batch);
    llama_batch_free(batchd);
    for (pair_ctx * p : pairs) {
        p->tgt.reset();
        p->dft.reset();
    }
    model.reset();
    llama_backend_free();
    return n_mismatch == 0 ? 0 : 1;
}
