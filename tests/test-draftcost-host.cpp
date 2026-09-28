// LOCAL (draftcost): model-free probe of the MTP draft loop on a random tiny QWEN35 (+1 MTP nextn block).
// Never loads a real model. Measures the host side of the single-token draft decodes (wall time per draft token,
// llama_decode call, output sync+read) the way common_speculative_impl_draft_mtp::draft() drives them, and checks the
// draft outputs of a q8_0 draft KV cache against an f16 one.
//
//   draft chain: n_draft single-token decodes on the MTP context (token + hidden row), each followed by reading the
//   draft sampler's result (backend top-k candidates, or the raw logits) and the h_nextn row; then a KV rollback and a
//   catch-up decode of the "accepted" tokens, as in the server loop. Only the draft context is used.
//
// usage: test-draftcost-host [steps=300] [n_ctx=65536] [n_prompt=60000] [n_vocab=4096] [backend_sampling=1]
//                            [n_draft=5] [kv=f16|q8_0|cmp] [n_embd=512] [seed=1] [role=dft|tgt]
//   kv=cmp runs an f16-KV and a q8_0-KV draft context side by side on the same inputs and reports the max abs
//   difference of the h_nextn rows and the top-1 agreement of the draft candidates.
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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    const size_t seed0 = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    std::mt19937 gen(seed0 ^ hasher(tensor->name));
    std::normal_distribution<float> dis(0.0f, 0.1f);
    const bool is_ssm_a = strstr(tensor->name, "ssm_a") != nullptr;
    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
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

static gguf_context_ptr make_qwen35(uint32_t n_layer, uint32_t n_vocab, uint32_t n_embd) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_QWEN35, ret.get());
    const uint32_t n_head = 2, n_ff = 384, n_embd_head = n_embd / n_head; // head dim 256 at n_embd 512 (XMX FA)

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,        llm_arch_name(LLM_ARCH_QWEN35));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                  n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,              uint32_t(262144));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,            n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                 n_layer + 1u);
    ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,        uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,         n_ff);
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,     uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,        n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,     uint32_t(1));
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

struct acc_t { int64_t n = 0; double ms = 0; };

// FNV-1a over the draft outputs that decide the drafted token: the backend top-k candidates as a SET (sorted by id,
// with their logits) or, on the host path, the first n_hash logits of the row, plus the h_nextn row. Two runs that
// differ only in how the draft logits are laid out (LLAMA_MTP_DRAFT_NOPAD) must print the same value.
static uint64_t g_hash = 1469598103934665603ull;
static void hash_bytes(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++) {
        g_hash = (g_hash ^ b[i]) * 1099511628211ull;
    }
}

// role=tgt: two TARGET contexts (f16 KV, q8_0 KV) run the same prompt (2048-token ubatches -> the prefill FA path) and
// verify steps of 1..8 tokens with all logits + a random-prefix rollback (n_rs_seq 8, KV seq_rm), as the server does.
// Reports the q8_0-vs-f16 logit difference and argmax agreement; FAIL on non-finite logits.
static int run_tgt(llama_model * model, int steps, uint32_t n_ctx, int n_prompt, uint32_t n_vocab, size_t seed) {
    std::vector<llama_context_ptr> ctxs;
    for (ggml_type t : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx      = n_ctx;
        cp.n_batch    = 2048;
        cp.n_ubatch   = 2048;
        cp.n_seq_max  = 1;
        cp.kv_unified = true;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.n_rs_seq   = 8;
        cp.type_k     = t;
        cp.type_v     = t;
        cp.n_threads  = cp.n_threads_batch = 4;
        llama_context * c = llama_init_from_model(model, cp);
        GGML_ASSERT(c);
        ctxs.emplace_back(c);
    }
    std::mt19937 rng(seed * 7919 + 3);
    auto rnd = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
    llama_batch batch = llama_batch_init(2048, 0, 1);
    for (int i0 = 0; i0 < n_prompt; i0 += 2048) {
        const int n = std::min(2048, n_prompt - i0);
        common_batch_clear(batch);
        for (int k = 0; k < n; k++) {
            common_batch_add(batch, rnd(0, n_vocab - 1), i0 + k, { 0 }, i0 + k == n_prompt - 1);
        }
        for (auto & c : ctxs) {
            GGML_ASSERT(llama_decode(c.get(), batch) == 0);
        }
    }
    double max_d = 0.0, max_l = 0.0, sum_d = 0.0;
    int64_t n_rows = 0, n_arg_same = 0, n_bad = 0;
    std::map<int, int64_t> n_by_size;
    llama_pos pos = n_prompt;
    for (int step = 0; step < steps && pos + 10 < (llama_pos) n_ctx; step++) {
        const int n = rnd(1, 8);
        common_batch_clear(batch);
        for (int k = 0; k < n; k++) {
            common_batch_add(batch, rnd(0, n_vocab - 1), pos + k, { 0 }, true);
        }
        std::vector<std::vector<float>> lg(ctxs.size());
        for (size_t ci = 0; ci < ctxs.size(); ci++) {
            GGML_ASSERT(llama_decode(ctxs[ci].get(), batch) == 0);
            for (int k = 0; k < n; k++) {
                const float * l = llama_get_logits_ith(ctxs[ci].get(), k);
                lg[ci].insert(lg[ci].end(), l, l + n_vocab);
            }
        }
        for (int k = 0; k < n; k++) {
            const float * a = lg[0].data() + (size_t) k * n_vocab;
            const float * b = lg[1].data() + (size_t) k * n_vocab;
            for (uint32_t v = 0; v < n_vocab; v++) {
                if (!std::isfinite(a[v]) || !std::isfinite(b[v])) { n_bad++; break; }
                const double d = fabs((double) a[v] - b[v]);
                max_d = std::max(max_d, d);
                max_l = std::max(max_l, (double) fabsf(a[v]));
                sum_d += d;
            }
            n_arg_same += (std::max_element(a, a + n_vocab) - a) == (std::max_element(b, b + n_vocab) - b);
            n_rows++;
        }
        n_by_size[n]++;
        const int n_acc = rnd(1, n);
        for (auto & c : ctxs) {
            if (n_acc < n && !llama_memory_seq_rm(llama_get_memory(c.get()), 0, pos + n_acc, -1)) {
                fprintf(stderr, "seq_rm failed\n");
                return 1;
            }
        }
        pos += n_acc;
    }
    for (auto & c : ctxs) {
        llama_synchronize(c.get());
    }
    fprintf(stderr, "DRAFTCOST-TGT f16 vs q8_0 KV over %lld verify rows (sizes:", (long long) n_rows);
    for (auto & kv : n_by_size) fprintf(stderr, " n%d=%lld", kv.first, (long long) kv.second);
    fprintf(stderr, "): max|dlogit|=%.4g mean=%.4g (max|logit|=%.4g) argmax agree %lld/%lld\n",
            max_d, sum_d / std::max<int64_t>(1, n_rows * n_vocab), max_l, (long long) n_arg_same, (long long) n_rows);
    fprintf(stderr, "DRAFTCOST %s non-finite rows=%lld\n", n_bad == 0 ? "PASS" : "FAIL", (long long) n_bad);
    llama_batch_free(batch);
    ctxs.clear();
    return n_bad == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    const int         steps    = argc > 1 ? atoi(argv[1]) : 300;
    const uint32_t    n_ctx    = argc > 2 ? atoi(argv[2]) : 65536;
    const int         n_prompt = argc > 3 ? atoi(argv[3]) : 60000;
    const uint32_t    n_vocab  = argc > 4 ? atoi(argv[4]) : 4096;
    const bool        bsamp    = argc > 5 ? atoi(argv[5]) != 0 : true;
    const int         n_draft  = argc > 6 ? atoi(argv[6]) : 5;
    const std::string kv       = argc > 7 ? argv[7] : "f16";
    const uint32_t    n_embd   = argc > 8 ? atoi(argv[8]) : 512;
    const size_t      seed     = argc > 9 ? atoll(argv[9]) : 1;

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
    fprintf(stderr, "device: %s steps=%d n_ctx=%u n_prompt=%d n_vocab=%u backend_sampling=%d n_draft=%d kv=%s n_embd=%u\n",
            ggml_backend_dev_description(dev), steps, n_ctx, n_prompt, n_vocab, (int) bsamp, n_draft, kv.c_str(), n_embd);

    auto gguf = make_qwen35(4, n_vocab, n_embd);
    llama_model_params mp = llama_model_default_params();
    std::vector<ggml_backend_dev_t> devs = { dev, nullptr };
    mp.devices  = devs.data();
    mp.load_mtp = true;
    size_t wseed = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf.get(), set_tensor_data, &wseed, mp));
    GGML_ASSERT(model);
    const int n_embd_m = llama_model_n_embd(model.get());

    if (argc > 10 && std::string(argv[10]) == "tgt") {
        const int rc = run_tgt(model.get(), steps, n_ctx, n_prompt, n_vocab, seed);
        model.reset();
        llama_backend_free();
        return rc;
    }

    const bool cmp = kv == "cmp";
    // host path: hash only the ids a trimmed draft vocabulary can produce (LLAMA_MTP_DRAFT_VOCAB), since the tail is
    // -1e30 when padded and -inf with LLAMA_MTP_DRAFT_NOPAD
    const size_t n_hash = getenv("LLAMA_MTP_DRAFT_VOCAB") ? (size_t) atoll(getenv("LLAMA_MTP_DRAFT_VOCAB")) : (size_t) n_vocab;
    std::vector<ggml_type> kv_types;
    if (cmp) {
        kv_types = { GGML_TYPE_F16, GGML_TYPE_Q8_0 };
    } else {
        kv_types = { kv == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16 };
    }

    std::vector<llama_context_ptr> dfts;
    std::vector<llama_sampler *>   chains;
    for (ggml_type t : kv_types) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx      = n_ctx;
        cp.n_batch    = 2048;
        cp.n_ubatch   = 2048;
        cp.n_seq_max  = 1;
        cp.kv_unified = true;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.n_rs_seq   = 0;
        cp.ctx_type   = LLAMA_CONTEXT_TYPE_MTP;
        cp.type_k     = t;
        cp.type_v     = t;
        cp.n_threads  = cp.n_threads_batch = 4;
        llama_context * c = llama_init_from_model(model.get(), cp);
        GGML_ASSERT(c);
        llama_set_embeddings_nextn(c, true, true);
        llama_sampler * chain = nullptr;
        if (bsamp) {
            chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));
            if (!llama_set_sampler(c, 0, chain)) {
                fprintf(stderr, "backend sampler not supported\n");
                llama_sampler_free(chain);
                chain = nullptr;
            }
        }
        chains.push_back(chain);
        dfts.emplace_back(c);
    }

    std::mt19937 rng(seed * 7919 + 1);
    auto rnd = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
    std::normal_distribution<float> nd(0.0f, 1.0f);

    llama_batch batchd = llama_batch_init(2048, n_embd_m, 1);
    std::vector<llama_token> tok_store(2048);
    batchd.token = tok_store.data(); // llama_batch_init allocates either token or embd; MTP batches carry both

    // --- prompt: fill the draft KV to n_prompt (random hidden rows so the attention sees non-trivial values)
    for (int i0 = 0; i0 < n_prompt; i0 += 2048) {
        const int n = std::min(2048, n_prompt - i0);
        batchd.n_tokens = n;
        for (int k = 0; k < n; k++) {
            batchd.token[k] = rnd(0, n_vocab - 1); batchd.pos[k] = i0 + k; batchd.n_seq_id[k] = 1; batchd.seq_id[k][0] = 0; batchd.logits[k] = 0;
            for (int e = 0; e < n_embd_m; e++) batchd.embd[(size_t) k * n_embd_m + e] = nd(rng);
        }
        for (auto & c : dfts) {
            GGML_ASSERT(llama_decode(c.get(), batchd) == 0);
        }
    }
    for (auto & c : dfts) {
        llama_synchronize(c.get());
    }
    fprintf(stderr, "prompt done (%d tokens)\n", n_prompt);

    std::map<std::string, acc_t> tm;
    auto add = [&](const std::string & k, int64_t t0) { auto & a = tm[k]; a.n++; a.ms += (ggml_time_us() - t0) / 1000.0; };

    double max_dh = 0.0, max_h = 0.0;
    int64_t n_top1 = 0, n_top1_same = 0;
    int n_bad = 0;

    llama_pos pos = n_prompt;
    llama_token last = rnd(0, n_vocab - 1);
    std::vector<float> pending(n_embd_m);
    for (auto & x : pending) x = nd(rng);

    for (int step = 0; step < steps && pos + n_draft + 2 < (llama_pos) n_ctx; step++) {
        std::vector<std::vector<float>>       h_out(dfts.size());
        std::vector<std::vector<llama_token>> top1(dfts.size());
        for (size_t ci = 0; ci < dfts.size(); ci++) {
            llama_context * c = dfts[ci].get();
            std::vector<float> h = pending;
            llama_token tok = last;
            const int64_t t_chain = ggml_time_us();
            for (int d = 0; d < n_draft; d++) {
                batchd.n_tokens = 1;
                batchd.token[0] = tok; batchd.pos[0] = pos + d; batchd.n_seq_id[0] = 1; batchd.seq_id[0][0] = 0; batchd.logits[0] = 1;
                memcpy(batchd.embd, h.data(), n_embd_m * sizeof(float));
                int64_t t0 = ggml_time_us();
                GGML_ASSERT(llama_decode(c, batchd) == 0);
                add("draft.decode_call", t0);
                t0 = ggml_time_us();
                llama_token best = -1;
                if (chains[ci]) {
                    const uint32_t      nc  = llama_get_sampled_candidates_count_ith(c, 0);
                    const llama_token * cid = llama_get_sampled_candidates_ith(c, 0);
                    const float       * clg = llama_get_sampled_logits_ith(c, 0);
                    float bl = -INFINITY;
                    std::vector<std::pair<llama_token, float>> cand;
                    for (uint32_t k = 0; k < nc; k++) {
                        if (clg[k] > bl) { bl = clg[k]; best = cid[k]; }
                        cand.emplace_back(cid[k], clg[k]);
                    }
                    std::sort(cand.begin(), cand.end());
                    for (auto & pr : cand) {
                        hash_bytes(&pr.first, sizeof(pr.first));
                        hash_bytes(&pr.second, sizeof(pr.second));
                    }
                } else {
                    const float * lg = llama_get_logits_ith(c, 0);
                    best = (llama_token) (std::max_element(lg, lg + n_vocab) - lg);
                    hash_bytes(lg, sizeof(float) * std::min<size_t>(n_hash, n_vocab));
                    hash_bytes(&best, sizeof(best));
                }
                const float * hr = llama_get_embeddings_nextn_ith(c, 0);
                add("draft.sync+read", t0);
                hash_bytes(hr, sizeof(float) * n_embd_m);
                memcpy(h.data(), hr, n_embd_m * sizeof(float));
                for (int e = 0; e < n_embd_m; e++) {
                    if (!std::isfinite(hr[e])) { n_bad++; break; }
                }
                h_out[ci].insert(h_out[ci].end(), hr, hr + n_embd_m);
                top1[ci].push_back(best);
                // cmp mode: both contexts draft the SAME chain (the f16 one's tokens and rows) so they stay comparable
                tok = (cmp && ci > 0) ? top1[0][d] : (best >= 0 ? best : 0);
                if (cmp && ci > 0) {
                    memcpy(h.data(), h_out[0].data() + (size_t) d * n_embd_m, n_embd_m * sizeof(float));
                }
            }
            add("draft.chain", t_chain);
            const int64_t t0 = ggml_time_us();
            llama_memory_seq_rm(llama_get_memory(c), 0, pos, -1);
            add("draft.seq_rm", t0);
        }
        // accept a random prefix: catch-up decode of the accepted tokens with fresh random rows
        const int n_acc = rnd(1, n_draft + 1);
        batchd.n_tokens = n_acc;
        for (int k = 0; k < n_acc; k++) {
            batchd.token[k] = rnd(0, n_vocab - 1); batchd.pos[k] = pos + k; batchd.n_seq_id[k] = 1; batchd.seq_id[k][0] = 0; batchd.logits[k] = 0;
            for (int e = 0; e < n_embd_m; e++) batchd.embd[(size_t) k * n_embd_m + e] = nd(rng);
        }
        for (auto & c : dfts) {
            const int64_t t0 = ggml_time_us();
            GGML_ASSERT(llama_decode(c.get(), batchd) == 0);
            add("catchup.decode_call", t0);
        }
        for (int e = 0; e < n_embd_m; e++) pending[e] = nd(rng);
        pos += n_acc;
        last = rnd(0, n_vocab - 1);

        if (cmp) {
            for (size_t i = 0; i < h_out[0].size(); i++) {
                max_dh = std::max(max_dh, (double) fabsf(h_out[0][i] - h_out[1][i]));
                max_h  = std::max(max_h,  (double) fabsf(h_out[0][i]));
            }
            for (size_t i = 0; i < top1[0].size(); i++) {
                n_top1++;
                n_top1_same += top1[0][i] == top1[1][i];
            }
        }
    }
    for (auto & c : dfts) {
        llama_synchronize(c.get());
    }
    for (auto & kv2 : tm) {
        fprintf(stderr, "   %-22s n=%7lld avg=%8.3f ms\n", kv2.first.c_str(), (long long) kv2.second.n, kv2.second.ms / kv2.second.n);
    }
    const auto & ch = tm["draft.chain"];
    fprintf(stderr, "DRAFTCOST wall per draft token %.3f ms (%lld chains of %d)\n",
            ch.ms / std::max<int64_t>(1, ch.n) / n_draft, (long long) ch.n, n_draft);
    if (cmp) {
        fprintf(stderr, "DRAFTCOST-CMP f16 vs q8_0 KV: max|dh|=%.4g (max|h|=%.4g) top1 agree %lld/%lld\n",
                max_dh, max_h, (long long) n_top1_same, (long long) n_top1);
    }
    fprintf(stderr, "DRAFTCOST-HASH %016llx (host-path logits hashed: first %zu)\n", (unsigned long long) g_hash, n_hash);
    const bool pass = n_bad == 0;
    fprintf(stderr, "DRAFTCOST %s non-finite rows=%d\n", pass ? "PASS" : "FAIL", n_bad);

    batchd.token = nullptr;
    llama_batch_free(batchd);
    for (size_t ci = 0; ci < dfts.size(); ci++) {
        if (chains[ci]) {
            llama_set_sampler(dfts[ci].get(), 0, nullptr);
        }
        dfts[ci].reset();
        if (chains[ci]) {
            llama_sampler_free(chains[ci]);
        }
    }
    model.reset();
    llama_backend_free();
    return pass ? 0 : 1;
}
