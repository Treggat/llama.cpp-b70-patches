// LOCAL (hostv): host cost + exactness probe of the server's speculative sample-and-accept step.
//
// A random tiny QWEN35 on the CPU backend with the production vocabulary size produces real target logits for an
// 8-token verify batch; then, per trial (fixed seeds), a sampler with the production chain (top-k 20, top-p 0.95,
// temp 1.0, min-p default, LLAMA_SAMPLER_FAST_TOPK) runs common_sampler_sample_and_accept_n_rejection on a random
// draft with random draft distributions, exactly like srv.sample_accept (including the sampler clone the server
// takes first). Prints the host time per call and per sampled position, and a checksum of all accepted tokens so
// that two builds / env settings can be compared token for token.
//
// usage: test-hostv-sampling [trials=2000] [n_vocab=248320] [seed=1]
#include "common.h"
#include "sampling.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    const size_t seed0 = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    std::mt19937 gen(seed0 ^ hasher(tensor->name));
    // wide logits so top-p keeps a varying number of candidates
    std::normal_distribution<float> dis(0.0f, strstr(tensor->name, "output") ? 0.6f : 0.1f);
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
        GGML_ABORT("unexpected type");
    }
}

static gguf_context_ptr make_model(uint32_t n_vocab) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_QWEN35, ret.get());
    const uint32_t n_embd = 64, n_head = 1, n_embd_head = n_embd / n_head;
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,        llm_arch_name(LLM_ARCH_QWEN35));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                  n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,              uint32_t(4096));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,            n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                 uint32_t(4));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,         uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,     uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,        n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,     n_head);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,     std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,              uint32_t(128));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,             uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,              uint32_t(64));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,          uint32_t(2));
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,             uint32_t(2));
    std::vector<std::string> list(n_vocab);
    std::vector<float>       scores(n_vocab, 0.0f);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL, "test");
    for (uint32_t i = 0; i < n_vocab; i++) {
        list[i] = "tok_" + std::to_string(i);
    }
    ms.add_kv(LLM_KV_TOKENIZER_LIST,   list);
    ms.add_kv(LLM_KV_TOKENIZER_SCORES, scores);
    return ret;
}

int main(int argc, char ** argv) {
    const int      trials  = argc > 1 ? atoi(argv[1]) : 2000;
    const uint32_t n_vocab = argc > 2 ? atoi(argv[2]) : 248320;
    const size_t   seed    = argc > 3 ? atoll(argv[3]) : 1;

    llama_backend_init();
    auto gguf = make_model(n_vocab);
    llama_model_params mp = llama_model_default_params();
    std::vector<ggml_backend_dev_t> devs = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
    mp.devices = devs.data();
    size_t wseed = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf.get(), set_tensor_data, &wseed, mp));
    GGML_ASSERT(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 64; cp.n_ubatch = 64; cp.n_seq_max = 1;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
    GGML_ASSERT(ctx);

    std::mt19937 rng(seed);
    llama_batch batch = llama_batch_init(64, 0, 1);
    for (int i = 0; i < 16; i++) {
        common_batch_add(batch, std::uniform_int_distribution<int>(0, n_vocab - 1)(rng), i, { 0 }, i >= 8);
    }
    GGML_ASSERT(llama_decode(ctx.get(), batch) == 0);
    llama_synchronize(ctx.get());

    common_params_sampling sp;
    sp.top_k = 20;
    sp.top_p = 0.95f;
    sp.temp  = 1.0f;

    std::vector<int> idxs(8);
    for (int i = 0; i < 8; i++) {
        idxs[i] = 8 + i;
    }

    double t_total = 0, t_clone = 0;
    int64_t n_pos = 0;
    uint64_t checksum = 1469598103934665603ull;
    for (int t = 0; t < trials; t++) {
        sp.seed = (uint32_t) (seed * 1000003 + t);
        common_sampler_ptr smpl(common_sampler_init(model.get(), sp));

        // draft: 7 tokens taken from the target top candidates (so a varying prefix gets accepted), q peaked on them
        const int n_draft = 7;
        llama_tokens draft;
        std::vector<std::vector<llama_token_data>> draft_q;
        {
            common_sampler_ptr probe(common_sampler_init(model.get(), sp));
            for (int i = 0; i < n_draft; i++) {
                common_sampler_sample(probe.get(), ctx.get(), idxs[i]);
                const auto * cur = common_sampler_get_candidates(probe.get(), true);
                const int pick = std::uniform_int_distribution<int>(0, std::min<int>(3, (int) cur->size - 1))(rng);
                draft.push_back(cur->data[pick].id);
                std::vector<llama_token_data> q(cur->data, cur->data + cur->size);
                for (auto & e : q) {
                    const float qd = (t % 3 == 0) ? 0.7f : (t % 3 == 1) ? 0.05f : 0.001f; // vary the acceptance depth
                    e.p = e.id == draft.back() ? qd : (1.0f - qd) / std::max<size_t>(1, q.size() - 1);
                }
                draft_q.push_back(q);
            }
        }

        const int64_t t0 = ggml_time_us();
        common_sampler_ptr save(common_sampler_clone(smpl.get()));
        const int64_t t1 = ggml_time_us();
        const auto acc = common_sampler_sample_and_accept_n_rejection(smpl.get(), ctx.get(), idxs, draft, draft_q, false);
        const int64_t t2 = ggml_time_us();
        t_clone += (t1 - t0) / 1000.0;
        t_total += (t2 - t0) / 1000.0;
        n_pos   += (int64_t) acc.size();
        for (auto id : acc) {
            checksum = (checksum ^ (uint64_t) (uint32_t) id) * 1099511628211ull;
        }
        checksum = (checksum ^ 0xABCDu) * 1099511628211ull;
    }
    fprintf(stderr, "HOSTV-SAMPLING trials=%d n_vocab=%u: sample_accept %.4f ms/call (clone %.4f), %.2f positions/call, %.4f ms/position, checksum=%016llx\n",
            trials, n_vocab, t_total / trials, t_clone / trials, (double) n_pos / trials, t_total / n_pos, (unsigned long long) checksum);

    // breakdown: logits access, one full common_sampler_sample, and the raw top-k selection pass
    {
        const int N = 500;
        common_sampler_ptr smpl(common_sampler_init(model.get(), sp));
        fprintf(stderr, "HOSTV-SAMPLING chain: %s\n", common_sampler_print(smpl.get()).c_str());
        int64_t t0 = ggml_time_us();
        float acc_f = 0.0f;
        for (int i = 0; i < N; i++) {
            acc_f += llama_get_logits_ith(ctx.get(), 8 + (i & 7))[i & 1023];
        }
        const double t_logits = (ggml_time_us() - t0) / 1000.0 / N;
        t0 = ggml_time_us();
        llama_token tacc = 0;
        for (int i = 0; i < N; i++) {
            tacc += common_sampler_sample(smpl.get(), ctx.get(), 8 + (i & 7));
        }
        const double t_sample = (ggml_time_us() - t0) / 1000.0 / N;
        // the max pass alone (what select_top_k does for every 64-float block)
        t0 = ggml_time_us();
        float mx = 0.0f;
        for (int i = 0; i < N; i++) {
            const float * lg = llama_get_logits_ith(ctx.get(), 8 + (i & 7));
            float m = lg[0];
            for (uint32_t j = 1; j < n_vocab; j++) {
                m = std::max(m, lg[j]);
            }
            mx += m;
        }
        const double t_max = (ggml_time_us() - t0) / 1000.0 / N;
        // a copy of common_sampler::select_top_k (k = 20)
        t0 = ggml_time_us();
        std::vector<llama_token_data> cur;
        for (int i = 0; i < N; i++) {
            const float * logits = llama_get_logits_ith(ctx.get(), 8 + (i & 7));
            const int k = 20;
            auto better = [](const llama_token_data & a, const llama_token_data & b) {
                return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
            };
            cur.resize(k);
            for (int j = 0; j < k; ++j) {
                cur[j] = llama_token_data{j, logits[j], 0.0f};
            }
            std::make_heap(cur.begin(), cur.end(), better);
            float thr = cur[0].logit;
            constexpr int B = 64;
            for (int j0 = k, end; j0 < (int) n_vocab; j0 = end) {
                end = std::min((int) n_vocab, (j0 / B + 1) * B);
                float m = logits[j0];
                for (int j = j0 + 1; j < end; ++j) {
                    m = std::max(m, logits[j]);
                }
                if (!(m > thr)) {
                    continue;
                }
                for (int j = j0; j < end; ++j) {
                    const float l = logits[j];
                    if (l > thr) {
                        std::pop_heap(cur.begin(), cur.end(), better);
                        cur.back() = llama_token_data{j, l, 0.0f};
                        std::push_heap(cur.begin(), cur.end(), better);
                        thr = cur[0].logit;
                    }
                }
            }
            std::sort(cur.begin(), cur.end(), better);
        }
        const double t_topk = (ggml_time_us() - t0) / 1000.0 / N;
        fprintf(stderr, "HOSTV-SAMPLING breakdown: llama_get_logits_ith %.4f ms, common_sampler_sample %.4f ms, plain max pass %.4f ms, select_top_k copy %.4f ms (%g %d %g %d)\n",
                t_logits, t_sample, t_max, t_topk, acc_f, tacc, mx, cur[0].id);
    }
    llama_batch_free(batch);
    ctx.reset();
    model.reset();
    llama_backend_free();
    return 0;
}
