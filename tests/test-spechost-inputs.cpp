// LOCAL (spechost): correctness and host cost of ggml_backend_sched graph-input uploads to a GPU backend
// (GGML_SYCL_ASYNC_INPUTS). Never loads a model.
//
// Mimics how llama.cpp drives the scheduler:
//  - two schedulers (like the target and the MTP draft context) on the same GPU;
//  - scheduler A alternates between two graph topologies with differently sized inputs, re-allocating the graph
//    every time (llama "build_alloc"); scheduler B runs a graph with one LARGE input (a prefill-size KQ mask) that is
//    bigger than the staging ring, so it must take the synchronous path;
//  - bursts of 8 graphs are launched back to back WITHOUT any host synchronization in between (prefill ubatches):
//    each launch writes fresh inputs, calls graph_compute_async, immediately clobbers the host inputs (legal for the
//    caller once compute_async has returned), and queues an async readback of the outputs;
//  - after each burst: synchronize and compare every output EXACTLY with a host reference (all values are exact in
//    fp32, so any stale / clobbered / reordered upload shows up as a mismatch).
// Prints mismatches, an FNV hash over all outputs (must be identical across GGML_SYCL_ASYNC_INPUTS=0/1 and graph
// on/off) and the host time of graph_compute_async per graph.
//
// usage: test-spechost-inputs [bursts] [n_mask_small] [n_mask_big]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const int N_EMBD = 5120;

struct graph_def {
    ggml_context * ctx  = nullptr;
    ggml_cgraph  * gf   = nullptr;
    ggml_tensor  * h    = nullptr;
    ggml_tensor  * e    = nullptr;
    ggml_tensor  * sc   = nullptr;
    ggml_tensor  * mask = nullptr;
    ggml_tensor  * out1 = nullptr;  // (h + e) * sc          [N_EMBD]
    ggml_tensor  * out2 = nullptr;  // sum(mask) + sum(h)    [1]
    int n_mask = 0;
};

static graph_def make_graph(int n_mask) {
    graph_def g;
    g.n_mask = n_mask;
    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    g.ctx  = ggml_init(ip);
    g.h    = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, N_EMBD);  ggml_set_input(g.h);
    g.e    = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, N_EMBD);  ggml_set_input(g.e);
    g.sc   = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 1);       ggml_set_input(g.sc);
    g.mask = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_mask);  ggml_set_input(g.mask);
    g.out1 = ggml_mul(g.ctx, ggml_add(g.ctx, g.h, g.e), g.sc);
    g.out2 = ggml_add(g.ctx, ggml_sum(g.ctx, g.mask), ggml_sum(g.ctx, g.h));
    ggml_set_output(g.out1);
    ggml_set_output(g.out2);
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out1);
    ggml_build_forward_expand(g.gf, g.out2);
    return g;
}

struct launch_rec {
    int   gi;       // which graph
    int   key;      // value seed
    std::vector<float> o1;
    float o2 = 0.0f;
};

static void fill_inputs(int key, int n_mask, std::vector<float> & vh, std::vector<float> & ve, std::vector<float> & vm, float & s) {
    // all values are small multiples of 0.5, every partial sum stays below 2^22 -> exact in fp32 in any order
    for (int i = 0; i < N_EMBD; ++i) {
        vh[i] = (float) ((key * 7 + i) % 13) - 6.0f;
        ve[i] = 0.5f * (float) ((key + i) % 9);
    }
    vm.resize(n_mask);
    for (int i = 0; i < n_mask; ++i) {
        const int r = (key * 31 + i) % 7;
        vm[i] = r == 0 ? -1.0f : (r < 3 ? 0.5f : 0.0f);
    }
    s = (float) (1 + key % 3);
}

#define DBG(...) do { if (getenv("SPECHOST_DBG")) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while (0)

int main(int argc, char ** argv) {
    const int bursts  = argc > 1 ? atoi(argv[1]) : 200;
    const int n_small = argc > 2 ? atoi(argv[2]) : 65536;
    const int n_big   = argc > 3 ? atoi(argv[3]) : 12 * 1024 * 1024;   // 48 MB of f32: larger than half the ring
    const int burst   = 8;

    ggml_backend_load_all();
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!gpu || !cpu) {
        fprintf(stderr, "no GPU or CPU backend\n");
        return 1;
    }
    printf("gpu backend: %s  GGML_SYCL_ASYNC_INPUTS=%s  GGML_SYCL_ENABLE_GRAPH=%s  ring MB=%s\n", ggml_backend_name(gpu),
           getenv("GGML_SYCL_ASYNC_INPUTS") ? getenv("GGML_SYCL_ASYNC_INPUTS") : "(unset)",
           getenv("GGML_SYCL_ENABLE_GRAPH") ? getenv("GGML_SYCL_ENABLE_GRAPH") : "(unset)",
           getenv("GGML_SYCL_ASYNC_INPUTS_MB") ? getenv("GGML_SYCL_ASYNC_INPUTS_MB") : "(default)");

    ggml_backend_dev_t gdev = ggml_backend_get_device(gpu);
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(gdev);
    ggml_backend_t backends[2] = { gpu, cpu };
    ggml_backend_buffer_type_t bufts[2] = { ggml_backend_get_default_buffer_type(gpu),
                                            host_buft ? host_buft : ggml_backend_get_default_buffer_type(cpu) };
    ggml_backend_sched_t schedA = ggml_backend_sched_new(backends, bufts, 2, 256, false, true);
    ggml_backend_sched_t schedB = ggml_backend_sched_new(backends, bufts, 2, 256, false, true);
    DBG("scheds ok\n");

    // graph 0/1 on scheduler A (small / medium inputs), graph 2 on scheduler B (large input)
    graph_def G[3] = { make_graph(n_small), make_graph(n_small * 3 + 1000), make_graph(n_big) };
    ggml_backend_sched_t S[3] = { schedA, schedA, schedB };
    int alloc_cur[2] = { -1, -1 };  // graph currently allocated on schedA / schedB

    // size scheduler A's compute buffers with its larger graph first, so later re-allocations never grow (and free)
    // a buffer that queued work may still use
    {
        graph_def w = make_graph(G[1].n_mask);
        for (int i = 0; i < ggml_graph_n_nodes(w.gf); ++i) {
            ggml_backend_sched_set_tensor_backend(schedA, ggml_graph_node(w.gf, i), gpu);
        }
        if (!ggml_backend_sched_alloc_graph(schedA, w.gf)) {
            fprintf(stderr, "alloc failed\n");
            return 1;
        }
        ggml_backend_sched_reset(schedA);
        ggml_free(w.ctx);
    }

    std::vector<float> vh(N_EMBD), ve(N_EMBD), vm, junk;
    uint64_t hash = 1469598103934665603ull;
    int64_t  n_bad = 0, n_launch = 0;
    double   t_launch[3] = { 0, 0, 0 };
    int64_t  n_timed[3]  = { 0, 0, 0 };

    for (int b = 0; b < bursts; ++b) {
        std::vector<launch_rec> recs;
        recs.reserve(burst);
        for (int k = 0; k < burst; ++k) {
            // pattern: mostly A graphs (alternating topologies), a B graph every 4th launch
            const int gi  = (k % 4 == 3) ? 2 : ((b + k) & 1);
            const int key = b * burst + k;
            graph_def & g = G[gi];
            ggml_backend_sched_t s = S[gi];
            const int si = gi == 2 ? 1 : 0;

            if (alloc_cur[si] != gi) {
                // a fresh graph for every allocation, like llama.cpp (a graph whose tensors already carry data from an
                // earlier allocation must not be allocated again)
                const int nm = g.n_mask;
                ggml_free(g.ctx);
                g = make_graph(nm);
                ggml_backend_sched_reset(s);
                for (int i = 0; i < ggml_graph_n_nodes(g.gf); ++i) {
                    ggml_backend_sched_set_tensor_backend(s, ggml_graph_node(g.gf, i), gpu);
                }
                DBG("alloc graph %d\n", gi);
                if (!ggml_backend_sched_alloc_graph(s, g.gf)) {
                    fprintf(stderr, "alloc failed\n");
                    return 1;
                }
                alloc_cur[si] = gi;
            }

            float sc = 0.0f;
            fill_inputs(key, g.n_mask, vh, ve, vm, sc);
            ggml_backend_tensor_set(g.h,    vh.data(), 0, ggml_nbytes(g.h));
            ggml_backend_tensor_set(g.e,    ve.data(), 0, ggml_nbytes(g.e));
            ggml_backend_tensor_set(g.sc,   &sc,       0, sizeof(float));
            ggml_backend_tensor_set(g.mask, vm.data(), 0, ggml_nbytes(g.mask));

            DBG("launch %d graph %d\n", key, gi);
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_sched_graph_compute_async(s, g.gf);
            const double dt = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            if (b > 2) { t_launch[gi] += dt; n_timed[gi]++; }

            // the caller may reuse its input tensors as soon as compute_async returns: clobber them
            junk.assign(std::max(g.n_mask, N_EMBD), 12345.0f);
            ggml_backend_tensor_set(g.h,    junk.data(), 0, ggml_nbytes(g.h));
            ggml_backend_tensor_set(g.e,    junk.data(), 0, ggml_nbytes(g.e));
            ggml_backend_tensor_set(g.sc,   junk.data(), 0, sizeof(float));
            ggml_backend_tensor_set(g.mask, junk.data(), 0, ggml_nbytes(g.mask));

            launch_rec r;
            r.gi = gi; r.key = key; r.o1.resize(N_EMBD);
            recs.push_back(std::move(r));
            launch_rec & rr = recs.back();
            ggml_backend_t ob = ggml_backend_sched_get_tensor_backend(s, g.out1);
            ggml_backend_tensor_get_async(ob, g.out1, rr.o1.data(), 0, ggml_nbytes(g.out1));
            ggml_backend_tensor_get_async(ob, g.out2, &rr.o2,       0, sizeof(float));
            n_launch++;
            // no synchronization before the next launch, even when it re-allocates a different graph on the same
            // scheduler (llama.cpp does not either): the readback above is ordered before it on the in-order queue
        }
        DBG("burst sync\n");
        ggml_backend_sched_synchronize(schedA);
        ggml_backend_sched_synchronize(schedB);

        for (auto & r : recs) {
            float sc = 0.0f;
            fill_inputs(r.key, G[r.gi].n_mask, vh, ve, vm, sc);
            double msum = 0.0, hsum = 0.0;
            for (float v : vm) msum += v;
            bool ok = true;
            for (int i = 0; i < N_EMBD; ++i) {
                hsum += vh[i];
                if (r.o1[i] != (vh[i] + ve[i]) * sc) ok = false;
            }
            if (r.o2 != (float) (msum + hsum)) ok = false;
            if (!ok) {
                if (n_bad < 8) {
                    fprintf(stderr, "MISMATCH burst %d key %d graph %d: o1[0]=%g ref=%g o2=%.1f ref=%.1f\n", b, r.key, r.gi,
                            r.o1[0], (vh[0] + ve[0]) * sc, r.o2, msum + hsum);
                }
                n_bad++;
            }
            for (float v : r.o1) { uint32_t u; memcpy(&u, &v, 4); hash = (hash ^ u) * 1099511628211ull; }
            { uint32_t u; memcpy(&u, &r.o2, 4); hash = (hash ^ u) * 1099511628211ull; }
        }
    }

    printf("launches=%lld mismatches=%lld hash=%016llx\n", (long long) n_launch, (long long) n_bad, (unsigned long long) hash);
    printf("compute_async host us/graph: small(%d)=%.1f medium(%d)=%.1f large(%d)=%.1f\n",
           G[0].n_mask, n_timed[0] ? t_launch[0] / n_timed[0] : 0.0,
           G[1].n_mask, n_timed[1] ? t_launch[1] / n_timed[1] : 0.0,
           G[2].n_mask, n_timed[2] ? t_launch[2] / n_timed[2] : 0.0);
    printf("%s\n", n_bad == 0 ? "SPECHOST_INPUTS_OK" : "SPECHOST_INPUTS_FAIL");

    ggml_backend_sched_free(schedA);
    ggml_backend_sched_free(schedB);
    for (auto & g : G) ggml_free(g.ctx);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return n_bad == 0 ? 0 : 1;
}
