//
// oneDNN int4 weight-decompression matmul for q4_K weights - see mmq-dnn-u4.hpp
//
#include "mmq-dnn-u4.hpp"
#include "convert.hpp"
#include "ggml-sycl.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <vector>

#if GGML_SYCL_DNNL
#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#endif

static int ggml_sycl_dnn_u4_env() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_SYCL_DNN_U4");
        v = e ? (atoi(e) != 0 ? 1 : 0) : 0;   // default off until it has been through the seat A/B
    }
    return v;
}

bool ggml_sycl_dnn_u4_enabled() {
#if GGML_SYCL_DNNL
    return ggml_sycl_dnn_u4_env() != 0;
#else
    return false;
#endif
}

bool ggml_sycl_dnn_u4_can_use(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                              const ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    if (!ggml_sycl_dnn_u4_enabled()) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_K || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->op != GGML_OP_MUL_MAT) {   // MUL_MAT_ID comes through the same dispatcher with a stack dst
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1) {   // batched src1 (several sequences per ubatch) is fine: the driver slices it
        return false;
    }
    if (src0->ne[0] % QK_K != 0 || src0->ne[1] < 2048 || !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }
    // shape policy (measured 2026-09-25 on the B70, test-backend-ops perf): the oneDNN u4 GEMM is flat across
    // 1..8 columns at ~120 us on 17408x5120 (MMVQ: 85/104/148/196/279) but ~175 us on 5120x17408 (MMVQ 86 at one
    // column, 250 at six), so it only pays where the weight has at least as many rows as columns (ffn up/gate,
    // attention projections), not for ffn_down. Tensors used mostly at one column (a draft/MTP layer) can be kept
    // on MMVQ with GGML_SYCL_DNN_U4_SKIP_PREFIX, e.g. "blk.64." for a 64-layer model with one nextn layer.
    if (src0->ne[1] < src0->ne[0]) {
        return false;
    }
    static const char * skip_prefix = getenv("GGML_SYCL_DNN_U4_SKIP_PREFIX");
    if (skip_prefix && *skip_prefix && strncmp(src0->name, skip_prefix, strlen(skip_prefix)) == 0) {
        return false;
    }
    if (src0->extra == nullptr || src0->buffer == nullptr) {   // split buffers are excluded by the dispatcher
        return false;
    }
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    if (extra->optimized_feature.reorder) {   // already in the MMVQ SoA layout; not converting from there
        return false;
    }
    return true;
}

bool ggml_sycl_dnn_u4_is_repacked(const ggml_tensor * src0) {
    const auto * extra = src0 ? static_cast<const ggml_tensor_extra_gpu *>(src0->extra) : nullptr;
    return extra && extra->optimized_feature.dnn_u4;
}

#if GGML_SYCL_DNNL

// ---- q4_K helpers (device) ----

static __dpct_inline__ void dnn_u4_scale_min(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// value of element k (0..255) of a q4_K block in the original nibble layout
static __dpct_inline__ uint8_t dnn_u4_q4k_elem(const uint8_t * qs, int k) {
    const int     chunk = k >> 6;          // 64-element chunk
    const int     l     = k & 31;
    const uint8_t byte  = qs[chunk * 32 + l];
    return (k & 32) ? (byte >> 4) : (byte & 0xF);
}

// ---- repack ----

bool ggml_sycl_dnn_u4_prepare(ggml_backend_sycl_context & ctx, const ggml_tensor * src0) {
    auto * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (extra->optimized_feature.dnn_u4) {
        return true;
    }
    if (extra->optimized_feature.reorder) {
        return false;
    }

    const int64_t K       = src0->ne[0];
    const int64_t N       = src0->ne[1];
    const int64_t KB      = K / QK_K;           // blocks per row
    const int64_t nblocks = N * KB;
    const size_t  size    = ggml_nbytes(src0);
    GGML_ASSERT(size == (size_t) nblocks * sizeof(block_q4_K));

    queue_ptr stream = ctx.stream();
    uint8_t * data   = (uint8_t *) src0->data;

    // side buffer: f16 scales [K/32][N]
    const size_t  sc_bytes = (size_t) (K / 32) * N * sizeof(sycl::half);
    // second side buffer: the 6-bit group mins as u8, [N][K/32] (g innermost: the epilogue's lanes stride over g),
    // so the min correction reads one byte per group instead of decoding the packed 12-byte scale block
    const size_t  mn_bytes = (size_t) N * (K / 32) + (size_t) N * KB * sizeof(sycl::half);   // u8 mins [G][N] + f16 dmin [KB][N]
    sycl::half *  sc_dev   = nullptr;
    uint8_t *     mn_dev   = nullptr;
    uint8_t *     tmp      = nullptr;
    try {
        sc_dev = (sycl::half *) ggml_sycl_malloc_device(sc_bytes, *stream);
        mn_dev = (uint8_t *) ggml_sycl_malloc_device(mn_bytes, *stream);
        tmp    = (uint8_t *) ggml_sycl_malloc_device(size, *stream);
    } catch (const sycl::exception & e) {
        GGML_LOG_WARN("%s: allocation failed (%s), keeping the MMVQ path for %s\n", __func__, e.what(), src0->name);
        if (sc_dev) { ggml_sycl_free_device(sc_dev, *stream); }
        if (mn_dev) { ggml_sycl_free_device(mn_dev, *stream); }
        if (tmp)    { ggml_sycl_free_device(tmp, *stream); }
        return false;
    }
    if (sc_dev == nullptr || mn_dev == nullptr || tmp == nullptr) {
        GGML_LOG_WARN("%s: allocation failed, keeping the MMVQ path for %s\n", __func__, src0->name);
        if (sc_dev) { ggml_sycl_free_device(sc_dev, *stream); }
        if (mn_dev) { ggml_sycl_free_device(mn_dev, *stream); }
        if (tmp)    { ggml_sycl_free_device(tmp, *stream); }
        return false;
    }

    stream->memcpy(tmp, data, size).wait();

    uint8_t *     qs_out = data;
    uint8_t *     sc_out = data + (size_t) nblocks * (QK_K / 2);
    sycl::half2 * dm_out = (sycl::half2 *) (sc_out + (size_t) nblocks * K_SCALE_SIZE);

    stream->parallel_for(sycl::range<1>(nblocks), [=](sycl::id<1> idx) {
        const int64_t      ib = idx[0];
        const block_q4_K * x  = (const block_q4_K *) tmp + ib;
        uint8_t *          out = qs_out + ib * (QK_K / 2);
        // nibble pairs in k order: byte p = elem 2p | elem 2p+1 << 4
        for (int p = 0; p < QK_K / 2; ++p) {
            const uint8_t v0 = dnn_u4_q4k_elem(x->qs, 2 * p);
            const uint8_t v1 = dnn_u4_q4k_elem(x->qs, 2 * p + 1);
            out[p] = (uint8_t) (v0 | (v1 << 4));
        }
        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            sc_out[ib * K_SCALE_SIZE + j] = x->scales[j];
        }
        dm_out[ib] = x->dm;
        // f16 group scales, [g][n] with n innermost
        const int64_t n = ib / KB;
        const int64_t b = ib % KB;
        const float   d = (float) x->dm.x();
        for (int j = 0; j < 8; ++j) {
            uint8_t sc, mn;
            dnn_u4_scale_min(j, x->scales, sc, mn);
            sc_dev[(b * 8 + j) * N + n] = (sycl::half) (d * (float) sc);
            mn_dev[(b * 8 + j) * N + n] = mn;
        }
        ((sycl::half *) (mn_dev + (size_t) N * KB * 8))[b * N + n] = x->dm.y();
    }).wait();

    ggml_sycl_free_device(tmp, *stream);

    extra->dnn_scales               = sc_dev;
    extra->dnn_mins                 = mn_dev;
    extra->dnn_scales_device        = ctx.device;
    extra->optimized_feature.dnn_u4 = true;
    g_ggml_sycl_opt_epoch.fetch_add(1, std::memory_order_relaxed); // [hostv]
    return true;
}

// ---- matmul ----

namespace {

// activations prepared once per graph per src1: f16 copy for oneDNN and f32 group sums for the min term.
// The backend pool is a stack (LIFO frees), so these live in their own device buffers, recycled across graphs.
struct dnn_u4_buf {
    void * ptr = nullptr;
    size_t bytes = 0;
    bool   in_use = false;
};
std::vector<dnn_u4_buf> g_dnn_u4_bufs;

// per-graph buffers: handed out once per graph, all released at the next graph start after the queue drained
struct dnn_u4_act;
extern std::vector<dnn_u4_act> g_dnn_u4_acts;
static void dnn_u4_acts_invalidate();

// per-graph buffers with a budget: a 2048-token prefill graph prepares ~256 distinct activation sources of
// 20-70 MB each; unbounded, that is several GB per graph and oversubscribes the card once a 131k KV cache
// sits beside the model. Past the budget the queue is drained, every buffer is freed for reuse and the
// activation cache is invalidated (later ops in the graph re-prepare).
void * dnn_u4_buf_get(const queue_ptr & stream, size_t bytes) {
    static const size_t budget = (size_t) (getenv("GGML_SYCL_DNN_U4_BUF_MB") ? atoi(getenv("GGML_SYCL_DNN_U4_BUF_MB")) : 256) * 1024 * 1024;
    for (int pass = 0; pass < 2; ++pass) {
        for (auto & b : g_dnn_u4_bufs) {
            if (!b.in_use && b.bytes >= bytes) { b.in_use = true; return b.ptr; }
        }
        size_t total = 0;
        for (const auto & b : g_dnn_u4_bufs) { total += b.bytes; }
        if (total + bytes <= budget || pass == 1) {
            dnn_u4_buf b;
            b.bytes  = bytes + bytes / 8;
            b.ptr    = ggml_sycl_malloc_device(b.bytes, *stream);
            b.in_use = true;
            g_dnn_u4_bufs.push_back(b);
            return b.ptr;
        }
        stream->wait();
        for (auto & b : g_dnn_u4_bufs) { b.in_use = false; }
        dnn_u4_acts_invalidate();
    }
    return nullptr;
}

// transient slots: taken for the rest of the graph, or until the budget runs out, at which point the queue is
// drained and every slot is free again. Event status is NOT used to free a slot: on this in-order Level Zero
// queue kernel events may be discarded and report complete early (the NaN regression of 2026-09-25).
struct dnn_u4_slot {
    void *      ptr = nullptr;
    size_t      bytes = 0;
    sycl::event done;   // kept for callers that record it; not consulted for reuse
    bool        busy = false;
};
std::vector<dnn_u4_slot> g_dnn_u4_slots;   // reserved once: callers hold references across further gets
size_t g_dnn_u4_slot_bytes = 0;

dnn_u4_slot & dnn_u4_slot_get(const queue_ptr & stream, size_t bytes) {
    static const size_t budget = (size_t) (getenv("GGML_SYCL_DNN_U4_SLOT_MB") ? atoi(getenv("GGML_SYCL_DNN_U4_SLOT_MB")) : 128) * 1024 * 1024;
    if (g_dnn_u4_slots.capacity() < 4096) { g_dnn_u4_slots.reserve(4096); }
    for (int pass = 0; pass < 2; ++pass) {
        for (auto & sl : g_dnn_u4_slots) {
            if (!sl.busy && sl.bytes >= bytes) { sl.busy = true; return sl; }
        }
        if (g_dnn_u4_slot_bytes + bytes <= budget || g_dnn_u4_slots.empty()) {
            dnn_u4_slot sl;
            sl.bytes = bytes + bytes / 8;
            sl.ptr   = ggml_sycl_malloc_device(sl.bytes, *stream);
            sl.busy  = true;
            g_dnn_u4_slot_bytes += sl.bytes;
            g_dnn_u4_slots.push_back(sl);
            return g_dnn_u4_slots.back();
        }
        // budget exhausted: everything queued so far must finish before any slot is reused
        stream->wait();
        for (auto & sl : g_dnn_u4_slots) { sl.busy = false; }
    }
    dnn_u4_slot sl;   // unreachable in practice: allocate anyway
    sl.bytes = bytes + bytes / 8;
    sl.ptr   = ggml_sycl_malloc_device(sl.bytes, *stream);
    sl.busy  = true;
    g_dnn_u4_slot_bytes += sl.bytes;
    g_dnn_u4_slots.push_back(sl);
    return g_dnn_u4_slots.back();
}

struct dnn_u4_act {
    const ggml_tensor * src1 = nullptr;   // tensor identity: buffer addresses get reused within a graph
    const void * src1_data = nullptr;
    int64_t      M = 0, K = 0;
    sycl::half * f16 = nullptr;   // activations / rs[m], power-of-two row scale so f16 cannot overflow
    float *      xg  = nullptr;   // [M][K/32] group sums of the unscaled activations
    float *      rs  = nullptr;   // [M] row scale
};
std::vector<dnn_u4_act> g_dnn_u4_acts;
static void dnn_u4_acts_invalidate() { g_dnn_u4_acts.clear(); }
// oneDNN reads memory handles lazily, so every execution gets its own memory objects and they stay alive
// until the queue has drained at the next graph start
std::vector<dnnl::memory> g_dnn_u4_live_mems;

struct dnn_u4_prims {
    dnnl::matmul main;      // X f16 [M x K] . W u4 [K x N] (scales) -> dst f16/f32 [M x N]
    // a primitive object executed again while an earlier execution is still pending corrupts that execution
    // (per-execution internal state), so each call within a graph takes its own object, round-robin
    std::vector<dnnl::matmul> main_pool;
    size_t                    main_next = 0;
    bool         main_f16_dst = false;
    size_t       main_scratch = 0;
    dnnl::memory src_m, w_m, sc_m, d_m, scratch_m;   // persistent wrappers, pointers rebound per call
    void * scratch_buf = nullptr;
    void * staging_buf = nullptr;   // f16 dst staging for M <= 16 (in-order queue serialises its reuse)
    dnnl::matmul corr;      // Xg f16 [M x G] . NEGMIN f16 [G x N] -> dst f32 += (only for M > 16)
    size_t       corr_scratch = 0;
    dnnl::memory ca_m, cb_m, cc_m, cscratch_m;
};

struct dnn_u4_key {
    int64_t M, K, N;
    bool operator==(const dnn_u4_key & o) const { return M == o.M && K == o.K && N == o.N; }
};
struct dnn_u4_key_hash {
    size_t operator()(const dnn_u4_key & k) const {
        return std::hash<int64_t>()(k.M) ^ (std::hash<int64_t>()(k.K) << 1) ^ (std::hash<int64_t>()(k.N) << 2);
    }
};

std::unordered_map<dnn_u4_key, dnn_u4_prims, dnn_u4_key_hash> g_dnn_u4_prims;

dnn_u4_prims & dnn_u4_get_prims(ggml_backend_sycl_context & ctx, const queue_ptr & stream, int64_t M, int64_t K, int64_t N) {
    const dnn_u4_key key{ M, K, N };
    auto it = g_dnn_u4_prims.find(key);
    if (it != g_dnn_u4_prims.end()) {
        return it->second;
    }
    using namespace dnnl;
    auto eng = ctx.engine_dnnl(stream);
    dnn_u4_prims p;

    memory::desc src_md({ M, K }, memory::data_type::f16, memory::format_tag::ab);
    memory::desc w_md({ K, N }, memory::data_type::u4, memory::format_tag::ba);
    primitive_attr attr;
    attr.set_fpmath_mode(fpmath_mode::f16, true);
    attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), { 32, 1 }, memory::data_type::f16);
    attr.set_scratchpad_mode(scratchpad_mode::user);
    // f16 output by default: the f32-output primitive selects a slower kernel on this device (GGML_SYCL_DNN_U4_F32DST=1 to try it)
    static const bool want_f32 = getenv("GGML_SYCL_DNN_U4_F32DST") && atoi(getenv("GGML_SYCL_DNN_U4_F32DST")) != 0;
    matmul::primitive_desc pd;
    try {
        if (!want_f32 && M <= 16) { throw dnnl::error(dnnl_unimplemented, "f16 dst requested"); }   // large M: direct f32 output
        memory::desc dst_md({ M, N }, memory::data_type::f32, memory::format_tag::ab);
        pd = matmul::primitive_desc(eng, src_md, w_md, dst_md, attr);
    } catch (const dnnl::error &) {
        memory::desc dst_md({ M, N }, memory::data_type::f16, memory::format_tag::ab);
        pd = matmul::primitive_desc(eng, src_md, w_md, dst_md, attr);
        p.main_f16_dst = true;
    }
    p.main         = matmul(pd);
    p.main_scratch = pd.scratchpad_desc().get_size();
    // one object per shape: a pool was tested (1 vs 4 vs 512: identical speed) and each extra object costs a
    // kernel-object creation on first use of a new M - 512 of them made the first 2048-token prefill take 82 s
    static const int pool_n = getenv("GGML_SYCL_DNN_U4_POOL") ? atoi(getenv("GGML_SYCL_DNN_U4_POOL")) : 1;
    p.main_pool.reserve(pool_n);
    for (int i = 0; i < pool_n; ++i) { p.main_pool.emplace_back(pd); }   // primitive cache hits after the first
    p.src_m     = memory(src_md, eng, DNNL_MEMORY_NONE);
    p.w_m       = memory(w_md, eng, DNNL_MEMORY_NONE);
    p.sc_m      = memory({ { K / 32, N }, memory::data_type::f16, memory::format_tag::ab }, eng, DNNL_MEMORY_NONE);
    p.d_m       = memory(pd.dst_desc(), eng, DNNL_MEMORY_NONE);
    p.scratch_m = memory(pd.scratchpad_desc(), eng, DNNL_MEMORY_NONE);
    if (M > 16) {
        const int64_t G = K / 32;
        memory::desc a_md({ M, G }, memory::data_type::f16, memory::format_tag::ab);
        memory::desc b_md({ G, N }, memory::data_type::f16, memory::format_tag::ab);
        memory::desc c_md({ M, N }, memory::data_type::f32, memory::format_tag::ab);
        primitive_attr cattr;
        post_ops po;
        po.append_sum(1.0f);
        cattr.set_post_ops(po);
        cattr.set_scratchpad_mode(scratchpad_mode::user);
        matmul::primitive_desc cpd(eng, a_md, b_md, c_md, cattr);
        p.corr         = matmul(cpd);
        p.corr_scratch = cpd.scratchpad_desc().get_size();
        p.ca_m = memory(a_md, eng, DNNL_MEMORY_NONE);
        p.cb_m = memory(b_md, eng, DNNL_MEMORY_NONE);
        p.cc_m = memory(c_md, eng, DNNL_MEMORY_NONE);
        p.cscratch_m = memory(cpd.scratchpad_desc(), eng, DNNL_MEMORY_NONE);
    }
    return g_dnn_u4_prims.emplace(key, std::move(p)).first->second;
}

}  // namespace

void ggml_sycl_dnn_u4_graph_begin(const queue_ptr & stream) {
    if (!g_dnn_u4_live_mems.empty() || !g_dnn_u4_acts.empty() || !g_dnn_u4_slots.empty()) {
        stream->wait();   // previous graph's oneDNN executions must have consumed their memory objects
    }
    g_dnn_u4_live_mems.clear();
    g_dnn_u4_acts.clear();
    for (auto & b : g_dnn_u4_bufs) { b.in_use = false; }
    for (auto & sl : g_dnn_u4_slots) { sl.busy = false; }
}

static dnn_u4_act & dnn_u4_prepare_act(ggml_backend_sycl_context & ctx, const ggml_tensor * src1, const float * src1_ddf_i,
                                       int64_t M, int64_t K, const queue_ptr & stream) {
    for (auto & a : g_dnn_u4_acts) {
        if (a.src1 == src1 && a.src1_data == src1_ddf_i && a.M == M && a.K == K) {
            return a;
        }
    }
    dnn_u4_act a;
    a.src1 = src1; a.src1_data = src1_ddf_i; a.M = M; a.K = K;
    a.f16 = (sycl::half *) dnn_u4_buf_get(stream, (size_t) M * K * sizeof(sycl::half));
    a.xg  = (float *) dnn_u4_buf_get(stream, (size_t) M * (K / 32) * sizeof(float));
    a.rs  = (float *) dnn_u4_buf_get(stream, (size_t) M * sizeof(float));
    GGML_UNUSED(ctx);
    sycl::half * f16 = a.f16;
    float *      xg  = a.xg;
    float *      rs  = a.rs;
    const int64_t G  = K / 32;
    // one work-item per 32-element group: converts its 32 values to f16 and sums them (row scale is 1)
    stream->parallel_for(sycl::range<1>((size_t) M * G), [=](sycl::id<1> idx) {
        const int64_t i = idx[0];
        const int64_t m = i / G;
        const float * x = src1_ddf_i + m * K + (i % G) * 32;
        sycl::half *  y = f16 + m * K + (i % G) * 32;
        float sum = 0.f;
        for (int k = 0; k < 32; ++k) { const float v = x[k]; y[k] = (sycl::half) v; sum += v; }
        xg[i] = sum;
        if (i % G == 0) { rs[m] = 1.f; }
    });
    g_dnn_u4_acts.push_back(std::move(a));
    return g_dnn_u4_acts.back();
}

// diagnostics (GGML_SYCL_DNN_U4_CHECK=1): count non-finite values in a device buffer
template <typename T>
static int64_t dnn_u4_count_nonfinite(const T * dev, size_t n, const queue_ptr & stream, float * absmax_out) {
    std::vector<T> h(n);
    stream->memcpy(h.data(), dev, n * sizeof(T)).wait();
    int64_t bad = 0; float mx = 0.f;
    for (size_t i = 0; i < n; ++i) {
        const float v = (float) h[i];
        if (!std::isfinite(v)) { bad++; } else { mx = std::max(mx, std::fabs(v)); }
    }
    *absmax_out = mx;
    return bad;
}

// stage timer (GGML_SYCL_DNN_U4_STAGES=1): waits after each stage of the op; totals printed at exit
struct dnn_u4_stage_acc { double prep = 0, host = 0, matmul = 0, epilogue = 0; int64_t calls = 0, preps = 0; };
static dnn_u4_stage_acc g_dnn_u4_stages;
static void dnn_u4_stages_print() {
    const auto & a = g_dnn_u4_stages;
    if (a.calls == 0) return;
    fprintf(stderr, "[dnn-u4 stages] calls %lld preps %lld | per call: host %.1f us, matmul %.1f us, epilogue %.1f us | per prep %.1f us\n",
            (long long) a.calls, (long long) a.preps, 1e3 * a.host / a.calls, 1e3 * a.matmul / a.calls, 1e3 * a.epilogue / a.calls,
            a.preps ? 1e3 * a.prep / a.preps : 0.0);
}
static bool dnn_u4_stages_on() {
    static int v = -1;
    if (v < 0) { v = (getenv("GGML_SYCL_DNN_U4_STAGES") && atoi(getenv("GGML_SYCL_DNN_U4_STAGES")) != 0) ? 1 : 0; if (v) std::atexit(dnn_u4_stages_print); }
    return v != 0;
}
static double dnn_u4_now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void ggml_sycl_op_mul_mat_dnn_u4(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                 ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                 const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low,
                                 const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_row_size,
                                 const queue_ptr & stream) try {
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
    using namespace dnnl;

    const int64_t K  = src0->ne[0];
    const int64_t N  = src0->ne[1];
    const int64_t M  = src1_ncols;
    const int64_t G  = K / 32;
    const int64_t KB = K / QK_K;
    GGML_ASSERT(row_low == 0 && row_high == N);
    GGML_ASSERT(src1->ne[0] == K);
    GGML_ASSERT(dst->ne[0] == N);

    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    GGML_ASSERT(extra && extra->optimized_feature.dnn_u4 && extra->dnn_scales && extra->dnn_mins);
    const uint8_t *    mins_u8 = (const uint8_t *) extra->dnn_mins;                       // [G][N]
    const sycl::half * dmin_h  = (const sycl::half *) (mins_u8 + (size_t) N * G);         // [KB][N]

    const uint8_t *     qs      = (const uint8_t *) src0_dd_i;
    const uint8_t *     sc_pack = qs + (size_t) N * KB * (QK_K / 2);
    const sycl::half2 * dm      = (const sycl::half2 *) (sc_pack + (size_t) N * KB * K_SCALE_SIZE);

    // activations -> f16 plus their 32-element group sums, shared across the graph
    const bool stages = dnn_u4_stages_on();
    double t0 = 0;
    if (stages) { stream->wait(); t0 = dnn_u4_now_ms(); }
    const size_t n_acts_before = g_dnn_u4_acts.size();
    const dnn_u4_act & act = dnn_u4_prepare_act(ctx, src1, src1_ddf_i, M, K, stream);
    if (stages) { stream->wait(); const double t1 = dnn_u4_now_ms(); if (g_dnn_u4_acts.size() != n_acts_before) { g_dnn_u4_stages.prep += t1 - t0; g_dnn_u4_stages.preps++; } t0 = t1; }

    auto & prims = dnn_u4_get_prims(ctx, stream, M, K, N);
    auto   eng   = ctx.engine_dnnl(stream);
    auto   st    = ctx.stream_dnnl(stream);

    static const int sync_mask = getenv("GGML_SYCL_DNN_U4_SYNC") ? atoi(getenv("GGML_SYCL_DNN_U4_SYNC")) : 0;   // diagnostics
    if (sync_mask & 1) { stream->wait(); }

    static const bool no_corr_flag = getenv("GGML_SYCL_DNN_U4_NOCORR") && atoi(getenv("GGML_SYCL_DNN_U4_NOCORR")) != 0;
    bool corr_done = false;
    static const bool plain = getenv("GGML_SYCL_DNN_U4_PLAIN") && atoi(getenv("GGML_SYCL_DNN_U4_PLAIN")) != 0;

    // main matmul: fresh memory objects per execution (kept alive in g_dnn_u4_live_mems), explicit event ordering,
    // every buffer oneDNN touches comes from event-guarded slots, never from the backend pool
    {
        if (prims.main_scratch > 0 && prims.scratch_buf == nullptr) { prims.scratch_buf = ggml_sycl_malloc_device(prims.main_scratch, *stream); }
        prims.src_m.set_data_handle((void *) act.f16);
        prims.w_m.set_data_handle((void *) qs);
        prims.sc_m.set_data_handle(extra->dnn_scales);
        prims.scratch_m.set_data_handle(prims.scratch_buf);
        std::unordered_map<int, memory> args = {
            { DNNL_ARG_SRC, prims.src_m }, { DNNL_ARG_WEIGHTS, prims.w_m },
            { DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, prims.sc_m }, { DNNL_ARG_SCRATCHPAD, prims.scratch_m } };
        sycl::event before = plain ? sycl::event() : stream->ext_oneapi_submit_barrier();
        dnnl::matmul & prim = prims.main_pool.empty() ? prims.main : prims.main_pool[prims.main_next++ % prims.main_pool.size()];
        const float * rs  = act.rs;
        float *       d32 = dst_dd_i;
        sycl::event   after;
        if (prims.main_f16_dst) {
            if (prims.staging_buf == nullptr) { prims.staging_buf = ggml_sycl_malloc_device((size_t) M * N * sizeof(sycl::half), *stream); }
            prims.d_m.set_data_handle(prims.staging_buf);
            args[DNNL_ARG_DST] = prims.d_m;
            if (stages) { g_dnn_u4_stages.host += dnn_u4_now_ms() - t0; }
            sycl::event ev = plain ? (prim.execute(st, args), sycl::event()) : sycl_interop::execute(prim, st, args, { before });
            if (stages) { const double th = dnn_u4_now_ms(); g_dnn_u4_stages.host += th - t0 - (th - t0) /* host only counted once above */; stream->wait(); g_dnn_u4_stages.matmul += dnn_u4_now_ms() - th; t0 = dnn_u4_now_ms(); }
            const sycl::half * d16 = (const sycl::half *) prims.staging_buf;
            if (M <= 16 && !no_corr_flag) {
                // fused epilogue: one work-item per output row n, consecutive rows on consecutive lanes so the min
                // bytes [g][n], the block dmin [b][n], the f16 matmul result and the f32 output are all coalesced.
                // The group sums xg [M][G] never touch local memory: per 256-block each lane loads one group's sums
                // (8 consecutive floats per m across lanes 0..7) and the eight values are shared with sub-group
                // broadcasts, so the inner loop is register-only.
                const float * xg_ptr = act.xg;
                const int     Mi     = (int) M;
                const int64_t Gi     = G;
                const int64_t KBi    = KB;
                const size_t  wg     = 256;
                const size_t  total  = ((size_t) N + wg - 1) / wg * wg;
                after = stream->submit([&](sycl::handler & h) {
                    if (!plain) { h.depends_on(ev); }
                    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(total), sycl::range<1>(wg)),
                                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                        auto          sg   = it.get_sub_group();
                        const int     lane = sg.get_local_id()[0];
                        const int64_t n0   = it.get_global_id(0);
                        const int64_t n    = n0 < N ? n0 : N - 1;   // keep every lane in the collectives; tail lanes do not store
                        float acc[16];
#pragma unroll
                        for (int m = 0; m < 16; ++m) { acc[m] = 0.f; }
                        for (int64_t b = 0; b < KBi; ++b) {
                            float xr[16];
                            const int64_t gl = b * 8 + (lane & 7);
#pragma unroll
                            for (int m = 0; m < 16; ++m) { xr[m] = m < Mi ? xg_ptr[m * Gi + gl] : 0.f; }
                            const float dmin = (float) dmin_h[b * N + n];
#pragma unroll
                            for (int jj = 0; jj < 8; ++jj) {
                                const float mf = dmin * (float) mins_u8[(b * 8 + jj) * N + n];
#pragma unroll
                                for (int m = 0; m < 16; ++m) {
                                    const float xg = sycl::select_from_group(sg, xr[m], jj);
                                    if (m < Mi) { acc[m] += mf * xg; }
                                }
                            }
                        }
                        if (n0 < N) {
#pragma unroll
                            for (int m = 0; m < 16; ++m) {
                                if (m < Mi) { d32[m * N + n0] = (float) d16[m * N + n0] - acc[m]; }
                            }
                        }
                    });
                });
                corr_done = true;
            } else {
                after = stream->submit([&](sycl::handler & h) {
                    if (!plain) { h.depends_on(ev); }
                    h.parallel_for(sycl::range<1>((size_t) M * N), [=](sycl::id<1> idx) { d32[idx[0]] = (float) d16[idx[0]] * rs[idx[0] / N]; });
                });
            }
            if (stages) { stream->wait(); g_dnn_u4_stages.epilogue += dnn_u4_now_ms() - t0; g_dnn_u4_stages.calls++; }
        } else {
            prims.d_m.set_data_handle(dst_dd_i);
            args[DNNL_ARG_DST] = prims.d_m;
            sycl::event ev = plain ? (prim.execute(st, args), sycl::event()) : sycl_interop::execute(prim, st, args, { before });
            after = stream->submit([&](sycl::handler & h) {
                if (!plain) { h.depends_on(ev); }
                h.parallel_for(sycl::range<1>((size_t) M * N), [=](sycl::id<1> idx) { d32[idx[0]] *= rs[idx[0] / N]; });
            });
        }
    }

    if (sync_mask & 2) { stream->wait(); }

    static const bool check = getenv("GGML_SYCL_DNN_U4_CHECK") && atoi(getenv("GGML_SYCL_DNN_U4_CHECK")) != 0;
    if (check) {
        static int reports = 0;
        float mx_x, mx_h, mx_s, mx_y;
        const int64_t bad_x = dnn_u4_count_nonfinite(src1_ddf_i, (size_t) M * K, stream, &mx_x);
        const int64_t bad_h = dnn_u4_count_nonfinite(act.f16, (size_t) M * K, stream, &mx_h);
        const int64_t bad_s = dnn_u4_count_nonfinite((const sycl::half *) extra->dnn_scales, (size_t) G * N, stream, &mx_s);
        const int64_t bad_y = dnn_u4_count_nonfinite(dst_dd_i, (size_t) M * N, stream, &mx_y);
        if ((bad_x || bad_h || bad_s || bad_y) && reports < 8) {
            reports++;
            GGML_LOG_INFO("[dnn-u4 check] %s N=%lld K=%lld M=%lld | x nonfinite %lld absmax %g | x16 nonfinite %lld absmax %g | scales nonfinite %lld absmax %g | y(main) nonfinite %lld absmax %g\n",
                          src0->name, (long long) N, (long long) K, (long long) M, (long long) bad_x, mx_x, (long long) bad_h, mx_h, (long long) bad_s, mx_s, (long long) bad_y, mx_y);
        }
    }

    // min correction: dst[m][n] -= sum_g (dmin*mn)[n][g] * xg[m][g]
    if (no_corr_flag || corr_done) {
    } else if (M <= 16) {
        const float * xg_ptr = act.xg;
        float *       out    = dst_dd_i;
        const int64_t Mi     = M;
        const size_t  wg     = 256;
        const size_t  total  = ((size_t) N * 16 + wg - 1) / wg * wg;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(total), sycl::range<1>(wg)),
                             [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto          sg   = it.get_sub_group();
            const int64_t n    = it.get_global_id(0) / 16;
            const int     lane = sg.get_local_id()[0];
            float acc[16];
            for (int m = 0; m < 16; ++m) { acc[m] = 0.f; }
            if (n < N) {
                for (int64_t b = lane; b < KB; b += 16) {
                    const uint8_t * sc   = sc_pack + (n * KB + b) * K_SCALE_SIZE;
                    const float     dmin = (float) dm[n * KB + b].y();
                    for (int j = 0; j < 8; ++j) {
                        uint8_t s, mn;
                        dnn_u4_scale_min(j, sc, s, mn);
                        const float   mf = dmin * (float) mn;
                        const int64_t g  = b * 8 + j;
                        for (int m = 0; m < Mi; ++m) { acc[m] += mf * xg_ptr[m * G + g]; }
                    }
                }
            }
            for (int m = 0; m < Mi; ++m) {
                const float v = sycl::reduce_over_group(sg, acc[m], sycl::plus<float>());
                if (lane == 0 && n < N) { out[m * N + n] -= v; }
            }
        });
    } else {
        // large M (prefill): materialise -mins as f16 [G][N] once per call and let oneDNN accumulate
        dnn_u4_slot & nm_slot = dnn_u4_slot_get(stream, (size_t) G * N * sizeof(sycl::half));
        dnn_u4_slot & xg_slot = dnn_u4_slot_get(stream, (size_t) M * G * sizeof(sycl::half));
        struct { sycl::half * p; sycl::half * get() const { return p; } } negmin{ (sycl::half *) nm_slot.ptr };
        struct { sycl::half * p; sycl::half * get() const { return p; } } xg16{ (sycl::half *) xg_slot.ptr };
        {
            sycl::half * nm = negmin.get();
            stream->parallel_for(sycl::range<1>((size_t) N * KB), [=](sycl::id<1> idx) {
                const int64_t   nb   = idx[0];
                const int64_t   n    = nb / KB, b = nb % KB;
                const uint8_t * sc   = sc_pack + nb * K_SCALE_SIZE;
                const float     dmin = (float) dm[nb].y();
                for (int j = 0; j < 8; ++j) {
                    uint8_t s, mn;
                    dnn_u4_scale_min(j, sc, s, mn);
                    nm[(b * 8 + j) * N + n] = (sycl::half) (-dmin * (float) mn * 256.f);   // x 2^8, undone in xg16
                }
            });
            const float * xg_ptr = act.xg;
            sycl::half *  x16    = xg16.get();
            stream->parallel_for(sycl::range<1>((size_t) M * G), [=](sycl::id<1> idx) { x16[idx[0]] = (sycl::half) (xg_ptr[idx[0]] * (1.f / 256.f)); });
        }
        memory ca_m(prims.ca_m.get_desc(), eng, xg16.get());
        memory cb_m(prims.cb_m.get_desc(), eng, negmin.get());
        memory cc_m(prims.cc_m.get_desc(), eng, dst_dd_i);
        dnn_u4_slot * cs_slot = prims.corr_scratch > 0 ? &dnn_u4_slot_get(stream, prims.corr_scratch) : nullptr;
        memory cscratch_m(prims.cscratch_m.get_desc(), eng, cs_slot ? cs_slot->ptr : nullptr);
        sycl::event before = stream->ext_oneapi_submit_barrier();
        sycl::event ev = sycl_interop::execute(prims.corr, st,
            { { DNNL_ARG_SRC, ca_m }, { DNNL_ARG_WEIGHTS, cb_m }, { DNNL_ARG_DST, cc_m }, { DNNL_ARG_SCRATCHPAD, cscratch_m } }, { before });
        g_dnn_u4_live_mems.insert(g_dnn_u4_live_mems.end(), { ca_m, cb_m, cc_m, cscratch_m });
        sycl::event done = stream->ext_oneapi_submit_barrier({ ev });   // later queue work waits for the correction
        nm_slot.done = done; xg_slot.done = done;
        if (cs_slot) { cs_slot->done = done; }
    }
    if (sync_mask & 4) { stream->wait(); }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    GGML_SYCL_EXIT_OR_RETHROW();
} catch (const dnnl::error & e) {
    GGML_LOG_ERROR("%s: oneDNN error: %s\n", __func__, e.what());
    GGML_SYCL_EXIT_OR_RETHROW();
}

#else  // !GGML_SYCL_DNNL

bool ggml_sycl_dnn_u4_prepare(ggml_backend_sycl_context &, const ggml_tensor *) { return false; }
void ggml_sycl_dnn_u4_graph_begin(const queue_ptr &) {}

void ggml_sycl_op_mul_mat_dnn_u4(ggml_backend_sycl_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *,
                                 const char *, const float *, const char *, float *, const int64_t, const int64_t,
                                 const int64_t, const int64_t, const queue_ptr &) {
    GGML_ABORT("oneDNN u4 path not built");
}

#endif
