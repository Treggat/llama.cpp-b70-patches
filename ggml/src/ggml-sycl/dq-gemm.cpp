//
// LOCAL (not for upstream): prefill dq-gemm dispatch target, see dq-gemm.hpp.
//
#include "dq-gemm.hpp"

#include "convert.hpp"
#include "fattn-xmx.hpp"
#include "gemm.hpp"
#include "mmq-dnn-u4.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(GGML_SYCL_DQ_GEMM_HAVE_FUSED) && GGML_SYCL_DQ_GEMM_HAVE_FUSED
#    include "dq-gemm-kernel/dq_gemm_q4k.hpp"
#    define DQ_GEMM_FUSED 1
#else
#    define DQ_GEMM_FUSED 0
#endif

extern int g_ggml_sycl_enable_dnn;

static int dq_gemm_env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e && *e ? atoi(e) : def;
}

bool ggml_sycl_dq_gemm_env() {
#if GGML_SYCL_XMX
    static const bool v = dq_gemm_env_int("GGML_SYCL_DQ_GEMM", 0) != 0;   // default off
    return v;
#else
    return false;
#endif
}

int ggml_sycl_dq_gemm_min_cols() {
    // 17 at the least: 1..16 columns are the decode / verify window (MMVQ, XMX q4_K 3..16, XMX q6_K 3..8).
    // Default 33: the dequantize is a fixed cost (~850 us on 17408x5120) whatever the column count, so the path wins
    // from 17 columns up, but ubatches <= GGML_SYCL_GRAPH_MAX_TOKENS (32) are recorded into SYCL graphs; starting at
    // 33 keeps dq-gemm eager-only (17..32 pass the graph self-test too, GGML_SYCL_DQ_GEMM_MIN_COLS=17 opts in)
    static const int v = std::max(17, dq_gemm_env_int("GGML_SYCL_DQ_GEMM_MIN_COLS", 33));
    return v;
}

int ggml_sycl_dq_gemm_max_cols() {
    // GGML_SYCL_DQ_GEMM_MAX_COLS (default 0 = no limit): above it the op stays on dequantize + oneDNN, for a kernel
    // that wins on short ubatches / tails but not (yet) on full 2048-token ones
    static const int v = std::max(0, dq_gemm_env_int("GGML_SYCL_DQ_GEMM_MAX_COLS", 0));
    return v;
}

static bool ggml_sycl_dq_gemm_q6k_env() {
    static const bool v = dq_gemm_env_int("GGML_SYCL_DQ_GEMM_Q6K", 0) != 0;
    return v;
}

// which kernel entry serves the op: 0 = reference (dequantize + oneDNN), 1 = fused kernel
enum dq_gemm_impl { DQ_GEMM_IMPL_REF = 0, DQ_GEMM_IMPL_FUSED = 1 };

static dq_gemm_impl ggml_sycl_dq_gemm_impl(int device, const ggml_tensor * src0, const ggml_tensor * src1,
                                           const ggml_tensor * dst) {
    // GGML_SYCL_DQ_GEMM_IMPL: "ref" forces the reference; default (or "fused"): the fused kernel whenever it is
    // compiled in and takes the shape, else the reference
    static const bool want_ref = [] {
        const char * e = getenv("GGML_SYCL_DQ_GEMM_IMPL");
        return e && strcmp(e, "ref") == 0;
    }();
    // GGML_SYCL_DQ_GEMM_FUSED_MAX_COLS (default 1536, 0 = no limit): above it the reference serves the op. The fused
    // kernel runs ~100 TFLOPS in steady state vs oneDNN's ~146 on the f16 scratch, so it wins while the fixed
    // dequantize cost dominates (short ubatches, ragged tails; measured up to 1536 columns) and is at parity or
    // behind from ~1792 (full 2048-token ubatches)
    static const int fused_max = std::max(0, dq_gemm_env_int("GGML_SYCL_DQ_GEMM_FUSED_MAX_COLS", 1536));
    if (want_ref || (fused_max > 0 && src1->ne[1] > fused_max)) {
        return DQ_GEMM_IMPL_REF;
    }
#if DQ_GEMM_FUSED
    const int64_t M   = src0->ne[1];
    const int64_t N   = src1->ne[1];
    const int64_t K   = src0->ne[0];
    const int64_t ldd = dst->nb[1] / sizeof(float);
    if (src0->type == GGML_TYPE_Q4_K && dq_gemm_q4k_supported(M, N, K) && ldd % 4 == 0 &&
        ((uintptr_t) dst->data) % 16 == 0 && ((uintptr_t) src0->data) % 16 == 0 &&
        ggml_sycl_fattn_xmx_device_ok(device)) {   // the same f16 matrix-hardware probe as the XMX FA path
        return DQ_GEMM_IMPL_FUSED;
    }
#else
    GGML_UNUSED(device);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
#endif
    return DQ_GEMM_IMPL_REF;
}

bool ggml_sycl_dq_gemm_can_use(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               const ggml_tensor * dst) {

    if (!ggml_sycl_dq_gemm_env()) {
        return false;
    }
    if (!(src0->type == GGML_TYPE_Q4_K || (src0->type == GGML_TYPE_Q6_K && ggml_sycl_dq_gemm_q6k_env()))) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->op != GGML_OP_MUL_MAT) {   // MUL_MAT_ID comes through the same dispatcher with a stack dst
        return false;
    }
    if (dst->op_params[0] != GGML_PREC_DEFAULT) {   // the f16 path the reference stands in for has the same rule
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src1->ne[1] < ggml_sycl_dq_gemm_min_cols() ||
        (ggml_sycl_dq_gemm_max_cols() > 0 && src1->ne[1] > ggml_sycl_dq_gemm_max_cols())) {
        return false;
    }
    if (src0->ne[0] % QK_K != 0 || src1->ne[0] != src0->ne[0] || dst->ne[0] != src0->ne[1] ||
        dst->ne[1] != src1->ne[1] || ggml_nbytes(src0) >= (size_t) INT_MAX) {
        return false;
    }
    if (g_ggml_sycl_prioritize_dmmv || ggml_sycl_dnn_u4_enabled()) {
        return false;
    }
#if !GGML_SYCL_DNNL
    // the reference entry is built on oneDNN
    if (ggml_sycl_dq_gemm_impl(ctx.device, src0, src1, dst) == DQ_GEMM_IMPL_REF) {
        return false;
    }
#else
    if (ggml_sycl_dq_gemm_impl(ctx.device, src0, src1, dst) == DQ_GEMM_IMPL_REF && !g_ggml_sycl_enable_dnn) {
        return false;
    }
#endif
    return true;
}

// f32 activations -> f16, N rows of K (the same round-to-nearest-even conversion ggml_sycl_op_mul_mat_sycl applies to
// src1). The generic convert_unary kernel does 4D index math per element and reaches ~330 GB/s; this contiguous
// float4 -> half4 kernel is used when the source is 16-byte aligned (GGML_SYCL_DQ_GEMM_FASTCVT=0 disables it).
static void dq_gemm_act_to_f16(ggml_backend_sycl_context & ctx, const ggml_tensor * src1, ggml_tensor * dst,
                               sycl::half * a16, const queue_ptr & stream) {
    GGML_UNUSED(ctx);
    static const bool fast = dq_gemm_env_int("GGML_SYCL_DQ_GEMM_FASTCVT", 1) != 0;
    const float *     x    = static_cast<const float *>(src1->data);
    const int64_t     n    = ggml_nelements(src1);
    if (fast && n % 4 == 0 && ((uintptr_t) x) % 16 == 0 && ((uintptr_t) a16) % 8 == 0) {
        const int64_t n4 = n / 4;
        const int64_t wg = 256;
        const auto *  xv = reinterpret_cast<const sycl::float4 *>(x);
        auto *        yv = reinterpret_cast<sycl::vec<sycl::half, 4> *>(a16);
        stream->parallel_for(sycl::nd_range<1>(((n4 + wg - 1) / wg) * wg, wg), [=](sycl::nd_item<1> it) {
            const int64_t i = it.get_global_id(0);
            if (i < n4) {
                yv[i] = xv[i].convert<sycl::half, sycl::rounding_mode::rte>();
            }
        });
        return;
    }
    const to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(src1->type, dst);
    GGML_ASSERT(to_fp16 != nullptr);
    to_fp16(static_cast<const float *>(src1->data), a16, ggml_nelements(src1), stream);
}

void ggml_sycl_mul_mat_dq_gemm(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst) try {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2, " : dq-gemm");

    const int64_t K = src0->ne[0];   // reduction
    const int64_t M = src0->ne[1];   // weight rows = output features
    const int64_t N = src1->ne[1];   // tokens
    GGML_ASSERT(src1->ne[0] == K && dst->ne[0] == M && dst->ne[1] == N);
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    GGML_ASSERT(extra && extra->optimized_feature.reorder);

    const queue_ptr stream = ctx.stream();
    ggml_sycl_phase_timer pt(stream);

    float *       out   = static_cast<float *>(dst->data);
    const int64_t ldd   = dst->nb[1] / sizeof(float);
    const bool    fused = ggml_sycl_dq_gemm_impl(ctx.device, src0, src1, dst) == DQ_GEMM_IMPL_FUSED;

    // fused kernel, ragged token tail (N % tile): the kernel's bounds-checked tail launch is slow, so by default
    // (GGML_SYCL_DQ_GEMM_PAD=1) the tail runs as one full, unchecked tile on zero-padded activations into a scratch,
    // whose first rows are then copied to dst (dst is contiguous: ldd == M, the rows are one block)
    static const int64_t tile = std::max(1, dq_gemm_env_int("GGML_SYCL_DQ_GEMM_TILE", 256));   // kernel's token tile
    static const bool    pad  = dq_gemm_env_int("GGML_SYCL_DQ_GEMM_PAD", 1) != 0;
    const int64_t n_full = (N / tile) * tile;
    const int64_t n_tail = N - n_full;
    const bool    padded = fused && pad && n_tail != 0 && ldd == M;

    ggml_sycl_pool_alloc<sycl::half> a16(ctx.pool(), (padded ? n_full + tile : N) * K);
    dq_gemm_act_to_f16(ctx, src1, dst, a16.get(), stream);
    if (padded) {
        stream->memset(a16.get() + N * K, 0, (size_t) (tile - n_tail) * K * sizeof(sycl::half));
    }
    pt.mark("act_f16");

    if (fused) {
#if DQ_GEMM_FUSED
        if (!padded) {
            dq_gemm_q4k(*stream, src0->data, a16.get(), out, M, N, K, ldd);
        } else {
            if (n_full > 0) {
                dq_gemm_q4k(*stream, src0->data, a16.get(), out, M, n_full, K, ldd);
            }
            ggml_sycl_pool_alloc<float> tail(ctx.pool(), tile * M);
            dq_gemm_q4k(*stream, src0->data, a16.get() + n_full * K, tail.get(), M, tile, K, M);
            stream->memcpy(out + n_full * ldd, tail.get(), (size_t) n_tail * M * sizeof(float));
        }
        pt.mark("fused", "fused_host");
        return;
#endif
    }

#if GGML_SYCL_DNNL
    // reference: the existing pieces (reorder-aware dequantize to an f16 scratch, oneDNN f16 GEMM)
    GGML_ASSERT(ldd == M);
    const to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(src0->type, dst);   // picks the _reorder variant
    GGML_ASSERT(to_fp16 != nullptr);
    ggml_sycl_pool_alloc<sycl::half> w16(ctx.pool(), M * K);
    to_fp16(src0->data, w16.get(), M * K, stream);
    pt.mark("dequant");
    DnnlGemmWrapper::row_gemm(ctx, (int) M, (int) N, (int) K, w16.get(), DnnlGemmWrapper::to_dt<sycl::half>(),
                              a16.get(), DnnlGemmWrapper::to_dt<sycl::half>(), out,
                              DnnlGemmWrapper::to_dt<float>(), stream);
    pt.mark("gemm", "gemm_host");
#else
    GGML_ABORT("dq-gemm reference entry needs oneDNN");
#endif
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    GGML_SYCL_EXIT_OR_RETHROW();
}
