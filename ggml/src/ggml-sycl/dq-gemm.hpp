//
// LOCAL (not for upstream): prefill mul_mat for q4_K (and q6_K) weights in the MMVQ reorder layout, with the
// dequantization fused into an f16 DPAS GEMM ("dq-gemm").
//
// Today's prefill path (ggml_sycl_op_mul_mat_sycl) dequantizes the whole weight to an f16 N x K scratch buffer
// (17408x5120 -> 178 MB written, then read back by the GEMM), converts the f32 activations to f16 and runs the
// oneDNN f16 GEMM. dq-gemm replaces the three steps with one kernel entry that reads the quantized weights in place.
//
// Dispatch (ggml_sycl_mul_mat): q4_K (q6_K with GGML_SYCL_DQ_GEMM_Q6K=1) x f32 -> f32, 2D contiguous, src1 columns
// >= GGML_SYCL_DQ_GEMM_MIN_COLS (default 33 = above GGML_SYCL_GRAPH_MAX_TOKENS, so never graph-recorded; never below
// 17: the decode / verify window <= 16 belongs to the XMX MMVQ-replacement paths) and <= GGML_SYCL_DQ_GEMM_MAX_COLS
// (default 0 = no limit), weights reordered by the caller (the same reorder MMVQ installs at decode).
// Fused kernel, ragged token tails: run as one padded full tile into a scratch (GGML_SYCL_DQ_GEMM_PAD=1, default).
// Runtime switch GGML_SYCL_DQ_GEMM=1 (default 0). Built only with -DGGML_SYCL_XMX=ON.
//
// The kernel entry is pluggable (GGML_SYCL_DQ_GEMM_IMPL): "ref" is a reference implementation made of the existing
// pieces (reorder dequantize -> f16 scratch, f32 -> f16 activations, oneDNN f16 GEMM), "fused" is the fused kernel
// when it is compiled in (GGML_SYCL_DQ_GEMM_HAVE_FUSED, dq-gemm-kernel/). Default: fused when present and it takes
// the shape and N <= GGML_SYCL_DQ_GEMM_FUSED_MAX_COLS (default 1536, 0 = no limit), else ref. Both entries convert the
// activations with a contiguous vectorised f32 -> f16 kernel (GGML_SYCL_DQ_GEMM_FASTCVT=1, default).
//
// Prefill stays eager: a ubatch above GGML_SYCL_GRAPH_MAX_TOKENS (32) is never recorded into a SYCL graph, and the
// path makes no host synchronisation of its own.
//
#ifndef GGML_SYCL_DQ_GEMM_HPP
#define GGML_SYCL_DQ_GEMM_HPP

#include "common.hpp"

#include <chrono>

// value of GGML_SYCL_DQ_GEMM (default 0), read once; false when not built with -DGGML_SYCL_XMX=ON
bool ggml_sycl_dq_gemm_env();

// GGML_SYCL_DQ_GEMM_MIN_COLS (default 33, clamped to >= 17)
int ggml_sycl_dq_gemm_min_cols();

// GGML_SYCL_DQ_GEMM_MAX_COLS (default 0 = no upper limit)
int ggml_sycl_dq_gemm_max_cols();

// true when src0 x src1 -> dst is a prefill mul_mat this path serves (type, shape, column count, env); no side
// effects. The caller must still install the reorder layout on src0 and check it took.
bool ggml_sycl_dq_gemm_can_use(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               const ggml_tensor * dst);

// the whole mul_mat on ctx.stream() for a src0 already in the reorder layout
void ggml_sycl_mul_mat_dq_gemm(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst);

// ---- sub-phase profiling hooks for GGML_SYCL_OP_PROFILE (defined in ggml-sycl.cpp) ----
// true while the per-op profiler is timing the current op
bool ggml_sycl_op_prof_active();
// adds `ms` under "<current op key>/<sub>" (not counted into the graph total)
void ggml_sycl_op_prof_sub_add(const char * sub, double ms);

// waits on the stream at each mark() and books the time since the previous mark to a sub-phase of the current op;
// no-op (no waits) unless the profiler is active
struct ggml_sycl_phase_timer {
    dpct::queue_ptr                       q;
    bool                                  on;
    std::chrono::steady_clock::time_point t;

    explicit ggml_sycl_phase_timer(dpct::queue_ptr q) : q(q), on(ggml_sycl_op_prof_active()) {
        if (on) {
            q->wait();
            t = std::chrono::steady_clock::now();
        }
    }

    // host_sub != nullptr: also book the host time until mark() was called (submission cost) under host_sub
    void mark(const char * sub, const char * host_sub = nullptr) {
        if (!on) {
            return;
        }
        const auto th = std::chrono::steady_clock::now();
        q->wait();
        const auto t1 = std::chrono::steady_clock::now();
        if (host_sub) {
            ggml_sycl_op_prof_sub_add(host_sub, std::chrono::duration<double, std::milli>(th - t).count());
        }
        ggml_sycl_op_prof_sub_add(sub, std::chrono::duration<double, std::milli>(t1 - t).count());
        t = std::chrono::steady_clock::now();
    }
};

#endif // GGML_SYCL_DQ_GEMM_HPP
