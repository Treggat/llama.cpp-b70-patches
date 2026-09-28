//
// q8_0 x q8_1 matmul on the Xe2 matrix units - see mmq-xmx-q80.hpp (LOCAL)
//
// Kernel: the q4_K kernel of mmq-xmx-q4k.cpp (VAR 12 scheme) with the nibble unpack and the q4_K scale math removed:
// q8_0 weights are already s8, one half scale per 32-group. Two launches: the fused f32 -> operands quantizer (MMVQ's
// quantize_q8_1_impl, as the q4_K path) and the matmul, both 256-GRF.
//
#include "mmq-xmx-q80.hpp"
#include "mmq-xmx-q4k.hpp"
#include "quants.hpp"
#include "quantize.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <iostream>

#if GGML_SYCL_XMX
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#endif

static int ggml_sycl_xmx_q80_env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e && *e ? atoi(e) : def;
}

static bool ggml_sycl_xmx_q80_env() {
    static const bool v = ggml_sycl_xmx_q80_env_int("GGML_SYCL_XMX_Q80", 0) != 0;   // default off
    return v;
}

static constexpr int XMX_Q80_ROWS     = 16;
static constexpr int XMX_Q80_MAX_COLS = 8;    // the one-tile kernel
static constexpr int XMX_Q80_WIDE_COLS = 16;  // LOCAL (longdraft): the two-tile kernel, 9..16 columns (opt-in)

bool ggml_sycl_xmx_wide() {
    // GGML_SYCL_XMX_WIDE=1 (LOCAL longdraft, default off): the XMX q6_K / q8_0 matmuls and the XMX flash attention take
    // 9..16 columns (speculative verify batches of up to 16 tokens) instead of stopping at 8; 1..8 unchanged
    static const bool v = ggml_sycl_xmx_q80_env_int("GGML_SYCL_XMX_WIDE", 0) != 0;
    return v;
}

bool ggml_sycl_xmx_direct_wide() {
    // LOCAL (wideverify): with GGML_SYCL_XMX_WIDE=1, the register-fed q4_K / q6_K kernels (GGML_SYCL_XMX_Q4K_DIRECT /
    // GGML_SYCL_XMX_Q6K_DIRECT) also serve 9..16 columns (two A tiles per staged B operand, bit-identical to the
    // two-tile joint_matrix kernels at the same split-K). GGML_SYCL_XMX_DIRECT_WIDE=0 keeps 9..16 on the joint_matrix
    // kernels (for A/B). Without WIDE this is always off.
    static const bool v = ggml_sycl_xmx_wide() && ggml_sycl_xmx_q80_env_int("GGML_SYCL_XMX_DIRECT_WIDE", 1) != 0;
    return v;
}
static constexpr int XMX_Q80_SB       = 256;   // elements per staged super-block (2 parts x 4 groups)

bool ggml_sycl_xmx_q80_can_use(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               const ggml_tensor * dst) {
    if (!ggml_sycl_xmx_q4k_built() || !ggml_sycl_xmx_q80_env()) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        dst->op != GGML_OP_MUL_MAT) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    static const int min_cols = std::max(1, ggml_sycl_xmx_q80_env_int("GGML_SYCL_XMX_Q80_MIN_COLS", 1));
    static const int max_cols = std::min(XMX_Q80_WIDE_COLS,
                                         ggml_sycl_xmx_q80_env_int("GGML_SYCL_XMX_Q80_MAX_COLS", ggml_sycl_xmx_wide() ? 16 : 8));
    if (src1->ne[1] < min_cols || src1->ne[1] > max_cols) {
        return false;
    }
    if (src0->ne[0] % XMX_Q80_SB != 0 || src0->ne[1] % XMX_Q80_ROWS != 0 || src1->ne[0] != src0->ne[0] ||
        dst->ne[0] != src0->ne[1] || ggml_nbytes(src0) >= (size_t) INT_MAX) {   // reorder offset helpers use int
        return false;
    }
    static const size_t min_bytes = (size_t) std::max(0, ggml_sycl_xmx_q80_env_int("GGML_SYCL_XMX_Q80_MIN_MB", 64)) << 20;
    if (src1->ne[1] <= 8 && ggml_nbytes(src0) < min_bytes) {   // above 8 columns MMVQ is out of reach
        return false;
    }
    if (g_ggml_sycl_prioritize_dmmv || src0->extra == nullptr || src0->buffer == nullptr) {
        return false;
    }
    return ggml_sycl_xmx_q4k_device_ok(ctx.device);   // the same s8 8x16x32 DPAS combination
}

#if GGML_SYCL_XMX

namespace mx  = sycl::ext::oneapi::experimental::matrix;
namespace imx = sycl::ext::intel::experimental::matrix;

namespace {

constexpr int X8_TM = 8, X8_TN = 16, X8_TK = 32, X8_SG = 16;
constexpr int X8_PB = 128;   // SLM bytes per row per staged part (4 groups)

// fused f32 -> operands quantizer: the q4_K path's xmx_q4k_quant_act without the integer group sums.
//   xa [8][K] int8 (column c's quants contiguous), d8 [8][G] half (the d MMVQ sees). Columns >= M unwritten: they only
//   feed accumulator rows that are never stored (integer DPAS, per-column rows).
sycl::event xmx_q80_quant_act(const float * x, int8_t * xa, sycl::half * d8, int K, int M, sycl::queue & q) {
    constexpr int EPW = QK8_1 / WARP_SIZE;
    const int G = K / QK8_1;
    return q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) M * G * WARP_SIZE), sycl::range<1>(WARP_SIZE)),
                          sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        sycl::vec<int8_t, EPW> qv;
        float d = 0.0f, sum = 0.0f;
        quantize_q8_1_impl<EPW>(x, qv, d, sum, it);
        const size_t blk  = it.get_group(0);   // c * G + g
        const int    lane = (int) it.get_local_id(0);
        *reinterpret_cast<sycl::vec<int8_t, EPW> *>(xa + blk * QK8_1 + (size_t) lane * EPW) = qv;
        if (lane == 0) {
            d8[blk] = sycl::half(d);
        }
    });
}

// The matmul. Work-group = ks sub-groups on the same 16 rows (split-K over 256-element super-blocks, SLM reduction
// in a fixed order); per super-block two staged parts of 4 groups (128 B per row), the next part prefetched into
// registers while the current one runs. Per group g, column i (accumulator element i), lane n (row n):
//   F[i] += (float) dot(q, u) * (d_w[n][g] * d8[i][g])
//   qs [N][K] int8, dd [N*K/32] half: the q8_0 reorder layout; out: column i of row r at out[i*ldd + r], i < M
sycl::event xmx_q80_launch(const int8_t * dQs, const uint16_t * dDd, const int8_t * dXa, const sycl::half * dD8,
                           float * dOut, int N, int K, int M, int64_t ldd, int ks, sycl::queue & q) {
    const int KB = K / XMX_Q80_SB, G = K / QK8_0;
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<int8_t, 1> bslm(sycl::range<1>(ks * 16 * X8_PB), h);
        sycl::local_accessor<float, 1>  red(sycl::range<1>(ks > 1 ? ks * 8 * 16 : 1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (N / 16) * ks * X8_SG), sycl::range<1>(ks * X8_SG)),
                       sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(X8_SG)]] {
            using sycl::access::address_space;
            using sycl::access::decorated;
            sycl::sub_group sg = it.get_sub_group();
            const int lane = sg.get_local_id()[0];
            const int s    = sg.get_group_id()[0];
            const int row0 = (int) it.get_group(0) * 16;
            const int b0   = s * KB / ks, b1 = (s + 1) * KB / ks;
            auto bp = bslm.get_multi_ptr<decorated::no>() + s * 16 * X8_PB;

            // staging: 16 B pieces, 8 lanes per row part (128 B), 2 rows per load instruction, 8 instructions
            const int pp = lane % 8, rr = lane / 8;
            const int8_t * gb = dQs + (size_t) (row0 + rr) * K + pp * 16;
            const int so = rr * X8_PB + pp * 16;
            const sycl::uint4 * hdw = reinterpret_cast<const sycl::uint4 *>(dDd + (size_t) (row0 + lane) * G);   // 8 halves / super-block
            const sycl::uint4 * d8b = reinterpret_cast<const sycl::uint4 *>(dD8 + (size_t) (lane & 7) * G);      // column lane&7

            auto ldq = [](const int8_t * p) -> sycl::uint4 { return *reinterpret_cast<const sycl::uint4 *>(p); };
            sycl::uint4 v[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) { v[j] = ldq(gb + (size_t) 2 * j * K + (size_t) b0 * XMX_Q80_SB); }
            sycl::uint4 hw = hdw[b0], hc = d8b[b0];

            float F[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) { F[i] = 0.f; }
            for (int b = b0; b < b1; ++b) {
                float dw[8], dc[8];   // this lane's row scales and column (lane & 7)'s d8 for the 8 groups
                {
                    const uint32_t a[4] = { hw.x(), hw.y(), hw.z(), hw.w() };
                    const uint32_t c[4] = { hc.x(), hc.y(), hc.z(), hc.w() };
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        dw[2 * j]     = (float) sycl::bit_cast<sycl::half>((uint16_t) (a[j] & 0xFFFF));
                        dw[2 * j + 1] = (float) sycl::bit_cast<sycl::half>((uint16_t) (a[j] >> 16));
                        dc[2 * j]     = (float) sycl::bit_cast<sycl::half>((uint16_t) (c[j] & 0xFFFF));
                        dc[2 * j + 1] = (float) sycl::bit_cast<sycl::half>((uint16_t) (c[j] >> 16));
                    }
                    const int bn = b + 1 < b1 ? b + 1 : b;   // next header (clamped, unconditional)
                    hw = hdw[bn];
                    hc = d8b[bn];
                }
#pragma unroll
                for (int part = 0; part < 2; ++part) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        *reinterpret_cast<sycl::uint4 *>(&bp[so + 2 * j * X8_PB]) = v[j];
                    }
                    {   // prefetch the next part (clamped, unconditional)
                        const int nb = part == 0 ? b : (b + 1 < b1 ? b + 1 : b);
                        const int np = part == 0 ? 1 : (b + 1 < b1 ? 0 : 1);
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            v[j] = ldq(gb + (size_t) 2 * j * K + (size_t) nb * XMX_Q80_SB + np * X8_PB);
                        }
                    }
                    sycl::group_barrier(sg);
#pragma unroll
                    for (int gg = 0; gg < 4; ++gg) {
                        const int g = part * 4 + gg;
                        mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, X8_TK, X8_TN, mx::layout::col_major> B;
                        mx::joint_matrix_load(sg, B, bp + gg * X8_TK, X8_PB);
                        mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, X8_TM, X8_TK, mx::layout::row_major> A;
                        mx::joint_matrix_load(sg, A,
                            sycl::address_space_cast<address_space::global_space, decorated::no>(
                                dXa + (size_t) b * XMX_Q80_SB + g * X8_TK), K);
                        mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, X8_TM, X8_TN> C;
                        mx::joint_matrix_fill(sg, C, 0);
                        mx::joint_matrix_mad(sg, C, A, B, C);
                        // column factors from lane i (register regions); outside the joint_matrix_apply lambda
                        float sb[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) { sb[i] = dw[g] * sycl::select_from_group(sg, dc[g], i); }
                        int i = 0;
                        imx::joint_matrix_apply(sg, C, [&](int32_t & dot, size_t, size_t) {
                            if (i < 8) { F[i] = sycl::fma((float) dot, sb[i], F[i]); }
                            ++i;
                        });
                    }
                    sycl::group_barrier(sg);
                }
            }
            if (ks > 1) {
#pragma unroll
                for (int i = 0; i < 8; ++i) { red[(s * 8 + i) * 16 + lane] = F[i]; }
                sycl::group_barrier(it.get_group());
                if (s == 0) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        float acc = red[i * 16 + lane];
                        for (int t = 1; t < ks; ++t) { acc += red[(t * 8 + i) * 16 + lane]; }
                        F[i] = acc;
                    }
                }
            }
            if (s == 0) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (i < M) { dOut[(size_t) i * ldd + row0 + lane] = F[i]; }
                }
            }
        });
    });
}

// LOCAL (longdraft): the kernel above as a template on the column count, instantiated only for MC = 16 (two A tiles,
// columns 0..7 and 8..15, on every staged B tile; xa [16][K], d8 [16][G]); per column the float operations and their
// order are the one-tile kernel's. 1..8 columns keep the unmodified kernel above.
template <int MC>
sycl::event xmx_q80_launch_wide(const int8_t * dQs, const uint16_t * dDd, const int8_t * dXa, const sycl::half * dD8,
                                float * dOut, int N, int K, int M, int64_t ldd, int ks, sycl::queue & q) {
    static_assert(MC == 16, "two A tiles");
    constexpr int NT = MC / X8_TM;
    const int KB = K / XMX_Q80_SB, G = K / QK8_0;
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<int8_t, 1> bslm(sycl::range<1>(ks * 16 * X8_PB), h);
        sycl::local_accessor<float, 1>  red(sycl::range<1>(ks > 1 ? ks * MC * 16 : 1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (N / 16) * ks * X8_SG), sycl::range<1>(ks * X8_SG)),
                       sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(X8_SG)]] {
            using sycl::access::address_space;
            using sycl::access::decorated;
            sycl::sub_group sg = it.get_sub_group();
            const int lane = sg.get_local_id()[0];
            const int s    = sg.get_group_id()[0];
            const int row0 = (int) it.get_group(0) * 16;
            const int b0   = s * KB / ks, b1 = (s + 1) * KB / ks;
            auto bp = bslm.get_multi_ptr<decorated::no>() + s * 16 * X8_PB;

            const int pp = lane % 8, rr = lane / 8;
            const int8_t * gb = dQs + (size_t) (row0 + rr) * K + pp * 16;
            const int so = rr * X8_PB + pp * 16;
            const sycl::uint4 * hdw = reinterpret_cast<const sycl::uint4 *>(dDd + (size_t) (row0 + lane) * G);   // 8 halves / super-block
            const sycl::uint4 * d8b = reinterpret_cast<const sycl::uint4 *>(dD8 + (size_t) lane * G);            // column lane

            auto ldq = [](const int8_t * p) -> sycl::uint4 { return *reinterpret_cast<const sycl::uint4 *>(p); };
            sycl::uint4 v[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) { v[j] = ldq(gb + (size_t) 2 * j * K + (size_t) b0 * XMX_Q80_SB); }
            sycl::uint4 hw = hdw[b0], hc = d8b[b0];

            float F[MC];
#pragma unroll
            for (int i = 0; i < MC; ++i) { F[i] = 0.f; }
            for (int b = b0; b < b1; ++b) {
                float dw[8], dc[8];   // this lane's row scales and column `lane`'s d8 for the 8 groups
                {
                    const uint32_t a[4] = { hw.x(), hw.y(), hw.z(), hw.w() };
                    const uint32_t c[4] = { hc.x(), hc.y(), hc.z(), hc.w() };
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        dw[2 * j]     = (float) sycl::bit_cast<sycl::half>((uint16_t) (a[j] & 0xFFFF));
                        dw[2 * j + 1] = (float) sycl::bit_cast<sycl::half>((uint16_t) (a[j] >> 16));
                        dc[2 * j]     = (float) sycl::bit_cast<sycl::half>((uint16_t) (c[j] & 0xFFFF));
                        dc[2 * j + 1] = (float) sycl::bit_cast<sycl::half>((uint16_t) (c[j] >> 16));
                    }
                    const int bn = b + 1 < b1 ? b + 1 : b;   // next header (clamped, unconditional)
                    hw = hdw[bn];
                    hc = d8b[bn];
                }
#pragma unroll
                for (int part = 0; part < 2; ++part) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        *reinterpret_cast<sycl::uint4 *>(&bp[so + 2 * j * X8_PB]) = v[j];
                    }
                    {   // prefetch the next part (clamped, unconditional)
                        const int nb = part == 0 ? b : (b + 1 < b1 ? b + 1 : b);
                        const int np = part == 0 ? 1 : (b + 1 < b1 ? 0 : 1);
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            v[j] = ldq(gb + (size_t) 2 * j * K + (size_t) nb * XMX_Q80_SB + np * X8_PB);
                        }
                    }
                    sycl::group_barrier(sg);
#pragma unroll
                    for (int gg = 0; gg < 4; ++gg) {
                        const int g = part * 4 + gg;
                        mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, X8_TK, X8_TN, mx::layout::col_major> B;
                        mx::joint_matrix_load(sg, B, bp + gg * X8_TK, X8_PB);
#pragma unroll
                        for (int t = 0; t < NT; ++t) {
                            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, X8_TM, X8_TK, mx::layout::row_major> A;
                            mx::joint_matrix_load(sg, A,
                                sycl::address_space_cast<address_space::global_space, decorated::no>(
                                    dXa + (size_t) t * X8_TM * K + (size_t) b * XMX_Q80_SB + g * X8_TK), K);
                            mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, X8_TM, X8_TN> C;
                            mx::joint_matrix_fill(sg, C, 0);
                            mx::joint_matrix_mad(sg, C, A, B, C);
                            float sb[8];
#pragma unroll
                            for (int i = 0; i < 8; ++i) { sb[i] = dw[g] * sycl::select_from_group(sg, dc[g], t * 8 + i); }
                            int i = 0;
                            imx::joint_matrix_apply(sg, C, [&](int32_t & dot, size_t, size_t) {
                                if (i < 8) { F[t * 8 + i] = sycl::fma((float) dot, sb[i], F[t * 8 + i]); }
                                ++i;
                            });
                        }
                    }
                    sycl::group_barrier(sg);
                }
            }
            if (ks > 1) {
#pragma unroll
                for (int i = 0; i < MC; ++i) { red[(s * MC + i) * 16 + lane] = F[i]; }
                sycl::group_barrier(it.get_group());
                if (s == 0) {
#pragma unroll
                    for (int i = 0; i < MC; ++i) {
                        float acc = red[i * 16 + lane];
                        for (int t = 1; t < ks; ++t) { acc += red[(t * MC + i) * 16 + lane]; }
                        F[i] = acc;
                    }
                }
            }
            if (s == 0) {
#pragma unroll
                for (int i = 0; i < MC; ++i) {
                    if (i < M) { dOut[(size_t) i * ldd + row0 + lane] = F[i]; }
                }
            }
        });
    });
}

int xmx_q80_pick_ks(int N, int K) {
    // 2 super-blocks per sub-group, at most 10 sub-groups per work-group. Measured on the B70 (test-backend-ops perf,
    // 248320 x 5120, us at 1 / 8 columns): ks 1 2498 / 2537, 2 - / 2369, 4 - / 2315, 5 2250 / 2308, 8 - / 2304,
    // 10 2248 / 2301, 16 2377 / 2411; MMVQ 2499 / 2952 (= 587 GB/s at ks 10 vs 458 GB/s at 8 columns)
    GGML_UNUSED(N);
    const int KB = K / XMX_Q80_SB;
    int ks = std::max(1, std::min(KB / 2, 10));
    static const int ks_env = ggml_sycl_xmx_q80_env_int("GGML_SYCL_XMX_Q80_KS", 0);
    if (ks_env > 0) {
        ks = ks_env;
    }
    return std::max(1, std::min(ks, std::min(16, KB)));
}

} // namespace

void ggml_sycl_mul_mat_xmx_q80(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst) try {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2, " : XMX q8_0");

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const int64_t G = K / QK8_1;
    GGML_ASSERT(src1->ne[0] == K && dst->ne[0] == N && dst->ne[1] == M);
    GGML_ASSERT(K % XMX_Q80_SB == 0 && N % XMX_Q80_ROWS == 0 && M >= 1 && M <= XMX_Q80_WIDE_COLS);
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    GGML_ASSERT(extra && extra->optimized_feature.reorder);

    // q8_0 reorder layout (block_q_t<GGML_TYPE_Q8_0>): qs [N*K] int8, then d [N*K/32] half
    using q80_reordered = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q8_0>;
    const char * base = static_cast<const char *>(src0->data);
    const auto   d_off = q80_reordered::get_d_offset((int) N, (int) K, 0);
    const int8_t *   qs = reinterpret_cast<const int8_t *>(base);
    const uint16_t * dd = reinterpret_cast<const uint16_t *>(base + d_off.first);

    if (M > XMX_Q80_MAX_COLS) {
        // LOCAL (longdraft): 9..16 columns on the two-tile kernel
        const size_t xa16 = (size_t) XMX_Q80_WIDE_COLS * K;
        const size_t d816 = (size_t) XMX_Q80_WIDE_COLS * G * sizeof(sycl::half);
        ggml_sycl_pool_alloc<char> act16(ctx.pool(), xa16 + d816);
        int8_t *     xa = reinterpret_cast<int8_t *>(act16.get());
        sycl::half * d8 = reinterpret_cast<sycl::half *>(act16.get() + xa16);
        sycl::queue & q = *ctx.stream();
        xmx_q80_quant_act(static_cast<const float *>(src1->data), xa, d8, (int) K, (int) M, q);
        static const int ks16 = ggml_sycl_xmx_q80_env_int("GGML_SYCL_XMX_Q80_KS16", 0);
        const int ks = ks16 > 0 ? std::max(1, std::min(ks16, std::min(16, (int) (K / XMX_Q80_SB)))) : xmx_q80_pick_ks((int) N, (int) K);
        xmx_q80_launch_wide<XMX_Q80_WIDE_COLS>(qs, dd, xa, d8, static_cast<float *>(dst->data), (int) N, (int) K, (int) M,
                                               dst->ne[0], ks, q);
        return;
    }

    const size_t xa_bytes = (size_t) XMX_Q80_MAX_COLS * K;
    const size_t d8_bytes = (size_t) XMX_Q80_MAX_COLS * G * sizeof(sycl::half);
    ggml_sycl_pool_alloc<char> act_alloc(ctx.pool(), xa_bytes + d8_bytes);
    int8_t *     xa = reinterpret_cast<int8_t *>(act_alloc.get());
    sycl::half * d8 = reinterpret_cast<sycl::half *>(act_alloc.get() + xa_bytes);

    sycl::queue & q = *ctx.stream();
    xmx_q80_quant_act(static_cast<const float *>(src1->data), xa, d8, (int) K, (int) M, q);
    xmx_q80_launch(qs, dd, xa, d8, static_cast<float *>(dst->data), (int) N, (int) K, (int) M, dst->ne[0],
                   xmx_q80_pick_ks((int) N, (int) K), q);

    // GGML_SYCL_XMX_Q80_CHECK=1 (measurement only, synchronous): 256 sampled rows against a double-precision reference
    // on the same quantized operands; error relative to sum |d_w * d8 * dot| per output
    static const bool check = ggml_sycl_xmx_q80_env_int("GGML_SYCL_XMX_Q80_CHECK", 0) != 0;
    if (check) {
        q.wait();
        std::vector<int8_t>   hxa((size_t) M * K);
        std::vector<uint16_t> hd8((size_t) M * G);
        std::vector<float>    hout((size_t) M * N);
        q.memcpy(hxa.data(), xa, hxa.size()).wait();
        q.memcpy(hd8.data(), d8, hd8.size() * 2).wait();
        q.memcpy(hout.data(), dst->data, hout.size() * 4).wait();
        std::vector<int8_t>   wq(K);
        std::vector<uint16_t> wd(G);
        double max_rel = 0.0;
        int    bad     = 0;
        for (int t = 0; t < 256; ++t) {
            const int64_t r = (int64_t) ((t * 2654435761ull) % (uint64_t) N);
            q.memcpy(wq.data(), qs + r * K, K).wait();
            q.memcpy(wd.data(), dd + r * G, G * 2).wait();
            for (int64_t c = 0; c < M; ++c) {
                double ref = 0.0, mag = 0.0;
                for (int64_t g = 0; g < G; ++g) {
                    int dot = 0;
                    for (int e = 0; e < 32; ++e) { dot += (int) wq[g * 32 + e] * (int) hxa[c * K + g * 32 + e]; }
                    const double sc = (double) (float) sycl::bit_cast<sycl::half>(wd[g]) *
                                      (double) (float) sycl::bit_cast<sycl::half>(hd8[c * G + g]);
                    ref += sc * dot;
                    mag += std::fabs(sc * dot);
                }
                const double rel = std::fabs((double) hout[c * N + r] - ref) / std::max(mag, 1e-30);
                max_rel = std::max(max_rel, rel);
                bad += rel > 1e-5;
            }
        }
        {
            fprintf(stderr, "XMXQ80CHECK N=%lld K=%lld M=%lld: max rel err %.3e vs double (bad %d of %lld)\n",
                    (long long) N, (long long) K, (long long) M, max_rel, bad, (long long) (256 * M));
        }
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    GGML_SYCL_EXIT_OR_RETHROW();
}

#else  // !GGML_SYCL_XMX

void ggml_sycl_mul_mat_xmx_q80(ggml_backend_sycl_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {
    GGML_ABORT("XMX q8_0 path not built (configure with -DGGML_SYCL_XMX=ON)");
}

#endif
