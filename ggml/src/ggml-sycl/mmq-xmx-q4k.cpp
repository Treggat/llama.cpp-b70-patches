//
// q4_K x q8_1 matmul on the Xe2 matrix units - see mmq-xmx-q4k.hpp
//
// Kernel: b70-secret-sauce/xmx-probe/xmx_q4k.cpp, VAR 12 (PARTS 2, 256-GRF mode), host harness removed; MC 8 (one A
// tile) for 1..8 columns and MC 16 (two A tiles sharing every staged B tile) for 9..16 columns.
//
// Launch sequence (default): one fused kernel quantizes the f32 activations straight into the kernel's operands
// (int8 A tile rows, half-rounded d, integer group sums; the same arithmetic as quantize_and_reorder_q8_1_soa),
// then the matmul. No q8_1 SoA buffer, no second pack launch, no trip through ggml_sycl_op_mul_mat.
//
#include "mmq-xmx-q4k.hpp"
#include "mmq-dnn-u4.hpp"
#include "mmq-xmx-direct.hpp"
#include "mmq-xmx-q80.hpp"   // ggml_sycl_xmx_direct_wide (LOCAL wideverify)
#include "quants.hpp"
#include "quantize.hpp"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#if GGML_SYCL_XMX
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#endif

static int ggml_sycl_xmx_q4k_env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e && *e ? atoi(e) : def;
}

bool ggml_sycl_xmx_q4k_built() {
#if GGML_SYCL_XMX
    return true;
#else
    return false;
#endif
}

bool ggml_sycl_xmx_q4k_env() {
    static const bool v = ggml_sycl_xmx_q4k_env_int("GGML_SYCL_XMX_Q4K", 0) != 0;   // default off
    return v;
}

bool ggml_sycl_xmx_q4k_direct_env() {
    static const bool v = ggml_sycl_xmx_q4k_env_int("GGML_SYCL_XMX_Q4K_DIRECT", 0) != 0;   // default off
    return v;
}

int ggml_sycl_xmx_q4k_path() {
    // 0 (default): fused quantize + matmul, launched directly (2 kernels, both in the 256-register mode)
    // 1: legacy, through ggml_sycl_op_mul_mat (q8_1 SoA quantize, pack, matmul: 3 kernels), 1..8 columns only
    static const int v = ggml_sycl_xmx_q4k_env_int("GGML_SYCL_XMX_Q4K_PATH", 0);
    return v;
}

// kernel tile: 16 weight rows per sub-group, up to 16 activation columns (two 8-row A tiles)
static constexpr int XMX_Q4K_ROWS      = 16;
static constexpr int XMX_Q4K_MAX_COLS  = 16;

static int ggml_sycl_xmx_q4k_min_cols() {
    // default 3: at 2 columns MMVQ is within a few percent per op and the whole model measured ~5% slower on XMX.
    // With the direct kernel (GGML_SYCL_XMX_Q4K_DIRECT=1) the default is 2 (test-backend-ops perf, us, MMVQ vs direct
    // at 2 columns: 17408x5120 102.3 / 94.2, 12288x5120 62.7 / 58.9, 10240x5120 55.2 / 35.2, 6144x5120 37.6 / 15.7,
    // 5120x6144 41.3 / 15.8); 1 column stays on MMVQ (17408x5120 85.4 vs 91.5)
    static const int v = std::max(1, ggml_sycl_xmx_q4k_env_int("GGML_SYCL_XMX_Q4K_MIN_COLS",
                                                                ggml_sycl_xmx_q4k_direct_env() ? 2 : 3));
    return v;
}

static int ggml_sycl_xmx_q4k_max_cols() {
    // default 16: 9..16 columns otherwise fall to dequantize + f16 GEMM (~1150 us vs ~120 us on 17408x5120); the
    // legacy path only has the one-tile kernel
    static const int v = std::min(ggml_sycl_xmx_q4k_path() == 0 ? XMX_Q4K_MAX_COLS : 8,
                                  ggml_sycl_xmx_q4k_env_int("GGML_SYCL_XMX_Q4K_MAX_COLS", XMX_Q4K_MAX_COLS));
    return v;
}

bool ggml_sycl_xmx_glue_grf256(int op, int64_t nrows) {
    // GGML_SYCL_XMX_GLUE: 0 (default) off; 1 = only while the q4_K XMX path serves this column count (the glue kernels
    // then sit between 256-GRF XMX kernels); 2 = always (measurement). GGML_SYCL_XMX_GLUE_OPS: bit mask of the ops
    // that take part (GGML_SYCL_XMX_GLUE_GLU | _NORM | _ADD, default all).
    static const int mode = ggml_sycl_xmx_q4k_env_int("GGML_SYCL_XMX_GLUE", 0);
    static const int ops  = ggml_sycl_xmx_q4k_env_int("GGML_SYCL_XMX_GLUE_OPS", 7);
    if (mode == 0 || !(ops & op) || !ggml_sycl_xmx_q4k_built()) {
        return false;
    }
    if (mode == 2) {
        return true;
    }
    return ggml_sycl_xmx_q4k_env() && nrows >= ggml_sycl_xmx_q4k_min_cols() && nrows <= ggml_sycl_xmx_q4k_max_cols();
}

bool ggml_sycl_xmx_q4k_can_use(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               const ggml_tensor * dst) {
    if (!ggml_sycl_xmx_q4k_built() || !ggml_sycl_xmx_q4k_env()) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_K || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->op != GGML_OP_MUL_MAT) {   // MUL_MAT_ID comes through the same dispatcher with a stack dst
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    // the activations are read as K contiguous floats per column
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src1->ne[1] < ggml_sycl_xmx_q4k_min_cols() || src1->ne[1] > ggml_sycl_xmx_q4k_max_cols()) {
        return false;
    }
    if (src0->ne[0] % QK_K != 0 || src0->ne[1] % XMX_Q4K_ROWS != 0 || src1->ne[0] != src0->ne[0] ||
        dst->ne[0] != src0->ne[1] || ggml_nbytes(src0) >= (size_t) INT_MAX) {   // reorder offset helpers use int
        return false;
    }
    // with DMMV prioritised the reorder of K-quants is not used by the rest of the backend
    if (g_ggml_sycl_prioritize_dmmv) {
        return false;
    }
    // the oneDNN u4 path repacks q4_K into a layout this kernel cannot read; when both are switched on, u4 keeps q4_K
    if (ggml_sycl_dnn_u4_enabled()) {
        return false;
    }
    if (src0->extra == nullptr || src0->buffer == nullptr) {   // split buffers are excluded by the dispatcher
        return false;
    }
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    if (extra->optimized_feature.dnn_u4) {
        return false;
    }
    // tensors used mostly at one column (a draft/MTP layer) can be kept on MMVQ, e.g. "blk.64."
    static const char * skip_prefix = getenv("GGML_SYCL_XMX_Q4K_SKIP_PREFIX");
    if (skip_prefix && *skip_prefix && strncmp(src0->name, skip_prefix, strlen(skip_prefix)) == 0) {
        return false;
    }
    return ggml_sycl_xmx_q4k_device_ok(ctx.device);
}

#if GGML_SYCL_XMX

namespace mx  = sycl::ext::oneapi::experimental::matrix;
namespace imx = sycl::ext::intel::experimental::matrix;

bool ggml_sycl_xmx_q4k_device_ok(int device) {
    // 0 = not probed, 1 = no, 2 = yes. A joint_matrix kernel the runtime refuses throws at submit and the backend's
    // handlers exit, so this must be settled before the first launch, never by trying.
    static std::atomic<int> cache[GGML_SYCL_MAX_DEVICES];
    if (device < 0 || device >= GGML_SYCL_MAX_DEVICES) {
        return false;
    }
    int v = cache[device].load(std::memory_order_relaxed);
    if (v == 0) {
        bool ok = false;
        try {
            const sycl::device dev = dpct::dev_mgr::instance().get_device(device);
            if (dev.has(sycl::aspect::ext_intel_matrix)) {
                for (const auto & c :
                     dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>()) {
                    // a size of 0 means "any up to max_*": the 2026.1 runtime lists s8 x s8 as M=0 (max 8), N=16, K=32
                    const bool m_ok = c.msize == 8 || (c.msize == 0 && c.max_msize >= 8);
                    const bool n_ok = c.nsize == 16 || (c.nsize == 0 && c.max_nsize >= 16);
                    const bool k_ok = c.ksize == 32 || (c.ksize == 0 && c.max_ksize >= 32);
                    if (c.atype == mx::matrix_type::sint8 && c.btype == mx::matrix_type::sint8 && m_ok && n_ok && k_ok) {
                        ok = true;
                        break;
                    }
                }
            }
        } catch (const sycl::exception & e) {
            GGML_LOG_WARN("%s: matrix capability query failed on device %d: %s\n", __func__, device, e.what());
            ok = false;
        }
        v = ok ? 2 : 1;
        cache[device].store(v, std::memory_order_relaxed);
    }
    return v == 2;
}

namespace {

constexpr int XQ_TM = 8, XQ_TN = 16, XQ_TK = 32, XQ_SG = 16;
constexpr int XQ_PARTS = 2;                     // staging parts per 256-block
constexpr int XQ_GP    = 8 / XQ_PARTS;          // groups per staged part
constexpr int XQ_PB    = XQ_GP * 32;            // SLM bytes per row per part
constexpr int XQ_RS    = 16 / XQ_GP;            // rows covered by one 16-lane x 16 B load instruction

// Activation operands, in one pool allocation:
//   xa [16][K] int8, column c's quants contiguous (row c of the row-major A tiles)
//   d8 [16][G] half, the d MMVQ sees (kept as half in memory: a float -> half -> float round trip in registers is
//              folded away under the backend's fast-math flags, which would hand the kernel the unrounded d)
//   us [16][G] int32, the integer sum of the 32 quants of each group (MMVQ sums the int8 quants, not ds.s)
// Columns >= M are left unwritten: they only feed accumulator rows (Fb[i], i >= M) that are never stored, and the
// DPAS is integer, so whatever the pool holds there cannot reach a stored result.

// Fused f32 -> operands quantizer. One sub-group per (column, 32-group), launched with the very geometry of
// quantize_row_q8_1_sycl (WARP_SIZE lanes, QK8_1 / WARP_SIZE values per lane, block index = c * G + g, which is
// also the float offset / QK8_1 for a contiguous src1) and calling the same quantize_q8_1_impl, so the int8 quants
// and d are the ones quantize_and_reorder_q8_1_soa hands MMVQ. A hand-written copy of that arithmetic (x / d under
// the backend's fast-math flags) was NOT bit-identical: 1-2 quants off by one in ~8% of calls (GGML_SYCL_XMX_Q4K_CHECKQ).
// It runs in the 256-register mode of the matmul: every switch between a 128- and a 256-register kernel costs the
// B70 ~6.5 us (an empty 128-GRF kernel ahead of the matmul adds 6.8 us per op, an empty 256-GRF one 0.5 us), so a
// default-mode quantizer made each op pay two switches.
// gmajor: xa in the direct kernel's group-major layout [G][16][32] (column c of group g at (g*16 + c)*32) instead of
// [16][K]; d8 and us are the same either way
sycl::event xmx_q4k_quant_act(const float * x, int8_t * xa, sycl::half * d8, int32_t * us, int K, int M, sycl::queue & q,
                              bool gmajor = false) {
    constexpr int GRF = 256;
    constexpr int EPW = QK8_1 / WARP_SIZE;
    const int G = K / QK8_1;
    return q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) M * G * WARP_SIZE), sycl::range<1>(WARP_SIZE)),
                          sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<GRF> },
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        sycl::vec<int8_t, EPW> qv;
        float d = 0.0f, sum = 0.0f;
        quantize_q8_1_impl<EPW>(x, qv, d, sum, it);
        const size_t blk  = it.get_group(0);
        const int    lane = (int) it.get_local_id(0);
        int s = 0;
#pragma unroll
        for (int i = 0; i < EPW; ++i) { s += (int) qv[i]; }
        s = sycl::reduce_over_group(it.get_sub_group(), s, sycl::plus<int>());
        // xa row c holds column c's K quants contiguously: offset c*K + g*QK8_1 = blk*QK8_1
        // (group-major: c = blk / G, g = blk % G -> (g*16 + c)*QK8_1)
        const size_t xo = gmajor ? ((blk % G) * XQ_TN + blk / G) * QK8_1 : blk * QK8_1;
        *reinterpret_cast<sycl::vec<int8_t, EPW> *>(xa + xo + (size_t) lane * EPW) = qv;
        if (lane == 0) {
            d8[blk] = sycl::half(d);   // [c][G] with c*G + g = blk
            us[blk] = s;
        }
    });
}

// legacy step 1: the driver's quantizer (quantize_and_reorder_q8_1_soa into a padded SoA q8_1 buffer)
sycl::event xmx_q4k_quant_soa(const float * x, void * vy, int K, int M, int K_padded, sycl::queue & q) {
    const size_t nblk = (size_t) M * (K / QK8_1);
    return q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nblk * WARP_SIZE), sycl::range<1>(WARP_SIZE)),
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        quantize_and_reorder_q8_1_soa<QK8_1 / WARP_SIZE>()(x, vy, K, K_padded, it);
    });
}

// legacy step 2: pack from the SoA q8_1 buffer: column c starts at y + c*stride_y, its K int8 quants at [0, K),
// its K/32 half2 (d, s) at K + g*4. Zeroes columns M..15 as the original did.
sycl::event xmx_q4k_pack_act(const char * y, size_t stride_y, int8_t * xa, sycl::half * d8, int32_t * us, int K, int M,
                             sycl::queue & q) {
    const int G = K / QK8_1;
    return q.parallel_for(sycl::range<1>((size_t) XQ_TN * G), [=](sycl::id<1> idx) {
        const int c = (int) (idx[0] / G);
        const int g = (int) (idx[0] % G);
        if (c < M) {
            const uint8_t *   col = reinterpret_cast<const uint8_t *>(y) + (size_t) c * stride_y;
            const sycl::uint4 lo  = *reinterpret_cast<const sycl::uint4 *>(col + (size_t) g * QK8_1);
            const sycl::uint4 hi  = *reinterpret_cast<const sycl::uint4 *>(col + (size_t) g * QK8_1 + 16);
            const uint32_t    w[8] = { lo.x(), lo.y(), lo.z(), lo.w(), hi.x(), hi.y(), hi.z(), hi.w() };
            int s = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                s += (int) (int8_t) (w[j]) + (int) (int8_t) (w[j] >> 8) + (int) (int8_t) (w[j] >> 16) +
                     (int) (int8_t) (w[j] >> 24);
            }
            int8_t * dst = xa + (size_t) c * K + (size_t) g * QK8_1;
            *reinterpret_cast<sycl::uint4 *>(dst)      = lo;
            *reinterpret_cast<sycl::uint4 *>(dst + 16) = hi;
            const sycl::half2 ds = *reinterpret_cast<const sycl::half2 *>(col + K + (size_t) g * sizeof(sycl::half2));
            d8[(size_t) c * G + g] = ds.x();
            us[(size_t) c * G + g] = s;
        } else {
            int8_t * dst = xa + (size_t) c * K + (size_t) g * QK8_1;
            *reinterpret_cast<sycl::uint4 *>(dst)      = sycl::uint4(0, 0, 0, 0);
            *reinterpret_cast<sycl::uint4 *>(dst + 16) = sycl::uint4(0, 0, 0, 0);
            d8[(size_t) c * G + g] = sycl::half(0.f);
            us[(size_t) c * G + g] = 0;
        }
    });
}

// The kernel (xmx_q4k.cpp VAR 12). Transposed roles: activations are the A tile (M = 8 columns x 32), weights are
// the B tile (32 x 16 rows), so every lane of the accumulator is one weight row and owns that row's scales in
// registers; element i = activation column i. MC = 16 runs two A tiles (columns 0..7, 8..15) against every staged
// B tile. Each sub-group stages its own 16 rows in a private SLM slice ([row][k], read back as a col_major B; only
// sub-group barriers), 2 parts per block (2 KB per sub-group); the next part's nibbles are prefetched into
// registers while the current part runs on DPAS.
// Weights are staged as (q - 8) * 16 in s8 (nibble moved to the high half of the byte, top bit flipped); the x16 is
// folded exactly (power of two) into the row scale. Per group g, element i (column i), lane n (row n), with
// dot16 = 16 * dot(q - 8, u) from the DPAS:
//   Fb += (float) dot16 * ((d * sc / 16) * d8)  +  (d8 * sum(u)) * (8 * d * sc - dmin * mn)
// i.e. d * d8 * sc * dot(q, u) - dmin * d8 * mn * sum(u): the -8 offset and the mins become one rank-1 term.
// (float) dot16 is exact (|dot16| < 2^24) and d * sc / 16 is exact; only float rounding order differs from MMVQ.
// Fb is a per-block partial sum added into F once per block. Split-K: a work-group is `ks` sub-groups on the same
// 16 rows, each taking a contiguous block range; partial sums are reduced in SLM in a fixed order (deterministic).
//   qs [N][K/2], sc [N*KB][12], dm [N*KB] half2 (d, dmin): the q4_K reorder layout
//   out: column i of row r at out[i*ldd + r], for i < M
template <int MC, int GRF>
sycl::event xmx_q4k_launch(const uint8_t * dQs, const uint8_t * dSc, const uint32_t * dDm, const int8_t * dXa,
                           const sycl::half * dD8c, const int32_t * dS8c, float * dOut, int N, int K, int M, int64_t ldd,
                           int ks, sycl::queue & q) {
    static_assert(MC == 8 || MC == 16, "one or two A tiles");
    constexpr int NT = MC / XQ_TM;
    const int KB = K / QK_K, G = K / QK8_1;
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<int8_t, 1> bslm(sycl::range<1>(ks * 16 * XQ_PB), h);
        sycl::local_accessor<float, 1>  red(sycl::range<1>(ks > 1 ? ks * MC * 16 : 1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (N / 16) * ks * XQ_SG), sycl::range<1>(ks * XQ_SG)),
                       sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<GRF> },
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(XQ_SG)]] {
            using sycl::access::address_space;
            using sycl::access::decorated;
            sycl::sub_group sg = it.get_sub_group();
            const int lane = sg.get_local_id()[0];
            const int s    = sg.get_group_id()[0];
            const int row0 = (int) it.get_group(0) * 16;
            const int b0   = s * KB / ks, b1 = (s + 1) * KB / ks;
            auto bp = bslm.get_multi_ptr<decorated::no>() + s * 16 * XQ_PB;

            // staging geometry: 16 B pieces, consecutive lanes on consecutive pieces of a row (whole 64 B lines);
            // load instruction j covers rows j*RS .. j*RS+RS-1 of the part
            const int pp = lane % XQ_GP;
            const uint8_t * gb = dQs + (size_t) (row0 + lane / XQ_GP) * (K / 2) + pp * 16;
            const int so = (lane / XQ_GP) * XQ_PB + (2 * (pp >> 1)) * 32 + (pp & 1) * 16;
            const uint32_t * hsc = reinterpret_cast<const uint32_t *>(dSc + (size_t) (row0 + lane) * KB * 12);
            const uint32_t * hdm = dDm + (size_t) (row0 + lane) * KB;
            const int8_t * xa = dXa;
            const sycl::uint4 *  d8base = reinterpret_cast<const sycl::uint4 *>(dD8c + (size_t) lane * G);   // 8 halves per block
            const sycl::int4 *   s8base = reinterpret_cast<const sycl::int4 *>(dS8c + (size_t) lane * G);

            auto ldq = [](const uint8_t * p) -> sycl::uint4 { return *reinterpret_cast<const sycl::uint4 *>(p); };
            sycl::uint4 v[XQ_GP];       // nibbles of the next part (register prefetch)
#pragma unroll
            for (int j = 0; j < XQ_GP; ++j) { v[j] = ldq(gb + (size_t) j * XQ_RS * (K / 2) + b0 * 128); }
            uint32_t hw0 = hsc[b0 * 3 + 0], hw1 = hsc[b0 * 3 + 1], hw2 = hsc[b0 * 3 + 2], hwd = hdm[b0];

            float F[MC];
            for (int i = 0; i < MC; ++i) { F[i] = 0.f; }
            for (int b = b0; b < b1; ++b) {
                // this lane's row: rsd = d * sc / 16 and the rank-1 weight rw = 8*d*sc - dmin*mn
                float rsd[8], rw[8];
                {
                    // 6-bit scales/mins, 4 per dword (the ggml get_scale_min_k4 layout), decoded SWAR
                    const uint32_t sc03 = hw0 & 0x3F3F3F3Fu, mn03 = hw1 & 0x3F3F3F3Fu;
                    const uint32_t sc47 = (hw2 & 0x0F0F0F0Fu) | ((hw0 >> 2) & 0x30303030u);
                    const uint32_t mn47 = ((hw2 >> 4) & 0x0F0F0F0Fu) | ((hw1 >> 2) & 0x30303030u);
                    const float d    = (float) sycl::bit_cast<sycl::half>((uint16_t) (hwd & 0xFFFF));
                    const float dmin = (float) sycl::bit_cast<sycl::half>((uint16_t) (hwd >> 16));
                    const float d16  = d * 0.0625f, d8x = 8.f * d;     // exact power-of-two rescales
#pragma unroll
                    for (int g = 0; g < 8; ++g) {
                        const float sc = (float) (uint8_t) ((g < 4 ? sc03 : sc47) >> (8 * (g & 3)));
                        const float mn = (float) (uint8_t) ((g < 4 ? mn03 : mn47) >> (8 * (g & 3)));
                        rw[g]  = d8x * sc - dmin * mn;
                        rsd[g] = d16 * sc;
                    }
                }
                float d8c[8], tc[8];     // column `lane`'s d8 and d8 * sum(u) for the 8 groups of this block
                {
                    const sycl::uint4 dh = d8base[b];
                    const sycl::int4  sa = s8base[b * 2], sb = s8base[b * 2 + 1];
                    const uint32_t    dw[4] = { dh.x(), dh.y(), dh.z(), dh.w() };
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        d8c[2 * j]     = (float) sycl::bit_cast<sycl::half>((uint16_t) (dw[j] & 0xFFFF));
                        d8c[2 * j + 1] = (float) sycl::bit_cast<sycl::half>((uint16_t) (dw[j] >> 16));
                    }
                    tc[0] = d8c[0] * (float) sa.x(); tc[1] = d8c[1] * (float) sa.y(); tc[2] = d8c[2] * (float) sa.z(); tc[3] = d8c[3] * (float) sa.w();
                    tc[4] = d8c[4] * (float) sb.x(); tc[5] = d8c[5] * (float) sb.y(); tc[6] = d8c[6] * (float) sb.z(); tc[7] = d8c[7] * (float) sb.w();
                }
                float Fb[MC];            // per-block partial sums
#pragma unroll
                for (int i = 0; i < MC; ++i) { Fb[i] = 0.f; }
                {   // next block's header (clamped, unconditional)
                    const int bn = b + 1 < b1 ? b + 1 : b;
                    hw0 = hsc[bn * 3 + 0]; hw1 = hsc[bn * 3 + 1]; hw2 = hsc[bn * 3 + 2]; hwd = hdm[bn];
                }
#pragma unroll
                for (int part = 0; part < XQ_PARTS; ++part) {
#pragma unroll
                    for (int j = 0; j < XQ_GP; ++j) {
                        // (q - 8) * 16 as signed bytes: nibble moved to the high half, top bit flipped
                        sycl::uint4 lo, hi;
                        lo.x() = ((v[j].x() << 4) & 0xF0F0F0F0u) ^ 0x80808080u; lo.y() = ((v[j].y() << 4) & 0xF0F0F0F0u) ^ 0x80808080u;
                        lo.z() = ((v[j].z() << 4) & 0xF0F0F0F0u) ^ 0x80808080u; lo.w() = ((v[j].w() << 4) & 0xF0F0F0F0u) ^ 0x80808080u;
                        hi.x() = (v[j].x() & 0xF0F0F0F0u) ^ 0x80808080u; hi.y() = (v[j].y() & 0xF0F0F0F0u) ^ 0x80808080u;
                        hi.z() = (v[j].z() & 0xF0F0F0F0u) ^ 0x80808080u; hi.w() = (v[j].w() & 0xF0F0F0F0u) ^ 0x80808080u;
                        *reinterpret_cast<sycl::uint4 *>(&bp[so + j * XQ_RS * XQ_PB])      = lo;
                        *reinterpret_cast<sycl::uint4 *>(&bp[so + j * XQ_RS * XQ_PB + 32]) = hi;
                    }
                    {   // prefetch the next part's nibbles (clamped, unconditional: no phi copies)
                        const int nb = part + 1 < XQ_PARTS ? b : (b + 1 < b1 ? b + 1 : b);
                        const int np = part + 1 < XQ_PARTS ? part + 1 : (b + 1 < b1 ? 0 : part);
#pragma unroll
                        for (int j = 0; j < XQ_GP; ++j) { v[j] = ldq(gb + (size_t) j * XQ_RS * (K / 2) + nb * 128 + np * (XQ_GP * 16)); }
                    }
                    sycl::group_barrier(sg);
#pragma unroll
                    for (int gg = 0; gg < XQ_GP; ++gg) {
                        const int g = part * XQ_GP + gg;
                        mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, XQ_TK, XQ_TN, mx::layout::col_major> B;
                        mx::joint_matrix_load(sg, B, bp + gg * XQ_TK, XQ_PB);
#pragma unroll
                        for (int t = 0; t < NT; ++t) {
                            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, XQ_TM, XQ_TK, mx::layout::row_major> A;
                            mx::joint_matrix_load(sg, A, sycl::address_space_cast<address_space::global_space, decorated::no>(xa + (size_t) t * XQ_TM * K + b * QK_K + g * XQ_TK), K);
                            mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, XQ_TM, XQ_TN> C;
                            mx::joint_matrix_fill(sg, C, 0);
                            mx::joint_matrix_mad(sg, C, A, B, C);
                            // column-side factors broadcast from lane t*8+i (register regions, no messages); these
                            // sub-group ops must stay outside the joint_matrix_apply lambda
                            float sb[8], tb[8];
#pragma unroll
                            for (int i = 0; i < 8; ++i) { sb[i] = rsd[g] * sycl::select_from_group(sg, d8c[g], t * 8 + i); tb[i] = sycl::select_from_group(sg, tc[g], t * 8 + i); }
                            int i = 0;
                            imx::joint_matrix_apply(sg, C, [&](int32_t & dot16, size_t, size_t) {
                                if (i < 8) {
                                    Fb[t * 8 + i] = sycl::fma((float) dot16, sb[i], Fb[t * 8 + i]);
                                    Fb[t * 8 + i] = sycl::fma(tb[i], rw[g], Fb[t * 8 + i]);
                                }
                                ++i;
                            });
                        }
                    }
                    sycl::group_barrier(sg);
                }
#pragma unroll
                for (int i = 0; i < MC; ++i) { F[i] += Fb[i]; }
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

// split-K factor (sub-groups per work-group sharing one 16-row tile). Swept on the B70 with test-backend-ops perf
// (2026-09-26, us per op at 6 columns / 16 columns; ks-to-time is not monotonic, so the model shapes are tabled):
//   N x K        ks=2      3      4      5      6      8     10   picked
//   17408x5120   113/158 102/123 105/124 101/117 105/134 110/139 104/131   5   (the rule gave 3)
//   5120x17408   129/141 105/120 122/164 110/139 106/132 110/130 109/129   3   (the rule gave 8)
//   10240x5120    54/75   49/69   46/60   47/61   49/69   55/79   51/75    4   (= rule)
//   6144x5120     26/35   33/46   29/40   26/35   32/48   31/48   30/42    2   (the rule gave 5)
// Other shapes keep the rule of thumb: >= 2560 sub-groups, >= 4 blocks each, capped 16.
int xmx_q4k_pick_ks(int N, int K, int M) {
    GGML_UNUSED(M);
    const int KB    = K / QK_K;
    const int tiles = N / XMX_Q4K_ROWS;
    int ks = std::max(1, std::min((2560 + tiles - 1) / tiles, KB / 4));
    static const struct { int n, k, ks; } tuned[] = {
        { 17408, 5120, 5 }, { 5120, 17408, 3 }, { 10240, 5120, 4 }, { 6144, 5120, 2 },
    };
    // GGML_SYCL_XMX_Q4K_KS_TABLE=0: the rule of thumb only (the pre-table behaviour, for A/B)
    static const bool use_table = ggml_sycl_xmx_q4k_env_int("GGML_SYCL_XMX_Q4K_KS_TABLE", 1) != 0;
    for (const auto & t : tuned) {
        if (use_table && t.n == N && t.k == K) {
            ks = t.ks;
        }
    }
    // GGML_SYCL_XMX_KS_V2=1 (LOCAL, default off): 5120x6144 (ssm_out, attn_output: 64 ops per verify graph) ks 3:
    // 26.0 -> 22.2 us at 3..8 columns (the rule's 6 leaves the 320 tiles at 5 work-groups per Xe core)
    static const bool ks_v2 = ggml_sycl_xmx_q4k_env_int("GGML_SYCL_XMX_KS_V2", 0) != 0;
    if (ks_v2 && N == 5120 && K == 6144) {
        ks = 3;
    }
    static const int ks_env = ggml_sycl_xmx_q4k_env_int("GGML_SYCL_XMX_Q4K_KS", 0);
    if (ks_env > 0) {
        ks = ks_env;
    }
    return std::max(1, std::min(ks, std::min(16, KB)));
}

// weights, read in place from the q4_K reorder layout (reorder_qw_q4_k): the section offsets are the ones
// ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K> gives MMVQ with nrows = N (block 0 of each section)
struct xmx_q4k_weights {
    const uint8_t *  qs;   // [N][K/2]
    const uint8_t *  sc;   // [N*KB][12]
    const uint32_t * dm;   // [N*KB] half2 (d, dmin)
};

xmx_q4k_weights xmx_q4k_weights_of(const char * src0_dd, int64_t N, int64_t K) {
    using q4k_reordered = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>;
    const int64_t KB = K / QK_K;
    GGML_ASSERT(N * KB * (int64_t) (QK_K / 2 + K_SCALE_SIZE + sizeof(sycl::half2)) < (int64_t) INT_MAX);   // helper's int math
    const auto q_off = q4k_reordered::get_block_offset(0, (int) (N * KB));
    const auto d_off = q4k_reordered::get_d_offset((int) N, (int) K, 0);
    return { reinterpret_cast<const uint8_t *>(src0_dd) + q_off.first,
             reinterpret_cast<const uint8_t *>(src0_dd) + d_off.first,
             reinterpret_cast<const uint32_t *>(src0_dd + d_off.second) };
}

sycl::event xmx_q4k_matmul(const xmx_q4k_weights & w, const int8_t * xa, const sycl::half * d8, const int32_t * us,
                           float * dst, int N, int K, int M, int64_t ldd, sycl::queue & q) {
    const int ks = xmx_q4k_pick_ks(N, K, M);
    if (M > 8) {
        return xmx_q4k_launch<16, 256>(w.qs, w.sc, w.dm, xa, d8, us, dst, N, K, M, ldd, ks, q);
    }
    // 256-GRF only: the 128-GRF build of this kernel spills and measured 2.2x slower (234 vs 105 us on 17408x5120)
    return xmx_q4k_launch<8, 256>(w.qs, w.sc, w.dm, xa, d8, us, dst, N, K, M, ldd, ks, q);
}

// ---- GGML_SYCL_XMX_Q4K_PROF=1: per-launch GPU times on a profiling queue (synchronous; for measurement only) ----
struct xmx_q4k_prof_rec {
    double sum[3] = { 0, 0, 0 }, mn[3] = { 1e30, 1e30, 1e30 };
    std::vector<double> tot, span;
    int    n = 0, nk = 0, ks = 0;
};

struct xmx_q4k_prof_db {
    std::mutex                              mu;
    std::map<std::string, xmx_q4k_prof_rec> recs;
    ~xmx_q4k_prof_db() {
        for (auto & kv : recs) {
            auto & r = kv.second;
            if (r.n == 0) {
                continue;
            }
            std::sort(r.tot.begin(), r.tot.end());
            std::sort(r.span.begin(), r.span.end());
            fprintf(stderr, "XMXPROF %s ks=%d calls=%d |", kv.first.c_str(), r.ks, r.n);
            static const char * nm[3] = { "quant", "pack", "matmul" };
            for (int k = 0; k < r.nk; ++k) {
                fprintf(stderr, " %s avg %.1f min %.1f |", nm[r.nk == 2 && k == 1 ? 2 : k], r.sum[k] / r.n, r.mn[k]);
            }
            fprintf(stderr, " kernels min %.1f med %.1f | first start..last end min %.1f med %.1f us\n", r.tot[0],
                    r.tot[r.tot.size() / 2], r.span[0], r.span[r.span.size() / 2]);
        }
    }
};

xmx_q4k_prof_db & xmx_q4k_prof() {
    static xmx_q4k_prof_db db;
    return db;
}

double ev_us(const sycl::event & e) {
    const auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return (double) (t1 - t0) * 1e-3;
}

void xmx_q4k_prof_add(int N, int K, int M, const std::vector<sycl::event> & ev) {
    char key[96];
    snprintf(key, sizeof(key), "N=%d K=%d M=%d path=%d", N, K, M, ggml_sycl_xmx_q4k_path());
    auto & db = xmx_q4k_prof();
    std::lock_guard<std::mutex> lk(db.mu);
    auto & r = db.recs[key];
    r.nk = (int) ev.size();
    r.ks = xmx_q4k_pick_ks(N, K, M);
    double tot = 0;
    for (size_t k = 0; k < ev.size(); ++k) {
        const double t = ev_us(ev[k]);
        r.sum[k] += t;
        r.mn[k] = std::min(r.mn[k], t);
        tot += t;
    }
    r.tot.push_back(tot);
    r.span.push_back((double) (ev.back().get_profiling_info<sycl::info::event_profiling::command_end>() -
                               ev.front().get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-3);
    r.n++;
}

// GGML_SYCL_XMX_Q4K_CHECKQ=1: compare the fused quantizer's operands with the legacy quantize + pack, byte for byte
void xmx_q4k_checkq(const float * x, const int8_t * xa, const sycl::half * d8, const int32_t * us, int K, int M,
                    ggml_backend_sycl_context & ctx, sycl::queue & q) {
    const int     G   = K / QK8_1;
    const int64_t Kp  = GGML_PAD(K, MATRIX_ROW_PADDING);
    const size_t  sty = (size_t) Kp * sizeof(block_q8_1) / QK8_1;
    ggml_sycl_pool_alloc<char> soa(ctx.pool(), sty * M);
    ggml_sycl_pool_alloc<char> ref(ctx.pool(), (size_t) XQ_TN * K + (size_t) XQ_TN * G * 8);
    int8_t *  rxa = reinterpret_cast<int8_t *>(ref.get());
    sycl::half * rd8 = reinterpret_cast<sycl::half *>(ref.get() + (size_t) XQ_TN * K);
    int32_t * rus = reinterpret_cast<int32_t *>(ref.get() + (size_t) XQ_TN * K + (size_t) XQ_TN * G * 4);
    xmx_q4k_quant_soa(x, soa.get(), K, M, (int) Kp, q);
    xmx_q4k_pack_act(soa.get(), sty, rxa, rd8, rus, K, M, q);
    std::vector<int8_t> a((size_t) M * K), ra((size_t) M * K);
    std::vector<uint16_t> b((size_t) M * G), rb((size_t) M * G);
    std::vector<int32_t> c((size_t) M * G), rc((size_t) M * G);
    q.memcpy(a.data(), xa, a.size()).wait();
    q.memcpy(ra.data(), rxa, ra.size()).wait();
    q.memcpy(b.data(), d8, b.size() * 2).wait();
    q.memcpy(rb.data(), rd8, rb.size() * 2).wait();
    q.memcpy(c.data(), us, c.size() * 4).wait();
    q.memcpy(rc.data(), rus, rc.size() * 4).wait();
    const bool ok = memcmp(a.data(), ra.data(), a.size()) == 0 && memcmp(b.data(), rb.data(), b.size() * 2) == 0 &&
                    memcmp(c.data(), rc.data(), c.size() * 4) == 0;
    if (!ok) {
        int na = 0, nb = 0, nc = 0;
        for (size_t i = 0; i < a.size(); ++i) { if (a[i] != ra[i]) { if (na++ < 3) fprintf(stderr, "  xa[%zu] %d vs %d\n", i, a[i], ra[i]); } }
        for (size_t i = 0; i < b.size(); ++i) { if (b[i] != rb[i]) { if (nb++ < 3) fprintf(stderr, "  d8[%zu] %04x vs %04x\n", i, b[i], rb[i]); } }
        for (size_t i = 0; i < c.size(); ++i) { if (c[i] != rc[i]) { if (nc++ < 3) fprintf(stderr, "  us[%zu] %d vs %d\n", i, c[i], rc[i]); } }
        fprintf(stderr, "  mismatches xa %d/%zu d8 %d/%zu us %d/%zu\n", na, a.size(), nb, b.size(), nc, c.size());
    }
    static std::atomic<int> n_ok{ 0 }, n_bad{ 0 };
    (ok ? n_ok : n_bad)++;
    if (!ok || (n_ok + n_bad) % 50 == 1) {
        fprintf(stderr, "XMXCHECKQ K=%d M=%d %s (ok %d, mismatched %d)\n", K, M, ok ? "IDENTICAL" : "MISMATCH", (int) n_ok,
                (int) n_bad);
    }
}

} // namespace

void ggml_sycl_mul_mat_xmx_q4k(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst) try {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2, " : XMX q4_K");

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const int64_t G = K / QK8_1;
    GGML_ASSERT(src1->ne[0] == K && dst->ne[0] == N && dst->ne[1] == M);
    GGML_ASSERT(K % QK_K == 0 && N % XMX_Q4K_ROWS == 0);
    GGML_ASSERT(M >= 1 && M <= XMX_Q4K_MAX_COLS);
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    GGML_ASSERT(extra && extra->optimized_feature.reorder);

    const xmx_q4k_weights w = xmx_q4k_weights_of(static_cast<const char *>(src0->data), N, K);
    const float * x   = static_cast<const float *>(src1->data);
    float *       out = static_cast<float *>(dst->data);

    const size_t xa_bytes = (size_t) XQ_TN * K;
    const size_t d8_bytes = (size_t) XQ_TN * G * sizeof(sycl::half);
    const size_t us_bytes = (size_t) XQ_TN * G * sizeof(int32_t);
    ggml_sycl_pool_alloc<char> act_alloc(ctx.pool(), xa_bytes + d8_bytes + us_bytes);
    int8_t *  xa = reinterpret_cast<int8_t *>(act_alloc.get());
    sycl::half * d8 = reinterpret_cast<sycl::half *>(act_alloc.get() + xa_bytes);
    int32_t * us = reinterpret_cast<int32_t *>(act_alloc.get() + xa_bytes + d8_bytes);

    const queue_ptr stream = ctx.stream();
    static const bool prof   = ggml_sycl_xmx_q4k_env_int("GGML_SYCL_XMX_Q4K_PROF", 0) != 0;
    static const bool checkq = ggml_sycl_xmx_q4k_env_int("GGML_SYCL_XMX_Q4K_CHECKQ", 0) != 0;
    sycl::queue * q = stream;
    if (prof) {
        static sycl::queue * pq = new sycl::queue(stream->get_context(), stream->get_device(),
                                                  sycl::property_list{ sycl::property::queue::in_order(),
                                                                       sycl::property::queue::enable_profiling() });
        stream->wait();
        q = pq;
    }

    // register-fed DPAS kernel for 1..8 columns (GGML_SYCL_XMX_Q4K_DIRECT=1): same partition (split-K) and operation
    // order as xmx_q4k_matmul, so bit-identical; very tall weights (a 98k-row draft head, where the rule of thumb gives
    // ks = 1) get ks 5: 530 -> 493 us on 98304x5120, not bit-identical to ks 1 there (float order of 5 partial sums)
    // LOCAL (wideverify): 9..16 columns on the two-tile register-fed kernel with GGML_SYCL_XMX_WIDE=1 (the same
    // partition and per-column operation order as xmx_q4k_launch<16>, so bit-identical to it at the same ks)
    const bool direct = ggml_sycl_xmx_q4k_direct_env() && (M <= 8 || ggml_sycl_xmx_direct_wide());
    std::vector<sycl::event> ev;
    ev.push_back(xmx_q4k_quant_act(x, xa, d8, us, (int) K, (int) M, *q, direct));
    if (checkq && !direct) {
        xmx_q4k_checkq(x, xa, d8, us, (int) K, (int) M, ctx, *q);
    }
    if (direct) {
        int ks = xmx_q4k_pick_ks((int) N, (int) K, (int) M);
        if (N >= 65536 && ks < 5) {
            ks = std::max(1, std::min(5, (int) (K / QK_K) / 4));
        }
        if (M > 8) {
            static const int ks16_env = ggml_sycl_xmx_q4k_env_int("GGML_SYCL_XMX_Q4K_DIRECT16_KS", 0);   // tuning only
            if (ks16_env > 0) {
                ks = std::max(1, std::min(ks16_env, std::min(16, (int) (K / QK_K))));
            }
            ev.push_back(ggml_sycl_xmx_q4k_direct16_launch(w.qs, w.sc, w.dm, xa, d8, us, out, (int) N, (int) K, (int) M,
                                                           dst->ne[0], ks, *q));
        } else {
            ev.push_back(ggml_sycl_xmx_q4k_direct_launch(w.qs, w.sc, w.dm, xa, d8, us, out, (int) N, (int) K, (int) M,
                                                         dst->ne[0], ks, *q));
        }
        // GGML_SYCL_XMX_Q4K_DIRECT_CHECK=1 (measurement only, synchronous): rerun the op on the joint_matrix kernel
        // (same split-K) and compare every output bit for bit
        static const bool dcheck = ggml_sycl_xmx_q4k_env_int("GGML_SYCL_XMX_Q4K_DIRECT_CHECK", 0) != 0;
        if (dcheck) {
            ggml_sycl_pool_alloc<char> ref_act(ctx.pool(), xa_bytes + d8_bytes + us_bytes);
            ggml_sycl_pool_alloc<float> ref_out(ctx.pool(), (size_t) M * N);
            int8_t *     rxa = reinterpret_cast<int8_t *>(ref_act.get());
            sycl::half * rd8 = reinterpret_cast<sycl::half *>(ref_act.get() + xa_bytes);
            int32_t *    rus = reinterpret_cast<int32_t *>(ref_act.get() + xa_bytes + d8_bytes);
            xmx_q4k_quant_act(x, rxa, rd8, rus, (int) K, (int) M, *q, false);
            if (M > 8) {
                xmx_q4k_launch<16, 256>(w.qs, w.sc, w.dm, rxa, rd8, rus, ref_out.get(), (int) N, (int) K, (int) M, N, ks, *q);
            } else {
                xmx_q4k_launch<8, 256>(w.qs, w.sc, w.dm, rxa, rd8, rus, ref_out.get(), (int) N, (int) K, (int) M, N, ks, *q);
            }
            q->wait();
            std::vector<float> a((size_t) M * N), r((size_t) M * N);
            for (int64_t i = 0; i < M; ++i) {
                q->memcpy(a.data() + i * N, out + i * dst->ne[0], N * sizeof(float));
            }
            q->memcpy(r.data(), ref_out.get(), r.size() * sizeof(float)).wait();
            size_t nd = 0;
            for (size_t i = 0; i < a.size(); ++i) { nd += memcmp(&a[i], &r[i], sizeof(float)) != 0; }
            static std::atomic<int> n_ok{ 0 }, n_bad{ 0 };
            (nd == 0 ? n_ok : n_bad)++;
            fprintf(stderr, "XMXDIRECTCHECK q4_K N=%lld K=%lld M=%lld ks=%d: %s (%zu of %zu differ; ok %d, mismatched %d)\n",
                    (long long) N, (long long) K, (long long) M, ks, nd == 0 ? "IDENTICAL" : "MISMATCH", nd, a.size(),
                    (int) n_ok, (int) n_bad);
        }
    } else {
        ev.push_back(xmx_q4k_matmul(w, xa, d8, us, out, (int) N, (int) K, (int) M, dst->ne[0], *q));
    }
    if (prof) {
        q->wait();
        xmx_q4k_prof_add((int) N, (int) K, (int) M, ev);
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    GGML_SYCL_EXIT_OR_RETHROW();
}

bool ggml_sycl_xmx_q4k_gateup(ggml_backend_sycl_context & ctx, const ggml_tensor * wg, const ggml_tensor * wu,
                              const ggml_tensor * act, float * outG, float * outU, int64_t ldd, bool glu) try {
    if (!ggml_sycl_xmx_q4k_direct_env()) {
        return false;
    }
    const int64_t K = wg->ne[0];
    const int64_t N = wg->ne[1];
    const int64_t M = act->ne[1];
    if (M < 1 || M > 8 || wu->ne[0] != K || wu->ne[1] != N || act->ne[0] != K || K % QK_K != 0 || N % XMX_Q4K_ROWS != 0) {
        return false;
    }
    const auto * eg = static_cast<const ggml_tensor_extra_gpu *>(wg->extra);
    const auto * eu = static_cast<const ggml_tensor_extra_gpu *>(wu->extra);
    if (!eg || !eu || !eg->optimized_feature.reorder || !eu->optimized_feature.reorder) {
        return false;
    }
    scope_op_debug_print scope_dbg_print(__func__, wg, /*num_src=*/0, " : XMX q4_K gate+up pair");
    const int64_t G = K / QK8_1;
    const xmx_q4k_weights a = xmx_q4k_weights_of(static_cast<const char *>(wg->data), N, K);
    const xmx_q4k_weights b = xmx_q4k_weights_of(static_cast<const char *>(wu->data), N, K);
    const size_t xa_bytes = (size_t) XQ_TN * K;
    const size_t d8_bytes = (size_t) XQ_TN * G * sizeof(sycl::half);
    const size_t us_bytes = (size_t) XQ_TN * G * sizeof(int32_t);
    ggml_sycl_pool_alloc<char> act_alloc(ctx.pool(), xa_bytes + d8_bytes + us_bytes);
    int8_t *     xa = reinterpret_cast<int8_t *>(act_alloc.get());
    sycl::half * d8 = reinterpret_cast<sycl::half *>(act_alloc.get() + xa_bytes);
    int32_t *    us = reinterpret_cast<int32_t *>(act_alloc.get() + xa_bytes + d8_bytes);
    sycl::queue & q = *ctx.stream();
    xmx_q4k_quant_act(static_cast<const float *>(act->data), xa, d8, us, (int) K, (int) M, q, true);
    const int ks = xmx_q4k_pick_ks((int) N, (int) K, (int) M);   // the single-weight launches' ks (N < 65536 here)
    ggml_sycl_xmx_q4k_direct_pair_launch(a.qs, a.sc, a.dm, b.qs, b.sc, b.dm, xa, d8, us, outG, outU, (int) N, (int) K,
                                         (int) M, ldd, ks, glu, q);
    return true;
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    GGML_SYCL_EXIT_OR_RETHROW();
    return false;
}

void ggml_sycl_op_mul_mat_xmx_q4k(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                  ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                  const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low,
                                  const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_row_size,
                                  const queue_ptr & stream) try {
    GGML_UNUSED(src1_ddf_i);
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2, " : XMX q4_K (legacy driver path)");

    const int64_t K  = src0->ne[0];
    const int64_t N  = src0->ne[1];
    const int64_t M  = src1_ncols;
    const int64_t G  = K / QK8_1;
    GGML_ASSERT(row_low == 0 && row_high == N);   // no split: the whole weight, dst column stride dst->ne[0]
    GGML_ASSERT(src1->ne[0] == K && dst->ne[0] == N);
    GGML_ASSERT(K % QK_K == 0 && N % XMX_Q4K_ROWS == 0);
    GGML_ASSERT(M >= 1 && M <= 8);
    GGML_ASSERT(src1_ddq_i != nullptr);
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    GGML_ASSERT(extra && extra->optimized_feature.reorder);

    const xmx_q4k_weights w = xmx_q4k_weights_of(src0_dd_i, N, K);

    // activations: the driver's SoA q8_1 (quantize_and_reorder_q8_1_soa), column stride as MMVQ's stride_col_y_bytes
    const size_t stride_y = (size_t) src1_padded_row_size * sizeof(block_q8_1) / QK8_1;

    const size_t xa_bytes = (size_t) XQ_TN * K;
    const size_t d8_bytes = (size_t) XQ_TN * G * sizeof(sycl::half);
    const size_t us_bytes = (size_t) XQ_TN * G * sizeof(int32_t);
    ggml_sycl_pool_alloc<char> act_alloc(ctx.pool(), xa_bytes + d8_bytes + us_bytes);
    int8_t *  xa = reinterpret_cast<int8_t *>(act_alloc.get());
    sycl::half * d8 = reinterpret_cast<sycl::half *>(act_alloc.get() + xa_bytes);
    int32_t * us = reinterpret_cast<int32_t *>(act_alloc.get() + xa_bytes + d8_bytes);

    xmx_q4k_pack_act(src1_ddq_i, stride_y, xa, d8, us, (int) K, (int) M, *stream);
    xmx_q4k_matmul(w, xa, d8, us, dst_dd_i, (int) N, (int) K, (int) M, dst->ne[0], *stream);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    GGML_SYCL_EXIT_OR_RETHROW();
}

#else  // !GGML_SYCL_XMX

bool ggml_sycl_xmx_q4k_device_ok(int) { return false; }

bool ggml_sycl_xmx_q4k_gateup(ggml_backend_sycl_context &, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *,
                              float *, float *, int64_t, bool) {
    return false;
}

void ggml_sycl_mul_mat_xmx_q4k(ggml_backend_sycl_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {
    GGML_ABORT("XMX q4_K path not built (configure with -DGGML_SYCL_XMX=ON)");
}

void ggml_sycl_op_mul_mat_xmx_q4k(ggml_backend_sycl_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *,
                                  const char *, const float *, const char *, float *, const int64_t, const int64_t,
                                  const int64_t, const int64_t, const queue_ptr &) {
    GGML_ABORT("XMX q4_K path not built (configure with -DGGML_SYCL_XMX=ON)");
}

#endif
