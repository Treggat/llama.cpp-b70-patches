//
// q6_K x q8_1 matmul on the Xe2 matrix units - see mmq-xmx-q6k.hpp
//
// Kernel: b70-secret-sauce/xmx-probe/xmx_q6k.cpp defaults (SHIFT 1, D8G 1, PIPE/PF2/SCG/BZ/ICOMB 0, MC 8, 256-GRF
// mode), host harness removed. The device capability check and the build switch are the q4_K path's.
//
#include "mmq-xmx-q6k.hpp"
#include "mmq-xmx-q4k.hpp"
#include "mmq-xmx-direct.hpp"
#include "mmq-xmx-q80.hpp"
#include "quants.hpp"
#include "quantize.hpp"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#if GGML_SYCL_XMX
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#endif

static int ggml_sycl_xmx_q6k_env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e && *e ? atoi(e) : def;
}

bool ggml_sycl_xmx_q6k_env() {
    static const bool v = ggml_sycl_xmx_q6k_env_int("GGML_SYCL_XMX_Q6K", 0) != 0;   // default off
    return v;
}

bool ggml_sycl_xmx_q6k_direct_env() {
    static const bool v = ggml_sycl_xmx_q6k_env_int("GGML_SYCL_XMX_Q6K_DIRECT", 0) != 0;   // default off
    return v;
}

// kernel tile: 16 weight rows per sub-group, up to 8 activation columns (the A tile's M); LOCAL (longdraft): a second
// kernel instance runs two A tiles (9..16 columns) against every staged B tile, opt-in (GGML_SYCL_XMX_WIDE=1 or
// GGML_SYCL_XMX_Q6K_MAX_COLS=9..16); 1..8 columns keep the one-tile kernel unchanged
static constexpr int XMX_Q6K_ROWS     = 16;
static constexpr int XMX_Q6K_MAX_COLS = 16;

static int ggml_sycl_xmx_q6k_min_cols() {
    // direct kernel (GGML_SYCL_XMX_Q6K_DIRECT=1): 2, for weights of at least BIG_MB only (see can_use)
    static const int v = std::max(1, ggml_sycl_xmx_q6k_env_int("GGML_SYCL_XMX_Q6K_MIN_COLS",
                                                                ggml_sycl_xmx_q6k_direct_env() ? 2 : 3));
    return v;
}

static int ggml_sycl_xmx_q6k_max_cols() {
    // the legacy driver path (GGML_SYCL_XMX_Q6K_PATH=1) only has the one-tile kernel
    static const int v = std::min(ggml_sycl_xmx_q6k_path() == 0 ? XMX_Q6K_MAX_COLS : 8,
                                  ggml_sycl_xmx_q6k_env_int("GGML_SYCL_XMX_Q6K_MAX_COLS", ggml_sycl_xmx_wide() ? 16 : 8));
    return v;
}

// Minimum-work guards. Every XMX launch pays a fixed ~10-14 us per op in test-backend-ops perf (the 256-GRF kernel
// forces a GRF-mode switch against the 128-GRF quantize/pack kernels around it; 128-GRF spills and runs 2-3x slower),
// while MMVQ grows per column. Measured on the B70 (test-backend-ops perf, us, MMVQ n=3/4/5 vs XMX, flat in n):
//   1024x5120   (4.3 MB): 13.1 / 15.2 / 17.9 vs 24.2  -> MMVQ up to 7 columns: weights under MIN_MB stay on MMVQ
//   10240x5120 (43 MB):   86.9 / 99.4 / 126  vs ~96   -> MMVQ at 3, tie at 4: weights under BIG_MB need
//   4096x14336 (48 MB):   91.9 / 102  / 114  vs ~113  -> SMALL_MIN_COLS (5) columns
//   5120x17408 (73 MB):   184  / 226  / 262  vs ~163  -> XMX from 3 columns
static size_t ggml_sycl_xmx_q6k_min_bytes() {
    static const size_t v = (size_t) std::max(0, ggml_sycl_xmx_q6k_env_int("GGML_SYCL_XMX_Q6K_MIN_MB", 8)) << 20;
    return v;
}

static size_t ggml_sycl_xmx_q6k_big_bytes() {
    static const size_t v = (size_t) std::max(0, ggml_sycl_xmx_q6k_env_int("GGML_SYCL_XMX_Q6K_BIG_MB", 64)) << 20;
    return v;
}

static int ggml_sycl_xmx_q6k_small_min_cols() {
    // direct kernel: 3 (test-backend-ops perf, us, 10240x5120 MMVQ vs direct at 3 / 4 columns: 91.4 / 81.3, 108.6 / 81.9;
    // at 2 columns MMVQ 77.0 vs 80.5)
    static const int v = std::max(1, ggml_sycl_xmx_q6k_env_int("GGML_SYCL_XMX_Q6K_SMALL_MIN_COLS",
                                                                ggml_sycl_xmx_q6k_direct_env() ? 3 : 5));
    return v;
}

bool ggml_sycl_xmx_q6k_can_use(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               const ggml_tensor * dst) {
    if (!ggml_sycl_xmx_q4k_built() || !ggml_sycl_xmx_q6k_env()) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q6_K || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->op != GGML_OP_MUL_MAT) {   // MUL_MAT_ID comes through the same dispatcher with a stack dst
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    // a non-contiguous src1 is re-quantized per slice with the AoS q8_1 layout, which this path cannot read
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src1->ne[1] < ggml_sycl_xmx_q6k_min_cols() || src1->ne[1] > ggml_sycl_xmx_q6k_max_cols()) {
        return false;
    }
    if (src0->ne[0] % QK_K != 0 || src0->ne[1] % XMX_Q6K_ROWS != 0 || src1->ne[0] != src0->ne[0] ||
        dst->ne[0] != src0->ne[1] || ggml_nbytes(src0) >= (size_t) INT_MAX) {   // reorder offset helpers use int
        return false;
    }
    // minimum work (see the guards above): small weights, and mid-size weights at few columns, are faster on MMVQ.
    // Above 8 columns MMVQ is out of reach (MMVQ_MAX_BATCH_SIZE) and the alternative is dequantize + f16 GEMM, so the
    // guards only apply up to 8 columns.
    if (src1->ne[1] <= 8 && ggml_nbytes(src0) < ggml_sycl_xmx_q6k_min_bytes()) {
        return false;
    }
    if (src1->ne[1] <= 8 && ggml_nbytes(src0) < ggml_sycl_xmx_q6k_big_bytes() &&
        src1->ne[1] < ggml_sycl_xmx_q6k_small_min_cols()) {
        return false;
    }
    // direct kernel at 2 columns: only weights of at least BIG_MB (5120x17408 ffn_down: MMVQ 152.0 vs direct 140.9 us)
    if (src1->ne[1] < 3 && ggml_nbytes(src0) < ggml_sycl_xmx_q6k_big_bytes()) {
        return false;
    }
    // with DMMV prioritised the reorder of K-quants is not used by the rest of the backend
    if (g_ggml_sycl_prioritize_dmmv) {
        return false;
    }
    if (src0->extra == nullptr || src0->buffer == nullptr) {   // split buffers are excluded by the dispatcher
        return false;
    }
    // tensors used mostly at one column (a draft/MTP layer) can be kept on MMVQ, e.g. "blk.64."
    static const char * skip_prefix = getenv("GGML_SYCL_XMX_Q6K_SKIP_PREFIX");
    if (skip_prefix && *skip_prefix && strncmp(src0->name, skip_prefix, strlen(skip_prefix)) == 0) {
        return false;
    }
    return ggml_sycl_xmx_q4k_device_ok(ctx.device);   // the same s8 8x16x32 DPAS combination
}

#if GGML_SYCL_XMX

namespace mx  = sycl::ext::oneapi::experimental::matrix;
namespace imx = sycl::ext::intel::experimental::matrix;

namespace {

constexpr int X6_TM = 8, X6_TN = 16, X6_TK = 32, X6_SG = 16;
constexpr int X6_MC = 8;                  // columns of the one-tile kernel (MC = 16: two A tiles)
constexpr int X6_PB = 128;                // SLM bytes per row per staged part (half a block)

// Activation pack from the driver's SoA q8_1 buffer (quantize_and_reorder_q8_1_soa): column c starts at
// y + c*stride_y, its K int8 quants at [0, K), its K/32 half2 (d, s) at K + g*4. Produces
//   xa [8][2K] int8: per 32-group 64 bytes (u0..u15, 0 x16, 0 x16, u16..u31), so the A tile at +0 carries the
//                    first 16-element scale group only and the one at +32 the second; columns >= M zero
//   d8 [G][8]  half: the d MMVQ sees, group-major (one coalesced uint2 per lane per block); columns >= M zero.
//                    Kept as half in memory: a float -> half -> float round trip in registers is folded away under
//                    the backend's fast-math flags (see the q4_K path), which would hand the kernel the unrounded d.
void xmx_q6k_pack_act(const char * y, size_t stride_y, int8_t * xa, sycl::half * d8, int K, int M, const queue_ptr & stream) {
    const int G = K / QK8_1;
    stream->parallel_for(sycl::range<1>((size_t) G * X6_TM), [=](sycl::id<1> idx) {
        const int g = (int) (idx[0] / X6_TM);
        const int c = (int) (idx[0] % X6_TM);
        int8_t *  dst = xa + (size_t) c * 2 * K + (size_t) g * 64;
        const sycl::uint4 z(0, 0, 0, 0);
        if (c < M) {
            const uint8_t *   col = reinterpret_cast<const uint8_t *>(y) + (size_t) c * stride_y;
            const sycl::uint4 lo  = *reinterpret_cast<const sycl::uint4 *>(col + (size_t) g * QK8_1);
            const sycl::uint4 hi  = *reinterpret_cast<const sycl::uint4 *>(col + (size_t) g * QK8_1 + 16);
            *reinterpret_cast<sycl::uint4 *>(dst)      = lo;
            *reinterpret_cast<sycl::uint4 *>(dst + 16) = z;
            *reinterpret_cast<sycl::uint4 *>(dst + 32) = z;
            *reinterpret_cast<sycl::uint4 *>(dst + 48) = hi;
            const sycl::half2 ds = *reinterpret_cast<const sycl::half2 *>(col + K + (size_t) g * sizeof(sycl::half2));
            d8[(size_t) g * X6_TM + c] = ds.x();
        } else {
            *reinterpret_cast<sycl::uint4 *>(dst)      = z;
            *reinterpret_cast<sycl::uint4 *>(dst + 16) = z;
            *reinterpret_cast<sycl::uint4 *>(dst + 32) = z;
            *reinterpret_cast<sycl::uint4 *>(dst + 48) = z;
            d8[(size_t) g * X6_TM + c] = sycl::half(0.f);
        }
    });
}

// Fused f32 -> operands quantizer (default path, GGML_SYCL_XMX_Q6K_PATH=0), the q6_K sibling of xmx_q4k_quant_act:
// one sub-group per (column, 32-group) with the geometry of quantize_row_q8_1_sycl, calling MMVQ's own
// quantize_q8_1_impl (so the int8 quants and d are bit-identical to quantize_and_reorder_q8_1_soa's), written
// straight into the xa / d8 layout above. Runs in the matmul's 256-register mode, so the op pays no register-mode
// switch between its two kernels (the legacy path ran a 128-GRF quantize, a default-GRF pack and the 256-GRF matmul).
// Columns >= M are left unwritten: they only feed accumulator rows the kernel never stores (integer DPAS, per-row).
// gmajor: xa in the direct kernel's group-major layout [G][2 tiles][8 columns][32]: tile 0 = elements 0..15 then 16
// zeros, tile 1 = 16 zeros then elements 16..31 (the same two A tiles, 512 B per group, one block read each)
// DC: columns per d8 row (8 for the one-tile kernel; 16 for the two-tile kernel, LOCAL longdraft). With DC = 16 the
// group-major layout is [G][2][16][32] (LOCAL wideverify, the 9..16-column register-fed kernel); DC = 8 is unchanged.
template <int DC = X6_TM>
sycl::event xmx_q6k_quant_act(const float * x, int8_t * xa, sycl::half * d8, int K, int M, sycl::queue & q,
                              bool gmajor = false) {
    constexpr int EPW = QK8_1 / WARP_SIZE;
    static_assert(EPW * WARP_SIZE == QK8_1 && 16 % EPW == 0, "lane split of a 32-group");
    const int G = K / QK8_1;
    return q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) M * G * WARP_SIZE), sycl::range<1>(WARP_SIZE)),
                          sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        sycl::vec<int8_t, EPW> qv;
        float d = 0.0f, sum = 0.0f;
        quantize_q8_1_impl<EPW>(x, qv, d, sum, it);
        const size_t blk  = it.get_group(0);   // c * G + g
        const int    c    = (int) (blk / G);
        const int    g    = (int) (blk % G);
        const int    lane = (int) it.get_local_id(0);
        const int    e    = lane * EPW;        // this lane's first element of the group
        if (gmajor) {
            int8_t * dst = xa + (size_t) g * (DC * 64) + (size_t) c * 32;
            *reinterpret_cast<sycl::vec<int8_t, EPW> *>(dst + (e < 16 ? 0 : DC * 32) + e) = qv;
            *reinterpret_cast<sycl::vec<int8_t, EPW> *>(dst + (e < 16 ? DC * 32 : 0) + e) = sycl::vec<int8_t, EPW>(0);
        } else {
            int8_t * dst = xa + (size_t) c * 2 * K + (size_t) g * 64;
            // elements 0..15 at +0, 16..31 at +48; bytes 16..47 zero (EPW per lane covers 32 bytes over the sub-group)
            *reinterpret_cast<sycl::vec<int8_t, EPW> *>(dst + (e < 16 ? e : e + 32)) = qv;
            *reinterpret_cast<sycl::vec<int8_t, EPW> *>(dst + 16 + e) = sycl::vec<int8_t, EPW>(0);
        }
        if (lane == 0) {
            d8[(size_t) g * DC + c] = sycl::half(d);
        }
    });
}

// split-K: aim for >= 3200 sub-groups with >= 2 blocks each, at most 10 sub-groups per work-group. Measured on
// the B70 at 6 columns (test-backend-ops perf, us): 5120x17408 ks 8 172 / ks 10 161.5; 10240x5120 ks 4 103 /
// ks 5 96.3 / ks 10 98.9; 4096x14336 ks 5 115.7 / ks 10 112.8 / ks 13 124; 1024x5120 ks 5 31.1 / ks 10 24.3
// (the old >= 2560 sub-groups target gave 8 / 4 / 10 / 10)
int xmx_q6k_pick_ks(int N, int K) {
    const int KB    = K / QK_K;
    const int tiles = N / XMX_Q6K_ROWS;
    int ks = std::max(1, std::min(std::min((3200 + tiles - 1) / tiles, 10), KB / 2));
    // GGML_SYCL_XMX_KS_V2=1 (LOCAL, default off): 5120x17408 (ffn_down) ks 3: 158-160 -> 144-156 us at 3..8 columns
    // (ks 10: 3 work-groups per Xe core, 3.3 waves; ks 3: 10 per core, one wave)
    static const bool ks_v2 = ggml_sycl_xmx_q6k_env_int("GGML_SYCL_XMX_KS_V2", 0) != 0;
    if (ks_v2 && N == 5120 && K == 17408) {
        ks = 3;
    }
    static const int ks_env = ggml_sycl_xmx_q6k_env_int("GGML_SYCL_XMX_Q6K_KS", 0);
    if (ks_env > 0) {
        ks = ks_env;
    }
    return std::max(1, std::min(ks, std::min(16, KB)));
}

// weights, read in place from the q6_K reorder layout (reorder_qw_q6_k): the section offsets are the ones
// ggml_sycl_reordered::block_q_t<GGML_TYPE_Q6_K> gives MMVQ with nrows = N (block 0 of each section)
struct xmx_q6k_weights {
    const uint8_t *  ql;   // [N*KB][128]
    const uint8_t *  qh;   // [N*KB][64]
    const int8_t *   sc;   // [N*KB][16]
    const uint16_t * dd;   // [N*KB] half
};

xmx_q6k_weights xmx_q6k_weights_of(const char * src0_dd, int64_t N, int64_t K) {
    using q6k_reordered = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q6_K>;
    const int64_t KB = K / QK_K;
    GGML_ASSERT(N * KB * (int64_t) sizeof(block_q6_K) < (int64_t) INT_MAX);   // helper's int math
    const auto q_off = q6k_reordered::get_block_offset(0, (int) (N * KB));
    const auto d_off = q6k_reordered::get_d_offset((int) N, (int) K, 0);
    return { reinterpret_cast<const uint8_t *>(src0_dd) + q_off.first,
             reinterpret_cast<const uint8_t *>(src0_dd) + q_off.second,
             reinterpret_cast<const int8_t *>(src0_dd) + d_off.first,
             reinterpret_cast<const uint16_t *>(src0_dd + d_off.second) };
}

// staged s8 weight bytes for one dword of ql and qh; T = quarter (0..3) of a 128-element half.
// 4 (q - 32) = (4q) ^ 0x80 per byte, 4q = (4-bit part << 2) | (2-bit part << 6); the 4 is folded into d.
template <int T> inline uint32_t xmx_q6k_stage(uint32_t ql, uint32_t qh) {
    const uint32_t lo = T < 2 ? (ql << 2) & 0x3C3C3C3Cu : (ql >> 2) & 0x3C3C3C3Cu;
    const uint32_t hi = (qh << (6 - 2 * T)) & 0xC0C0C0C0u;
    return (lo | hi) ^ 0x80808080u;
}

// The kernel (xmx_q6k.cpp). Transposed roles as in the q4_K kernel: activations are the A tile (M = 8 columns x 32),
// weights the B tile (32 x 16 rows, col_major from a private SLM slice per sub-group, [row][k], 128 B per row per
// part), so every accumulator lane is one weight row with its 16 scales and d in registers; element i = column i.
// Two parts per block (a half = 128 elements = 4 x 32-groups); the next part's ql/qh are prefetched into registers
// while the current part runs on DPAS. Staging unit (row, j): 16 B of ql[64h + 16j], 16 B of ql[64h + 32 + 16j],
// 16 B of qh[32h + 16j] -> the four 16-element runs 16j + {0, 32, 64, 96} of the half (the ggml q6_K bit layout:
// element 32t + l of half h is ((ql[64h + 32(t&1) + l] >> 4(t>>1)) & 15) | ((qh[32h + l] >> 2t) & 3) << 4).
// Per 32-group g, two DPAS on the same B: lo16 = dot over elements 0..15, hi16 = over 16..31 (the xa layout), each
// 4 * dot(q - 32, u) exactly; then, per column i (lane n = row n):
//   Fb += fma((float) hi16, sc[2g+1], (float) lo16 * sc[2g]) * d8[i][g]        F += (d / 4) * Fb once per block
// (float) of the integer dots is exact and the scale products are exact up to |4 dot sc| < 2^24 (beyond only for
// extreme inputs); only float rounding order differs from MMVQ. Split-K: a work-group is `ks` sub-groups on the same
// 16 rows, each taking a contiguous block range; partial sums are reduced in SLM in a fixed order (deterministic).
//   ql [N*KB][128], qh [N*KB][64], sc [N*KB][16] int8, dd [N*KB] half: the q6_K reorder layout
//   out: column i of row r at out[i*ldd + r], for i < M
sycl::event xmx_q6k_launch(const uint8_t * dQl, const uint8_t * dQh, const int8_t * dSc, const uint16_t * dD,
                           const int8_t * dXa, const sycl::half * dD8, float * dOut, int N, int K, int M, int64_t ldd,
                           int ks, sycl::queue & q) {
    const int KB = K / QK_K;
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<int8_t, 1> bslm(sycl::range<1>(ks * 16 * X6_PB), h);
        sycl::local_accessor<float, 1>  red(sycl::range<1>(ks > 1 ? ks * 8 * 16 : 1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (N / 16) * ks * X6_SG), sycl::range<1>(ks * X6_SG)),
                       sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(X6_SG)]] {
            using sycl::access::address_space;
            using sycl::access::decorated;
            sycl::sub_group sg = it.get_sub_group();
            const int lane = sg.get_local_id()[0];
            const int s    = sg.get_group_id()[0];
            const int row0 = (int) it.get_group(0) * 16;
            const int b0   = s * KB / ks, b1 = (s + 1) * KB / ks;
            auto bp = bslm.get_multi_ptr<decorated::no>() + s * 16 * X6_PB;

            const int j  = lane & 1;
            const int ra = lane >> 1;                   // unit 0 row; unit 1 row = ra + 8
            const uint8_t * gl = dQl + (size_t) (row0 + ra) * KB * 128 + j * 16;
            const uint8_t * gh = dQh + (size_t) (row0 + ra) * KB * 64 + j * 16;
            const size_t    rl = (size_t) 8 * KB * 128, rh = (size_t) 8 * KB * 64;   // +8 rows
            const int       so = ra * X6_PB + j * 16;
            const sycl::uint4 * hsc = reinterpret_cast<const sycl::uint4 *>(dSc + (size_t) (row0 + lane) * KB * 16);
            const uint16_t *    hd  = dD + (size_t) (row0 + lane) * KB;
            const int8_t *      xa  = dXa;
            // d8 [G][8] half: lane l's 4 halves (one uint2) of block b = group l/2, columns 4(l%2)..+3; column i of
            // group g is component i%4 of lane 2g + i/4
            const sycl::uint2 * d8base = reinterpret_cast<const sycl::uint2 *>(dD8) + lane;
            auto d8f = [](uint32_t w, int hi) -> float {
                return (float) sycl::bit_cast<sycl::half>((uint16_t) (hi ? (w >> 16) : (w & 0xFFFF)));
            };

            auto ldq = [](const uint8_t * p) -> sycl::uint4 { return *reinterpret_cast<const sycl::uint4 *>(p); };
            sycl::uint4 va[2], vb[2], vh[2];            // next part: ql run a, ql run b, qh, per unit
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                va[u] = ldq(gl + u * rl + b0 * 128);
                vb[u] = ldq(gl + u * rl + b0 * 128 + 32);
                vh[u] = ldq(gh + u * rh + b0 * 64);
            }
            sycl::uint4  hs  = hsc[b0];
            uint16_t     hdd = hd[b0];
            sycl::uint2  d8n = d8base[(size_t) b0 * 16];

            float F[8];
            for (int i = 0; i < 8; ++i) { F[i] = 0.f; }
            for (int b = b0; b < b1; ++b) {
                float scf[16];
                {
                    const uint32_t w[4] = { hs.x(), hs.y(), hs.z(), hs.w() };
#pragma unroll
                    for (int t = 0; t < 16; ++t) { scf[t] = (float) (int8_t) (w[t >> 2] >> (8 * (t & 3))); }
                }
                const float dq     = (float) sycl::bit_cast<sycl::half>(hdd) * 0.25f;   // exact
                const float d8c[4] = { d8f(d8n.x(), 0), d8f(d8n.x(), 1), d8f(d8n.y(), 0), d8f(d8n.y(), 1) };
                {   // next block's header (clamped, unconditional)
                    const int bn = b + 1 < b1 ? b + 1 : b;
                    hs  = hsc[bn];
                    hdd = hd[bn];
                    d8n = d8base[(size_t) bn * 16];
                }
                float Fb[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) { Fb[i] = 0.f; }
#pragma unroll
                for (int part = 0; part < 2; ++part) {
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        sycl::uint4 o0, o1, o2, o3;
                        o0.x() = xmx_q6k_stage<0>(va[u].x(), vh[u].x()); o0.y() = xmx_q6k_stage<0>(va[u].y(), vh[u].y());
                        o0.z() = xmx_q6k_stage<0>(va[u].z(), vh[u].z()); o0.w() = xmx_q6k_stage<0>(va[u].w(), vh[u].w());
                        o1.x() = xmx_q6k_stage<1>(vb[u].x(), vh[u].x()); o1.y() = xmx_q6k_stage<1>(vb[u].y(), vh[u].y());
                        o1.z() = xmx_q6k_stage<1>(vb[u].z(), vh[u].z()); o1.w() = xmx_q6k_stage<1>(vb[u].w(), vh[u].w());
                        o2.x() = xmx_q6k_stage<2>(va[u].x(), vh[u].x()); o2.y() = xmx_q6k_stage<2>(va[u].y(), vh[u].y());
                        o2.z() = xmx_q6k_stage<2>(va[u].z(), vh[u].z()); o2.w() = xmx_q6k_stage<2>(va[u].w(), vh[u].w());
                        o3.x() = xmx_q6k_stage<3>(vb[u].x(), vh[u].x()); o3.y() = xmx_q6k_stage<3>(vb[u].y(), vh[u].y());
                        o3.z() = xmx_q6k_stage<3>(vb[u].z(), vh[u].z()); o3.w() = xmx_q6k_stage<3>(vb[u].w(), vh[u].w());
                        const int o = so + u * 8 * X6_PB;
                        *reinterpret_cast<sycl::uint4 *>(&bp[o])      = o0;
                        *reinterpret_cast<sycl::uint4 *>(&bp[o + 32]) = o1;
                        *reinterpret_cast<sycl::uint4 *>(&bp[o + 64]) = o2;
                        *reinterpret_cast<sycl::uint4 *>(&bp[o + 96]) = o3;
                    }
                    {   // prefetch the next part (clamped, unconditional: no phi copies)
                        const int nb = part + 1 < 2 ? b : (b + 1 < b1 ? b + 1 : b);
                        const int np = part + 1 < 2 ? part + 1 : (b + 1 < b1 ? 0 : part);
#pragma unroll
                        for (int u = 0; u < 2; ++u) {
                            va[u] = ldq(gl + u * rl + nb * 128 + np * 64);
                            vb[u] = ldq(gl + u * rl + nb * 128 + np * 64 + 32);
                            vh[u] = ldq(gh + u * rh + nb * 64 + np * 32);
                        }
                    }
                    sycl::group_barrier(sg);
#pragma unroll
                    for (int gg = 0; gg < 4; ++gg) {
                        const int g = part * 4 + gg;
                        mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, X6_TM, X6_TK, mx::layout::row_major> Alo, Ahi;
                        mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, X6_TK, X6_TN, mx::layout::col_major> B;
                        const int8_t * xg = xa + (size_t) (b * 8 + g) * 64;
                        mx::joint_matrix_load(sg, Alo, sycl::address_space_cast<address_space::global_space, decorated::no>(xg), 2 * K);
                        mx::joint_matrix_load(sg, Ahi, sycl::address_space_cast<address_space::global_space, decorated::no>(xg + 32), 2 * K);
                        mx::joint_matrix_load(sg, B, bp + gg * X6_TK, X6_PB);
                        mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, X6_TM, X6_TN> Clo, Chi;
                        mx::joint_matrix_fill(sg, Clo, 0);
                        mx::joint_matrix_fill(sg, Chi, 0);
                        mx::joint_matrix_mad(sg, Clo, Alo, B, Clo);
                        mx::joint_matrix_mad(sg, Chi, Ahi, B, Chi);
                        // column-side d8 broadcast from lane 2g + i/4 (register regions); these sub-group ops must
                        // stay outside the joint_matrix_apply lambdas
                        float db[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) { db[i] = sycl::select_from_group(sg, d8c[i & 3], 2 * g + (i >> 2)); }
                        float lo[8];
                        {
                            int i = 0;
                            imx::joint_matrix_apply(sg, Clo, [&](int32_t & v, size_t, size_t) {
                                if (i < X6_MC) { lo[i] = (float) v * scf[2 * g]; }
                                ++i;
                            });
                        }
                        {
                            int i = 0;
                            imx::joint_matrix_apply(sg, Chi, [&](int32_t & v, size_t, size_t) {
                                if (i < X6_MC) { Fb[i] = sycl::fma(sycl::fma((float) v, scf[2 * g + 1], lo[i]), db[i], Fb[i]); }
                                ++i;
                            });
                        }
                    }
                    sycl::group_barrier(sg);
                }
#pragma unroll
                for (int i = 0; i < X6_MC; ++i) { F[i] = sycl::fma(dq, Fb[i], F[i]); }
            }
            if (ks > 1) {
#pragma unroll
                for (int i = 0; i < X6_MC; ++i) { red[(s * 8 + i) * 16 + lane] = F[i]; }
                sycl::group_barrier(it.get_group());
                if (s == 0) {
#pragma unroll
                    for (int i = 0; i < X6_MC; ++i) {
                        float acc = red[i * 16 + lane];
                        for (int t = 1; t < ks; ++t) { acc += red[(t * 8 + i) * 16 + lane]; }
                        F[i] = acc;
                    }
                }
            }
            if (s == 0) {
#pragma unroll
                for (int i = 0; i < X6_MC; ++i) {
                    if (i < M) { dOut[(size_t) i * ldd + row0 + lane] = F[i]; }
                }
            }
        });
    });
}

// LOCAL (longdraft): the kernel above as a template on the column count; instantiated only for MC = 16 (9..16
// columns), so the 1..8-column kernel stays the unmodified one-tile code
template <int MC>
sycl::event xmx_q6k_launch_wide(const uint8_t * dQl, const uint8_t * dQh, const int8_t * dSc, const uint16_t * dD,
                           const int8_t * dXa, const sycl::half * dD8, float * dOut, int N, int K, int M, int64_t ldd,
                           int ks, sycl::queue & q) {
    // MC = 8: one A tile (columns 0..7), d8 [G][8]. MC = 16 (LOCAL longdraft): two A tiles (columns 0..7, 8..15) on
    // every staged B tile, xa [16][2K], d8 [G][16]; per column the float operations and their order are the MC = 8
    // kernel's.
    static_assert(MC == 8 || MC == 16, "one or two A tiles");
    constexpr int NT = MC / X6_TM;
    const int KB = K / QK_K;
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<int8_t, 1> bslm(sycl::range<1>(ks * 16 * X6_PB), h);
        sycl::local_accessor<float, 1>  red(sycl::range<1>(ks > 1 ? ks * MC * 16 : 1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (N / 16) * ks * X6_SG), sycl::range<1>(ks * X6_SG)),
                       sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(X6_SG)]] {
            using sycl::access::address_space;
            using sycl::access::decorated;
            sycl::sub_group sg = it.get_sub_group();
            const int lane = sg.get_local_id()[0];
            const int s    = sg.get_group_id()[0];
            const int row0 = (int) it.get_group(0) * 16;
            const int b0   = s * KB / ks, b1 = (s + 1) * KB / ks;
            auto bp = bslm.get_multi_ptr<decorated::no>() + s * 16 * X6_PB;

            const int j  = lane & 1;
            const int ra = lane >> 1;                   // unit 0 row; unit 1 row = ra + 8
            const uint8_t * gl = dQl + (size_t) (row0 + ra) * KB * 128 + j * 16;
            const uint8_t * gh = dQh + (size_t) (row0 + ra) * KB * 64 + j * 16;
            const size_t    rl = (size_t) 8 * KB * 128, rh = (size_t) 8 * KB * 64;   // +8 rows
            const int       so = ra * X6_PB + j * 16;
            const sycl::uint4 * hsc = reinterpret_cast<const sycl::uint4 *>(dSc + (size_t) (row0 + lane) * KB * 16);
            const uint16_t *    hd  = dD + (size_t) (row0 + lane) * KB;
            const int8_t *      xa  = dXa;
            // MC 8:  d8 [G][8] half: lane l's 4 halves (one uint2) of block b = group l/2, columns 4(l%2)..+3; column i
            //        of group g is component i%4 of lane 2g + i/4
            // MC 16: d8 [G][16] half: lane l's 8 halves (one uint4) = group l/2, columns 8(l%2)..+7; column 8t + i of
            //        group g is component i of lane 2g + t
            using d8v_t = std::conditional_t<MC == 8, sycl::uint2, sycl::uint4>;
            constexpr int ND8 = MC == 8 ? 4 : 8;
            const d8v_t * d8base = reinterpret_cast<const d8v_t *>(dD8) + lane;
            auto d8f = [](uint32_t w, int hi) -> float {
                return (float) sycl::bit_cast<sycl::half>((uint16_t) (hi ? (w >> 16) : (w & 0xFFFF)));
            };

            auto ldq = [](const uint8_t * p) -> sycl::uint4 { return *reinterpret_cast<const sycl::uint4 *>(p); };
            sycl::uint4 va[2], vb[2], vh[2];            // next part: ql run a, ql run b, qh, per unit
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                va[u] = ldq(gl + u * rl + b0 * 128);
                vb[u] = ldq(gl + u * rl + b0 * 128 + 32);
                vh[u] = ldq(gh + u * rh + b0 * 64);
            }
            sycl::uint4  hs  = hsc[b0];
            uint16_t     hdd = hd[b0];
            d8v_t        d8n = d8base[(size_t) b0 * 16];

            float F[MC];
            for (int i = 0; i < MC; ++i) { F[i] = 0.f; }
            for (int b = b0; b < b1; ++b) {
                float scf[16];
                {
                    const uint32_t w[4] = { hs.x(), hs.y(), hs.z(), hs.w() };
#pragma unroll
                    for (int t = 0; t < 16; ++t) { scf[t] = (float) (int8_t) (w[t >> 2] >> (8 * (t & 3))); }
                }
                const float dq = (float) sycl::bit_cast<sycl::half>(hdd) * 0.25f;   // exact
                float d8c[ND8];
                if constexpr (MC == 8) {
                    d8c[0] = d8f(d8n.x(), 0); d8c[1] = d8f(d8n.x(), 1); d8c[2] = d8f(d8n.y(), 0); d8c[3] = d8f(d8n.y(), 1);
                } else {
                    d8c[0] = d8f(d8n.x(), 0); d8c[1] = d8f(d8n.x(), 1); d8c[2] = d8f(d8n.y(), 0); d8c[3] = d8f(d8n.y(), 1);
                    d8c[4] = d8f(d8n.z(), 0); d8c[5] = d8f(d8n.z(), 1); d8c[6] = d8f(d8n.w(), 0); d8c[7] = d8f(d8n.w(), 1);
                }
                {   // next block's header (clamped, unconditional)
                    const int bn = b + 1 < b1 ? b + 1 : b;
                    hs  = hsc[bn];
                    hdd = hd[bn];
                    d8n = d8base[(size_t) bn * 16];
                }
                float Fb[MC];
#pragma unroll
                for (int i = 0; i < MC; ++i) { Fb[i] = 0.f; }
#pragma unroll
                for (int part = 0; part < 2; ++part) {
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        sycl::uint4 o0, o1, o2, o3;
                        o0.x() = xmx_q6k_stage<0>(va[u].x(), vh[u].x()); o0.y() = xmx_q6k_stage<0>(va[u].y(), vh[u].y());
                        o0.z() = xmx_q6k_stage<0>(va[u].z(), vh[u].z()); o0.w() = xmx_q6k_stage<0>(va[u].w(), vh[u].w());
                        o1.x() = xmx_q6k_stage<1>(vb[u].x(), vh[u].x()); o1.y() = xmx_q6k_stage<1>(vb[u].y(), vh[u].y());
                        o1.z() = xmx_q6k_stage<1>(vb[u].z(), vh[u].z()); o1.w() = xmx_q6k_stage<1>(vb[u].w(), vh[u].w());
                        o2.x() = xmx_q6k_stage<2>(va[u].x(), vh[u].x()); o2.y() = xmx_q6k_stage<2>(va[u].y(), vh[u].y());
                        o2.z() = xmx_q6k_stage<2>(va[u].z(), vh[u].z()); o2.w() = xmx_q6k_stage<2>(va[u].w(), vh[u].w());
                        o3.x() = xmx_q6k_stage<3>(vb[u].x(), vh[u].x()); o3.y() = xmx_q6k_stage<3>(vb[u].y(), vh[u].y());
                        o3.z() = xmx_q6k_stage<3>(vb[u].z(), vh[u].z()); o3.w() = xmx_q6k_stage<3>(vb[u].w(), vh[u].w());
                        const int o = so + u * 8 * X6_PB;
                        *reinterpret_cast<sycl::uint4 *>(&bp[o])      = o0;
                        *reinterpret_cast<sycl::uint4 *>(&bp[o + 32]) = o1;
                        *reinterpret_cast<sycl::uint4 *>(&bp[o + 64]) = o2;
                        *reinterpret_cast<sycl::uint4 *>(&bp[o + 96]) = o3;
                    }
                    {   // prefetch the next part (clamped, unconditional: no phi copies)
                        const int nb = part + 1 < 2 ? b : (b + 1 < b1 ? b + 1 : b);
                        const int np = part + 1 < 2 ? part + 1 : (b + 1 < b1 ? 0 : part);
#pragma unroll
                        for (int u = 0; u < 2; ++u) {
                            va[u] = ldq(gl + u * rl + nb * 128 + np * 64);
                            vb[u] = ldq(gl + u * rl + nb * 128 + np * 64 + 32);
                            vh[u] = ldq(gh + u * rh + nb * 64 + np * 32);
                        }
                    }
                    sycl::group_barrier(sg);
#pragma unroll
                    for (int gg = 0; gg < 4; ++gg) {
                        const int g = part * 4 + gg;
                        mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, X6_TK, X6_TN, mx::layout::col_major> B;
                        mx::joint_matrix_load(sg, B, bp + gg * X6_TK, X6_PB);
#pragma unroll
                        for (int t = 0; t < NT; ++t) {
                            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, X6_TM, X6_TK, mx::layout::row_major> Alo, Ahi;
                            const int8_t * xg = xa + (size_t) t * X6_TM * 2 * K + (size_t) (b * 8 + g) * 64;
                            mx::joint_matrix_load(sg, Alo, sycl::address_space_cast<address_space::global_space, decorated::no>(xg), 2 * K);
                            mx::joint_matrix_load(sg, Ahi, sycl::address_space_cast<address_space::global_space, decorated::no>(xg + 32), 2 * K);
                            mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, X6_TM, X6_TN> Clo, Chi;
                            mx::joint_matrix_fill(sg, Clo, 0);
                            mx::joint_matrix_fill(sg, Chi, 0);
                            mx::joint_matrix_mad(sg, Clo, Alo, B, Clo);
                            mx::joint_matrix_mad(sg, Chi, Ahi, B, Chi);
                            // column-side d8 broadcast (register regions); these sub-group ops must stay outside the
                            // joint_matrix_apply lambdas
                            float db[8];
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                if constexpr (MC == 8) {
                                    db[i] = sycl::select_from_group(sg, d8c[i & 3], 2 * g + (i >> 2));
                                } else {
                                    db[i] = sycl::select_from_group(sg, d8c[i], 2 * g + t);
                                }
                            }
                            float lo[8];
                            {
                                int i = 0;
                                imx::joint_matrix_apply(sg, Clo, [&](int32_t & v, size_t, size_t) {
                                    if (i < 8) { lo[i] = (float) v * scf[2 * g]; }
                                    ++i;
                                });
                            }
                            {
                                int i = 0;
                                imx::joint_matrix_apply(sg, Chi, [&](int32_t & v, size_t, size_t) {
                                    if (i < 8) { Fb[t * 8 + i] = sycl::fma(sycl::fma((float) v, scf[2 * g + 1], lo[i]), db[i], Fb[t * 8 + i]); }
                                    ++i;
                                });
                            }
                        }
                    }
                    sycl::group_barrier(sg);
                }
#pragma unroll
                for (int i = 0; i < MC; ++i) { F[i] = sycl::fma(dq, Fb[i], F[i]); }
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

// GGML_SYCL_XMX_Q6K_CHECKQ=1: compare the fused quantizer's operands (columns < M) with the legacy quantize + pack
void xmx_q6k_checkq(const float * x, const int8_t * xa, const sycl::half * d8, int K, int M,
                    ggml_backend_sycl_context & ctx, sycl::queue & q) {
    const int     G   = K / QK8_1;
    const int64_t Kp  = GGML_PAD(K, MATRIX_ROW_PADDING);
    const size_t  sty = (size_t) Kp * sizeof(block_q8_1) / QK8_1;
    ggml_sycl_pool_alloc<char> soa(ctx.pool(), sty * M);
    ggml_sycl_pool_alloc<char> ref(ctx.pool(), (size_t) X6_TM * 2 * K + (size_t) G * X6_TM * sizeof(sycl::half));
    int8_t *     rxa  = reinterpret_cast<int8_t *>(ref.get());
    sycl::half * rd8  = reinterpret_cast<sycl::half *>(ref.get() + (size_t) X6_TM * 2 * K);
    char *       psoa = soa.get();
    const size_t nblk = (size_t) M * G;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nblk * WARP_SIZE), sycl::range<1>(WARP_SIZE)),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        quantize_and_reorder_q8_1_soa<QK8_1 / WARP_SIZE>()(x, psoa, K, (int) Kp, it);
    });
    const queue_ptr qp = &q;
    xmx_q6k_pack_act(psoa, sty, rxa, rd8, K, M, qp);
    std::vector<int8_t>   a((size_t) M * 2 * K), ra((size_t) M * 2 * K);
    std::vector<uint16_t> b((size_t) G * X6_TM), rb((size_t) G * X6_TM);
    q.memcpy(a.data(), xa, a.size()).wait();
    q.memcpy(ra.data(), rxa, ra.size()).wait();
    q.memcpy(b.data(), d8, b.size() * 2).wait();
    q.memcpy(rb.data(), rd8, rb.size() * 2).wait();
    int na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != ra[i] && na++ < 3) { fprintf(stderr, "  q6 xa[%zu] %d vs %d\n", i, a[i], ra[i]); }
    }
    for (int g = 0; g < G; ++g) {
        for (int c = 0; c < M; ++c) {
            const size_t i = (size_t) g * X6_TM + c;
            if (b[i] != rb[i] && nb++ < 3) { fprintf(stderr, "  q6 d8[%zu] %04x vs %04x\n", i, b[i], rb[i]); }
        }
    }
    const bool ok = na == 0 && nb == 0;
    static std::atomic<int> n_ok{ 0 }, n_bad{ 0 };
    (ok ? n_ok : n_bad)++;
    if (!ok || (n_ok + n_bad) % 50 == 1) {
        fprintf(stderr, "XMXCHECKQ6 K=%d M=%d %s (ok %d, mismatched %d; xa %d d8 %d)\n", K, M,
                ok ? "IDENTICAL" : "MISMATCH", (int) n_ok, (int) n_bad, na, nb);
    }
}

} // namespace

int ggml_sycl_xmx_q6k_path() {
    // 0 (default): fused quantize + matmul, launched directly (2 kernels, both in the 256-register mode)
    // 1: legacy, through ggml_sycl_op_mul_mat (q8_1 SoA quantize, pack, matmul: 3 kernels)
    static const int v = ggml_sycl_xmx_q6k_env_int("GGML_SYCL_XMX_Q6K_PATH", 0);
    return v;
}

void ggml_sycl_mul_mat_xmx_q6k(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst) try {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2, " : XMX q6_K");

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const int64_t G = K / QK8_1;
    GGML_ASSERT(src1->ne[0] == K && dst->ne[0] == N && dst->ne[1] == M);
    GGML_ASSERT(K % QK_K == 0 && N % XMX_Q6K_ROWS == 0);
    GGML_ASSERT(M >= 1 && M <= XMX_Q6K_MAX_COLS);
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    GGML_ASSERT(extra && extra->optimized_feature.reorder);

    const xmx_q6k_weights w = xmx_q6k_weights_of(static_cast<const char *>(src0->data), N, K);

    if (M > X6_MC) {
        // LOCAL (longdraft): 9..16 columns, two A tiles: xa [16][2K], d8 [G][16]
        const size_t xa16 = (size_t) 2 * X6_TM * 2 * K;
        const size_t d816 = (size_t) G * 2 * X6_TM * sizeof(sycl::half);
        ggml_sycl_pool_alloc<char> act16(ctx.pool(), xa16 + d816);
        int8_t *     xa = reinterpret_cast<int8_t *>(act16.get());
        sycl::half * d8 = reinterpret_cast<sycl::half *>(act16.get() + xa16);
        sycl::queue & q = *ctx.stream();
        static const int ks16 = ggml_sycl_xmx_q6k_env_int("GGML_SYCL_XMX_Q6K_KS16", 0);
        // LOCAL (wideverify): the register-fed two-tile kernel (GGML_SYCL_XMX_Q6K_DIRECT=1 + GGML_SYCL_XMX_WIDE=1),
        // activations in the group-major [G][2][16][32] layout; same per-column math and order as the joint_matrix
        // two-tile kernel, bit-identical to it at the same ks. Split-K: the 1..8 register-fed kernel's rule. Not for
        // ffn_down-like shapes (>= 64 blocks per row, <= 8192 rows), where the joint_matrix kernel is faster at 9..16
        // columns (B70, test-backend-ops perf, 5120x17408, us at 9 / 16 columns: joint_matrix ks 3 ~165; register-fed
        // ks 16 187 / 190, ks 10 230 / 235, ks 3 244 / 220); GGML_SYCL_XMX_Q6K_DIRECT16_ALL=1 takes them too.
        static const bool d16_all = ggml_sycl_xmx_q6k_env_int("GGML_SYCL_XMX_Q6K_DIRECT16_ALL", 0) != 0;
        const bool direct16 = ggml_sycl_xmx_q6k_direct_env() && ggml_sycl_xmx_direct_wide() &&
                              (d16_all || !((K / QK_K) >= 64 && N <= 8192));
        const float * x = static_cast<const float *>(src1->data);
        xmx_q6k_quant_act<2 * X6_TM>(x, xa, d8, (int) K, (int) M, q, direct16);
        int ks = xmx_q6k_pick_ks((int) N, (int) K);
        if (direct16) {
            static const int ks_env = ggml_sycl_xmx_q6k_env_int("GGML_SYCL_XMX_Q6K_DIRECT_KS", -1);
            if (ks_env < 0 && (K / QK_K) >= 64 && N <= 8192) {
                ks = 16;
            } else if (ks_env > 0) {
                ks = std::min(ks_env, (int) (K / QK_K));
            }
        }
        if (ks16 > 0) {
            ks = std::max(1, std::min(ks16, std::min(16, (int) (K / QK_K))));
        }
        if (direct16) {
            ggml_sycl_xmx_q6k_direct16_launch(w.ql, w.qh, w.sc, w.dd, xa, d8, static_cast<float *>(dst->data), (int) N,
                                              (int) K, (int) M, dst->ne[0], ks, q);
            static const bool dcheck = ggml_sycl_xmx_q6k_env_int("GGML_SYCL_XMX_Q6K_DIRECT_CHECK", 0) != 0;
            if (dcheck) {   // measurement only (synchronous): the joint_matrix two-tile kernel at the same ks, bit for bit
                ggml_sycl_pool_alloc<char>  ref_act(ctx.pool(), xa16 + d816);
                ggml_sycl_pool_alloc<float> ref_out(ctx.pool(), (size_t) M * N);
                int8_t *     rxa = reinterpret_cast<int8_t *>(ref_act.get());
                sycl::half * rd8 = reinterpret_cast<sycl::half *>(ref_act.get() + xa16);
                xmx_q6k_quant_act<2 * X6_TM>(x, rxa, rd8, (int) K, (int) M, q, false);
                xmx_q6k_launch_wide<2 * X6_TM>(w.ql, w.qh, w.sc, w.dd, rxa, rd8, ref_out.get(), (int) N, (int) K, (int) M,
                                               N, ks, q);
                q.wait();
                std::vector<float> a((size_t) M * N), r((size_t) M * N);
                const float * out = static_cast<const float *>(dst->data);
                for (int64_t i = 0; i < M; ++i) {
                    q.memcpy(a.data() + i * N, out + i * dst->ne[0], N * sizeof(float));
                }
                q.memcpy(r.data(), ref_out.get(), r.size() * sizeof(float)).wait();
                size_t nd = 0;
                for (size_t i = 0; i < a.size(); ++i) { nd += memcmp(&a[i], &r[i], sizeof(float)) != 0; }
                fprintf(stderr, "XMXDIRECTCHECK q6_K N=%lld K=%lld M=%lld ks=%d: %s (%zu of %zu differ)\n", (long long) N,
                        (long long) K, (long long) M, ks, nd == 0 ? "IDENTICAL" : "MISMATCH", nd, a.size());
            }
            return;
        }
        xmx_q6k_launch_wide<2 * X6_TM>(w.ql, w.qh, w.sc, w.dd, xa, d8, static_cast<float *>(dst->data), (int) N,
                                       (int) K, (int) M, dst->ne[0], ks, q);
        return;
    }

    const size_t xa_bytes = (size_t) X6_TM * 2 * K;
    const size_t d8_bytes = (size_t) G * X6_TM * sizeof(sycl::half);
    ggml_sycl_pool_alloc<char> act_alloc(ctx.pool(), xa_bytes + d8_bytes);
    int8_t *     xa = reinterpret_cast<int8_t *>(act_alloc.get());
    sycl::half * d8 = reinterpret_cast<sycl::half *>(act_alloc.get() + xa_bytes);

    sycl::queue & q = *ctx.stream();
    const float * x = static_cast<const float *>(src1->data);
    const bool direct = ggml_sycl_xmx_q6k_direct_env();
    xmx_q6k_quant_act(x, xa, d8, (int) K, (int) M, q, direct);
    static const bool checkq = ggml_sycl_xmx_q6k_env_int("GGML_SYCL_XMX_Q6K_CHECKQ", 0) != 0;
    if (checkq && !direct) {
        xmx_q6k_checkq(x, xa, d8, (int) K, (int) M, ctx, q);
    }
    if (direct) {
        // split-K: the joint_matrix kernel's, except ffn_down 5120x17408 (68 blocks per row) where the register-fed
        // kernel is fastest at 16 (B70 probe, us at 8 columns: ks 3 156.7, 10 147.2, 12 146.2, 16 142.9; the
        // joint_matrix kernel: 3 155.7, 10 158.5, 16 159.8). Same ks = bit-identical to the joint_matrix kernel.
        static const int ks_env = ggml_sycl_xmx_q6k_env_int("GGML_SYCL_XMX_Q6K_DIRECT_KS", -1);
        int ks = xmx_q6k_pick_ks((int) N, (int) K);
        if (ks_env < 0 && (K / QK_K) >= 64 && N <= 8192) {
            ks = 16;
        } else if (ks_env > 0) {
            ks = std::min(ks_env, (int) (K / QK_K));
        }
        ggml_sycl_xmx_q6k_direct_launch(w.ql, w.qh, w.sc, w.dd, xa, d8, static_cast<float *>(dst->data), (int) N,
                                        (int) K, (int) M, dst->ne[0], ks, q);
        // GGML_SYCL_XMX_Q6K_DIRECT_CHECK=1 (measurement only, synchronous): rerun the op on the joint_matrix kernel
        // (same split-K) and compare every output bit for bit
        static const bool dcheck = ggml_sycl_xmx_q6k_env_int("GGML_SYCL_XMX_Q6K_DIRECT_CHECK", 0) != 0;
        if (dcheck) {
            ggml_sycl_pool_alloc<char>  ref_act(ctx.pool(), xa_bytes + d8_bytes);
            ggml_sycl_pool_alloc<float> ref_out(ctx.pool(), (size_t) M * N);
            int8_t *     rxa = reinterpret_cast<int8_t *>(ref_act.get());
            sycl::half * rd8 = reinterpret_cast<sycl::half *>(ref_act.get() + xa_bytes);
            xmx_q6k_quant_act(x, rxa, rd8, (int) K, (int) M, q, false);
            xmx_q6k_launch(w.ql, w.qh, w.sc, w.dd, rxa, rd8, ref_out.get(), (int) N, (int) K, (int) M, N, ks, q);
            q.wait();
            std::vector<float> a((size_t) M * N), r((size_t) M * N);
            const float * out = static_cast<const float *>(dst->data);
            for (int64_t i = 0; i < M; ++i) {
                q.memcpy(a.data() + i * N, out + i * dst->ne[0], N * sizeof(float));
            }
            q.memcpy(r.data(), ref_out.get(), r.size() * sizeof(float)).wait();
            size_t nd = 0;
            for (size_t i = 0; i < a.size(); ++i) { nd += memcmp(&a[i], &r[i], sizeof(float)) != 0; }
            static std::atomic<int> n_ok{ 0 }, n_bad{ 0 };
            (nd == 0 ? n_ok : n_bad)++;
            fprintf(stderr, "XMXDIRECTCHECK q6_K N=%lld K=%lld M=%lld ks=%d: %s (%zu of %zu differ; ok %d, mismatched %d)\n",
                    (long long) N, (long long) K, (long long) M, ks, nd == 0 ? "IDENTICAL" : "MISMATCH", nd, a.size(),
                    (int) n_ok, (int) n_bad);
        }
    } else {
        xmx_q6k_launch(w.ql, w.qh, w.sc, w.dd, xa, d8, static_cast<float *>(dst->data), (int) N, (int) K, (int) M,
                       dst->ne[0], xmx_q6k_pick_ks((int) N, (int) K), q);
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    GGML_SYCL_EXIT_OR_RETHROW();
}

void ggml_sycl_op_mul_mat_xmx_q6k(ggml_backend_sycl_context & ctx,const ggml_tensor * src0, const ggml_tensor * src1,
                                  ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                  const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low,
                                  const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_row_size,
                                  const queue_ptr & stream) try {
    GGML_UNUSED(src1_ddf_i);
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2, " : XMX q6_K (legacy driver path)");

    const int64_t K  = src0->ne[0];
    const int64_t N  = src0->ne[1];
    const int64_t M  = src1_ncols;
    const int64_t G  = K / QK8_1;
    GGML_ASSERT(row_low == 0 && row_high == N);   // no split: the whole weight, dst column stride dst->ne[0]
    GGML_ASSERT(src1->ne[0] == K && dst->ne[0] == N);
    GGML_ASSERT(K % QK_K == 0 && N % XMX_Q6K_ROWS == 0);
    GGML_ASSERT(M >= 1 && M <= X6_MC);
    GGML_ASSERT(src1_ddq_i != nullptr);
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    GGML_ASSERT(extra && extra->optimized_feature.reorder);

    const xmx_q6k_weights w = xmx_q6k_weights_of(src0_dd_i, N, K);

    // activations: the driver's SoA q8_1 (quantize_and_reorder_q8_1_soa), column stride as MMVQ's stride_col_y_bytes
    const size_t stride_y = (size_t) src1_padded_row_size * sizeof(block_q8_1) / QK8_1;

    const size_t xa_bytes = (size_t) X6_TM * 2 * K;
    const size_t d8_bytes = (size_t) G * X6_TM * sizeof(sycl::half);
    ggml_sycl_pool_alloc<char> act_alloc(ctx.pool(), xa_bytes + d8_bytes);
    int8_t *     xa = reinterpret_cast<int8_t *>(act_alloc.get());
    sycl::half * d8 = reinterpret_cast<sycl::half *>(act_alloc.get() + xa_bytes);

    xmx_q6k_pack_act(src1_ddq_i, stride_y, xa, d8, (int) K, (int) M, stream);
    xmx_q6k_launch(w.ql, w.qh, w.sc, w.dd, xa, d8, dst_dd_i, (int) N, (int) K, (int) M, dst->ne[0],
                   xmx_q6k_pick_ks((int) N, (int) K), *stream);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    GGML_SYCL_EXIT_OR_RETHROW();
}

#else  // !GGML_SYCL_XMX

int ggml_sycl_xmx_q6k_path() { return 0; }

void ggml_sycl_mul_mat_xmx_q6k(ggml_backend_sycl_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {
    GGML_ABORT("XMX q6_K path not built (configure with -DGGML_SYCL_XMX=ON)");
}

void ggml_sycl_op_mul_mat_xmx_q6k(ggml_backend_sycl_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *,
                                  const char *, const float *, const char *, float *, const int64_t, const int64_t,
                                  const int64_t, const int64_t, const queue_ptr &) {
    GGML_ABORT("XMX q6_K path not built (configure with -DGGML_SYCL_XMX=ON)");
}

#endif
