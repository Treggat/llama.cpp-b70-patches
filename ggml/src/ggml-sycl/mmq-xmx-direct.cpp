//
// "Direct" XMX q4_K x q8_1 matmul (GGML_SYCL_XMX_Q4K_DIRECT=1, default off) - see mmq-xmx-q4k.hpp
//
// Same math, same split-K partition and the same float operation order as the joint_matrix kernel in
// mmq-xmx-q4k.cpp (VAR 12), so the outputs are bit-identical to it; what changes is how the weights reach the matrix
// unit. The joint_matrix kernel stages every 16-row x 32-k weight tile through SLM (nibbles -> s8 in registers, two
// 16 B SLM stores per lane, a col_major joint_matrix_load that IGC turns into 3 SLM gathers + a divergent branch,
// sub-group barriers) because a joint_matrix B operand can only be loaded from memory. Here each lane loads its own
// weight row (lane = row, as in the accumulator layout) and the s8 B operand is built in registers and handed to the
// DPAS directly through the documented OpenCL extension builtin intel_sub_group_i8_i8_matrix_mad_k32
// (cl_intel_subgroup_matrix_multiply_accumulate, the entry point Intel's sycl-tla uses for Xe DPAS). No SLM staging,
// no SLM bank conflicts, ~25% fewer instructions per 256-block.
//
// Measured on the B70 (standalone probe, random weights, cold L2, us at 8 / 3 columns): 17408x5120 98.2 -> 92.8
// (91.2 at 3), 10240x5120 64.2 -> 57.7, 6144x5120 39.8 -> 34.2, 5120x6144 39.3 -> 35.3, 12288x5120 74.2 -> 68.5,
// 1024x5120 12.3 -> 9.4; 0 of M*N outputs differ bitwise from the joint_matrix kernel on every one of these shapes.
//
// Built as its own shared library, see mmq-xmx-direct.hpp.
//
#include "mmq-xmx-direct.hpp"

#include <utility>

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

static constexpr int QK_K  = 256;
static constexpr int QK8_1 = 32;

namespace {

typedef short xd_v8s16 __attribute__((ext_vector_type(8)));
typedef int   xd_v8i32 __attribute__((ext_vector_type(8)));

}   // namespace

#ifdef __SYCL_DEVICE_ONLY__
// cl_intel_subgroup_matrix_multiply_accumulate (sub-group size 16): C[8x16 s32] = A[8x32 s8] * B[32x16 s8].
// Lane l holds A as short8 (row m, k = 2l..2l+1), B as int8 (column l, dword j = k 4j..4j+3) and C as int8 (column l,
// element m = row m).
SYCL_EXTERNAL xd_v8i32 intel_sub_group_i8_i8_matrix_mad_k32(xd_v8s16 a, xd_v8i32 b, xd_v8i32 acc);
#else
static inline xd_v8i32 intel_sub_group_i8_i8_matrix_mad_k32(xd_v8s16, xd_v8i32 b, xd_v8i32) { return b; }
#endif

namespace {

template <typename F, int... GI> inline void xd_unroll(std::integer_sequence<int, GI...>, F && f) {
    (f(std::integral_constant<int, GI>{}), ...);
}

}   // namespace

// qs [N][K/2], sc [N*KB][12], dm [N*KB] half2 (d, dmin): the q4_K reorder layout (read in place)
// xa: activation quants group-major, [G][16 columns][32] int8 (column c of group g at (g*16 + c)*32); d8 [16][G] half;
// us [16][G] int32 (integer group sums); out: column i of row r at out[i*ldd + r], i < M <= 8
sycl::event ggml_sycl_xmx_q4k_direct_launch(const uint8_t * dQs, const uint8_t * dSc, const uint32_t * dDm,
                                            const int8_t * dXa, const sycl::half * dD8c, const int32_t * dS8c,
                                            float * dOut, int N, int K, int M, int64_t ldd, int ks, sycl::queue & q) {
    const int KB = K / QK_K, G = K / QK8_1;
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(ks > 1 ? ks * 8 * 16 : 1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (N / 16) * ks * 16), sycl::range<1>(ks * 16)),
                       sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            using sycl::access::address_space;
            using sycl::access::decorated;
            sycl::sub_group sg = it.get_sub_group();
            const int lane = sg.get_local_id()[0];
            const int s    = sg.get_group_id()[0];
            const int row0 = (int) it.get_group(0) * 16;
            const int b0   = s * KB / ks, b1 = (s + 1) * KB / ks;
            const int row  = row0 + lane;
            const uint8_t *  qrow = dQs + (size_t) row * (K / 2);
            const uint32_t * hsc  = reinterpret_cast<const uint32_t *>(dSc + (size_t) row * KB * 12);
            const uint32_t * hdm  = dDm + (size_t) row * KB;
            const sycl::uint4 * d8base = reinterpret_cast<const sycl::uint4 *>(dD8c + (size_t) lane * G);
            const sycl::int4 *  s8base = reinterpret_cast<const sycl::int4 *>(dS8c + (size_t) lane * G);
            const uint16_t *    xa16   = reinterpret_cast<const uint16_t *>(dXa);

            // this lane's row, one 256-block = 128 B of nibbles (8 x 16 B), next block prefetched into registers
            sycl::uint4 w[8], wn[8];
            {
                const sycl::uint4 * pp = reinterpret_cast<const sycl::uint4 *>(qrow + (size_t) b0 * 128);
#pragma unroll
                for (int j = 0; j < 8; ++j) { w[j] = pp[j]; }
            }
            uint32_t hw0 = hsc[b0 * 3 + 0], hw1 = hsc[b0 * 3 + 1], hw2 = hsc[b0 * 3 + 2], hwd = hdm[b0];
            float F[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) { F[i] = 0.f; }
            for (int b = b0; b < b1; ++b) {
                // row scales: rsd = d * sc / 16 (exact), rw = 8 * d * sc - dmin * mn (the rank-1 term), as VAR 12
                float rsd[8], rw[8];
                {
                    const uint32_t sc03 = hw0 & 0x3F3F3F3Fu, mn03 = hw1 & 0x3F3F3F3Fu;
                    const uint32_t sc47 = (hw2 & 0x0F0F0F0Fu) | ((hw0 >> 2) & 0x30303030u);
                    const uint32_t mn47 = ((hw2 >> 4) & 0x0F0F0F0Fu) | ((hw1 >> 2) & 0x30303030u);
                    const float d    = (float) sycl::bit_cast<sycl::half>((uint16_t) (hwd & 0xFFFF));
                    const float dmin = (float) sycl::bit_cast<sycl::half>((uint16_t) (hwd >> 16));
                    const float d16  = d * 0.0625f, d8x = 8.f * d;
#pragma unroll
                    for (int g = 0; g < 8; ++g) {
                        const float sc = (float) (uint8_t) ((g < 4 ? sc03 : sc47) >> (8 * (g & 3)));
                        const float mn = (float) (uint8_t) ((g < 4 ? mn03 : mn47) >> (8 * (g & 3)));
                        rw[g]  = d8x * sc - dmin * mn;
                        rsd[g] = d16 * sc;
                    }
                }
                // column `lane`'s d8 and d8 * sum(u) for the 8 groups of this block
                float d8c[8], tc[8];
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
                {   // next block's header and nibbles (clamped, unconditional)
                    const int bn = b + 1 < b1 ? b + 1 : b;
                    hw0 = hsc[bn * 3 + 0]; hw1 = hsc[bn * 3 + 1]; hw2 = hsc[bn * 3 + 2]; hwd = hdm[bn];
                    const sycl::uint4 * pp = reinterpret_cast<const sycl::uint4 *>(qrow + (size_t) bn * 128);
#pragma unroll
                    for (int j = 0; j < 8; ++j) { wn[j] = pp[j]; }
                }
                // A tiles of the block's 8 groups (columns 0..7, 256 B each): one sub-group block read per group
                xd_v8s16 A[8];
#pragma unroll
                for (int g = 0; g < 8; ++g) {
                    const auto av = sg.load<8>(sycl::address_space_cast<address_space::global_space, decorated::yes>(
                        xa16 + (size_t) (b * 8 + g) * 256));
#pragma unroll
                    for (int m = 0; m < 8; ++m) { A[g][m] = (short) av[m]; }
                }
                float Fb[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) { Fb[i] = 0.f; }
                xd_unroll(std::make_integer_sequence<int, 8>{}, [&](auto gi) {
                    constexpr int g  = decltype(gi)::value;
                    constexpr int ch = g >> 1;   // q4_K: group 2ch = low nibbles, 2ch+1 = high nibbles of qs[ch*32 ..]
                    const sycl::uint4 w0 = w[2 * ch], w1 = w[2 * ch + 1];
                    // (q - 8) * 16 as s8: nibble moved to the high half of the byte, top bit flipped (the x16 is folded
                    // exactly into rsd)
                    auto cv = [](uint32_t x) -> int {
                        return (int) ((g & 1) ? ((x & 0xF0F0F0F0u) ^ 0x80808080u) : (((x << 4) & 0xF0F0F0F0u) ^ 0x80808080u));
                    };
                    const xd_v8i32 B = { cv(w0.x()), cv(w0.y()), cv(w0.z()), cv(w0.w()),
                                         cv(w1.x()), cv(w1.y()), cv(w1.z()), cv(w1.w()) };
                    const xd_v8i32 z   = 0;
                    const xd_v8i32 acc = intel_sub_group_i8_i8_matrix_mad_k32(A[g], B, z);   // dot16 per (column, row)
                    float sbv[8], tbv[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        sbv[i] = rsd[g] * sycl::select_from_group(sg, d8c[g], i);
                        tbv[i] = sycl::select_from_group(sg, tc[g], i);
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        Fb[i] = sycl::fma((float) acc[i], sbv[i], Fb[i]);
                        Fb[i] = sycl::fma(tbv[i], rw[g], Fb[i]);
                    }
                });
#pragma unroll
                for (int i = 0; i < 8; ++i) { F[i] += Fb[i]; }
#pragma unroll
                for (int j = 0; j < 8; ++j) { w[j] = wn[j]; }
            }
            if (ks > 1) {   // split-K reduction in SLM, fixed order (the joint_matrix kernel's)
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
                for (int i = 0; i < 8; ++i) { if (i < M) { dOut[(size_t) i * ldd + row] = F[i]; } }
            }
        });
    });
}

// q6_K: ql [N*KB][128], qh [N*KB][64], sc [N*KB][16] int8, dd [N*KB] half (the q6_K reorder layout, read in place)
// xa: [G][2][8][32] int8 (tile 0: elements 0..15 of the group then zeros, tile 1: zeros then elements 16..31);
// d8 [G][8] half. Same math and order as xmx_q6k_launch (mmq-xmx-q6k.cpp): per 32-group two DPAS on one B (the
// 16-element halves carry different 6-bit scales), Fb += fma(fma(hi, sc[2g+1], lo * sc[2g]), d8, Fb), F += (d/4) Fb.
// The 4 (q - 32) byte of element 32t + l of half h: stage<t>(ql dword of ql[64h + 32(t&1) + l], qh dword of
// qh[32h + l]) - the joint_matrix kernel's SLM staging, done in registers.
template <int T> static inline uint32_t xd_q6k_stage(uint32_t ql, uint32_t qh) {
    const uint32_t lo = T < 2 ? (ql << 2) & 0x3C3C3C3Cu : (ql >> 2) & 0x3C3C3C3Cu;
    const uint32_t hi = (qh << (6 - 2 * T)) & 0xC0C0C0C0u;
    return (lo | hi) ^ 0x80808080u;
}

sycl::event ggml_sycl_xmx_q6k_direct_launch(const uint8_t * dQl, const uint8_t * dQh, const int8_t * dSc,
                                            const uint16_t * dD, const int8_t * dXa, const sycl::half * dD8,
                                            float * dOut, int N, int K, int M, int64_t ldd, int ks, sycl::queue & q) {
    const int KB = K / QK_K;
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(ks > 1 ? ks * 8 * 16 : 1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (N / 16) * ks * 16), sycl::range<1>(ks * 16)),
                       sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            using sycl::access::address_space;
            using sycl::access::decorated;
            sycl::sub_group sg = it.get_sub_group();
            const int lane = sg.get_local_id()[0];
            const int s    = sg.get_group_id()[0];
            const int row0 = (int) it.get_group(0) * 16;
            const int b0   = s * KB / ks, b1 = (s + 1) * KB / ks;
            const int row  = row0 + lane;
            const sycl::uint4 * gl   = reinterpret_cast<const sycl::uint4 *>(dQl + (size_t) row * KB * 128);
            const sycl::uint4 * gh   = reinterpret_cast<const sycl::uint4 *>(dQh + (size_t) row * KB * 64);
            const sycl::uint4 * hsc  = reinterpret_cast<const sycl::uint4 *>(dSc + (size_t) row * KB * 16);
            const uint16_t *    hd   = dD + (size_t) row * KB;
            const uint16_t *    xa16 = reinterpret_cast<const uint16_t *>(dXa);
            // d8 [G][8] half: lane l's uint2 of block b = group l/2, columns 4(l%2)..+3 (as the joint_matrix kernel)
            const sycl::uint2 * d8base = reinterpret_cast<const sycl::uint2 *>(dD8) + lane;
            auto d8f = [](uint32_t w, int hi) -> float {
                return (float) sycl::bit_cast<sycl::half>((uint16_t) (hi ? (w >> 16) : (w & 0xFFFF)));
            };

            sycl::uint4 wl[8], wh[4], wln[8], whn[4];   // this lane's row: ql / qh of the block, next block prefetched
#pragma unroll
            for (int j = 0; j < 8; ++j) { wl[j] = gl[(size_t) b0 * 8 + j]; }
#pragma unroll
            for (int j = 0; j < 4; ++j) { wh[j] = gh[(size_t) b0 * 4 + j]; }
            sycl::uint4 hs  = hsc[b0];
            uint16_t    hdd = hd[b0];
            sycl::uint2 d8n = d8base[(size_t) b0 * 16];
            float F[8];
#pragma unroll
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
                {   // next block (clamped, unconditional)
                    const int bn = b + 1 < b1 ? b + 1 : b;
                    hs  = hsc[bn];
                    hdd = hd[bn];
                    d8n = d8base[(size_t) bn * 16];
#pragma unroll
                    for (int j = 0; j < 8; ++j) { wln[j] = gl[(size_t) bn * 8 + j]; }
#pragma unroll
                    for (int j = 0; j < 4; ++j) { whn[j] = gh[(size_t) bn * 4 + j]; }
                }
                float Fb[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) { Fb[i] = 0.f; }
                xd_unroll(std::make_integer_sequence<int, 8>{}, [&](auto gi) {
                    constexpr int g = decltype(gi)::value;
                    constexpr int hh = g >> 2, t = g & 3;
                    const sycl::uint4 l0 = wl[4 * hh + 2 * (t & 1)], l1 = wl[4 * hh + 2 * (t & 1) + 1];
                    const sycl::uint4 h0 = wh[2 * hh], h1 = wh[2 * hh + 1];
                    const xd_v8i32 B = { (int) xd_q6k_stage<t>(l0.x(), h0.x()), (int) xd_q6k_stage<t>(l0.y(), h0.y()),
                                         (int) xd_q6k_stage<t>(l0.z(), h0.z()), (int) xd_q6k_stage<t>(l0.w(), h0.w()),
                                         (int) xd_q6k_stage<t>(l1.x(), h1.x()), (int) xd_q6k_stage<t>(l1.y(), h1.y()),
                                         (int) xd_q6k_stage<t>(l1.z(), h1.z()), (int) xd_q6k_stage<t>(l1.w(), h1.w()) };
                    xd_v8s16 alo, ahi;
                    {
                        const auto av = sg.load<8>(sycl::address_space_cast<address_space::global_space, decorated::yes>(
                            xa16 + (size_t) (b * 8 + g) * 256));
#pragma unroll
                        for (int m = 0; m < 8; ++m) { alo[m] = (short) av[m]; }
                    }
                    {
                        const auto av = sg.load<8>(sycl::address_space_cast<address_space::global_space, decorated::yes>(
                            xa16 + (size_t) (b * 8 + g) * 256 + 128));
#pragma unroll
                        for (int m = 0; m < 8; ++m) { ahi[m] = (short) av[m]; }
                    }
                    const xd_v8i32 z   = 0;
                    const xd_v8i32 clo = intel_sub_group_i8_i8_matrix_mad_k32(alo, B, z);
                    const xd_v8i32 chi = intel_sub_group_i8_i8_matrix_mad_k32(ahi, B, z);
                    float db[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) { db[i] = sycl::select_from_group(sg, d8c[i & 3], 2 * g + (i >> 2)); }
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const float lo = (float) clo[i] * scf[2 * g];
                        Fb[i] = sycl::fma(sycl::fma((float) chi[i], scf[2 * g + 1], lo), db[i], Fb[i]);
                    }
                });
#pragma unroll
                for (int i = 0; i < 8; ++i) { F[i] = sycl::fma(dq, Fb[i], F[i]); }
#pragma unroll
                for (int j = 0; j < 8; ++j) { wl[j] = wln[j]; }
#pragma unroll
                for (int j = 0; j < 4; ++j) { wh[j] = whn[j]; }
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
                for (int i = 0; i < 8; ++i) { if (i < M) { dOut[(size_t) i * ldd + row] = F[i]; } }
            }
        });
    });
}
