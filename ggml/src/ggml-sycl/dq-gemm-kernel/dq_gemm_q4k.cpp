// dq_gemm_q4k.cpp: see dq_gemm_q4k.hpp for the interface.
//
// Design (defaults; tunables are the DQ_* macros below):
//   - Work-group tile WG_T = 256 tokens x WG_R = 128 weight rows, 16 sub-groups (8 along tokens x 2 along rows),
//     sub-group tile 32 tokens x 64 rows = NA 4 A tiles (8 tok x 16 k) x NB 4 B tiles (16 k x 16 rows)
//     -> 16 f32 accumulators (8x16) that stay in registers for the whole K loop (128 GRF, 256-GRF mode).
//   - Roles: A = activations (row_major, straight from global, 2D block loads), B = dequantized weights
//     (VNNI-packed f16 from SLM), C: lane = weight row, element i = token i; stored row_major into dst[n][m].
//   - K is walked in stages of KC = 128 (half a q4_K block = 4 sub-blocks). Per stage the whole work-group
//     dequantizes its 128 rows x 128 k into SLM (32 KB, double-buffered = 64 KB, ONE work-group barrier per stage).
//     Each work-item owns PB-byte pieces of nibbles (row, PB of the 32 bytes of the 64-chunk) = PB low-nibble
//     weights of sub-block 2c + PB high-nibble weights of sub-block 2c+1, plus its row's 12 scale bytes + d/dmin.
//   - Per stage (default DQ_ILV=0): issue the global loads of stage s+2 into registers, dequantize stage s+1 (held
//     in registers since the previous stage) into the other SLM buffer, then run the 4 DPAS k-steps of stage s
//     (A 2D block loads + B SLM loads + 16 DPAS each), then one barrier. With DQ_BW=1 the 16 lanes of a sub-group
//     are the 16 rows of a tile, so the dequantized VNNI dwords go out as sub-group block stores (2 per piece).
//     DQ_ILV=1 spreads the dequant over the k-steps instead (measured slower).
//   - Dequant = production numerics: w = (half) fma(d*sc, q, -(dmin*m)) in f32, rounded to f16 once.
//   - Ragged N: ONE launch; the last token tile (N % WG_T) takes a separate code path in the same kernel whose A
//     loads / C stores are bounds-checked only for the 8-token tiles that cross N.
#include "dq_gemm_q4k.hpp"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#ifndef DQ_NA
#define DQ_NA 4          // 8-token A tiles per sub-group
#endif
#ifndef DQ_NB
#define DQ_NB 4          // 16-row B tiles per sub-group
#endif
#ifndef DQ_WGT
#define DQ_WGT 8         // sub-groups along tokens
#endif
#ifndef DQ_WGR
#define DQ_WGR 2         // sub-groups along rows
#endif
#ifndef DQ_KC
#define DQ_KC 128        // k per SLM stage (multiple of 64)
#endif
#ifndef DQ_PB
#define DQ_PB 16         // nibble bytes per staging piece (4, 8 or 16)
#endif
#ifndef DQ_GRF
#define DQ_GRF 256
#endif
#ifndef DQ_ILV
#define DQ_ILV 0         // 1: interleave the dequant of the next stage with the DPAS steps; 0: dequant before the DPAS
#endif
#ifndef DQ_BW
#define DQ_BW 1          // 1: sub-group block stores of the dequantized VNNI dwords into SLM; 0: per-lane dword stores
#endif
#ifndef DQ_DBG
#define DQ_DBG 0         // timing only (wrong results): 1 = no dequant math (raw nibble words to SLM), 2 = no DPAS,
                         // 3 = no weight staging at all (DPAS on stale SLM), 4 = no A loads (A loaded once)
#endif

namespace dqg {
namespace mx  = sycl::ext::oneapi::experimental::matrix;
namespace imx = sycl::ext::intel::experimental::matrix;
using sycl::access::address_space;
using sycl::access::decorated;

#ifndef DQ_TM
#define DQ_TM 8          // tokens per A tile / accumulator (8, 16 or 32; the f16 combinations allow M <= 32)
#endif
constexpr int TM = DQ_TM, TN = 16, TK = 16, SG = 16;
constexpr int NA = DQ_NA, NB = DQ_NB, WGT = DQ_WGT, WGR = DQ_WGR, KC = DQ_KC, PB = DQ_PB;
constexpr int WG_T = WGT * NA * TM;          // tokens per work-group
constexpr int WG_R = WGR * NB * TN;          // weight rows per work-group
constexpr int NSG  = WGT * WGR;
constexpr int WGI  = NSG * SG;               // work-items per work-group
constexpr int CH   = KC / 64;                // 64-k chunks per stage
constexpr int NH   = 32 / PB;                // pieces per row per chunk
constexpr int NPC  = WG_R * NH * CH;         // pieces per stage
constexpr int PPI  = NPC / WGI;              // pieces per work-item
constexpr int PDW  = PB / 4;                 // dwords per piece
constexpr int NU   = PPI * PDW;              // dequant units (1 dword of nibbles = 8 weights) per work-item per stage
constexpr int KS   = KC / TK;                // DPAS k-steps per stage
constexpr int BUF_DW = WG_R * KC / 2;        // dwords per SLM buffer ([row/16][k/2][row%16] of half2)
constexpr int RT_DW  = KC / 2 * 16;          // dwords per 16-row tile in a buffer
static_assert(KC % 64 == 0, "KC must be a multiple of 64");
static_assert(PB == 4 || PB == 8 || PB == 16, "PB");
static_assert(NPC % WGI == 0 && PPI >= 1, "pieces must divide evenly over the work-items (choose PB)");

using Amat = mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TM, TK, mx::layout::row_major>;
using Bmat = mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::ext_intel_packed>;
using Cmat = mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN>;

class dq_kernel;

struct Piece {                               // global data of one piece for one stage
    uint32_t q[PDW];
    uint32_t s0, s1, s2, sd;
};

// one launch: token tiles 0..nfull-1 are full (unchecked code path), tile nfull (if ntt > nfull) is the ragged tail
static sycl::event launch(sycl::queue & q, const uint8_t * w, const sycl::half * act, float * dst,
                          int M, int N, int K, int ldd, int nfull, int ntt) {
    const int KB = K / 256;
    const size_t nb = (size_t) M * KB;
    const uint8_t *  wq  = w;
    const uint32_t * wsc = reinterpret_cast<const uint32_t *>(w + nb * 128);   // 3 dwords per block
    const uint32_t * wdm = reinterpret_cast<const uint32_t *>(w + nb * 140);   // 1 dword per block
    const int RT = M / WG_R;
    const size_t nwg = (size_t) ntt * RT;
    const int NS = K / KC;                   // stages
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(2 * BUF_DW), h);
        h.parallel_for<dq_kernel>(
            sycl::nd_range<1>(sycl::range<1>(nwg * WGI), sycl::range<1>(WGI)),
#if DQ_GRF > 0
            sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<DQ_GRF>},
#endif
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
          auto body = [&](auto tag) {
            constexpr bool CHECKED = decltype(tag)::value;
            sycl::sub_group sg = it.get_sub_group();
            const int s   = sg.get_group_id()[0];
            const int lid = it.get_local_id(0);
            const int wg  = it.get_group(0);
            const int tt  = wg % ntt, rt = wg / ntt;           // token tile fastest: neighbours share the weight tile in L2
            const int st  = s % WGT, sr = s / WGT;
            const int t0  = tt * WG_T + st * (NA * TM); // this sub-group's first token
            const int rw0 = rt * WG_R;                         // work-group's first row
            auto sp  = slm.get_multi_ptr<decorated::no>();
            auto sph = sycl::address_space_cast<address_space::local_space, decorated::no>(
                reinterpret_cast<sycl::half *>(&sp[0]));

            // ---- staging geometry: piece p = lid + i*WGI -> r16 = p&15, hh = (p>>4)%NH, rt16, c
            auto pgeo = [&](int i, int & r16, int & hh, int & rt16, int & c) {
                const int p = lid + i * WGI;
                r16 = p & 15; hh = (p >> 4) % NH;
                const int rest = (p >> 4) / NH;
                rt16 = rest % (WG_R / 16); c = rest / (WG_R / 16);
            };
            auto stage_load = [&](int stg, Piece (&P)[PPI]) {
#pragma unroll
                for (int i = 0; i < PPI; ++i) {
                    int r16, hh, rt16, c; pgeo(i, r16, hh, rt16, c);
                    const int row = rw0 + rt16 * 16 + r16;
                    const int k0  = stg * KC + c * 64;
                    const uint8_t * src = wq + (size_t) row * (K / 2) + (k0 >> 1) + hh * PB;
                    if constexpr (PB == 16) {
                        const sycl::uint4 v = *reinterpret_cast<const sycl::uint4 *>(src);
                        P[i].q[0] = v.x(); P[i].q[1] = v.y(); P[i].q[2] = v.z(); P[i].q[3] = v.w();
                    } else if constexpr (PB == 8) {
                        const sycl::uint2 v = *reinterpret_cast<const sycl::uint2 *>(src);
                        P[i].q[0] = v.x(); P[i].q[1] = v.y();
                    } else {
                        P[i].q[0] = *reinterpret_cast<const uint32_t *>(src);
                    }
                    const size_t bi = (size_t) row * KB + (k0 >> 8);
                    P[i].s0 = wsc[bi * 3 + 0]; P[i].s1 = wsc[bi * 3 + 1]; P[i].s2 = wsc[bi * 3 + 2]; P[i].sd = wdm[bi];
                }
            };
            // per-piece scale factors of a stage: da/ma (low nibbles, sub-block 2cb), db/mb (high nibbles, 2cb+1)
            float fda[PPI], fma_[PPI], fdb[PPI], fmb[PPI];
            auto stage_scales = [&](int stg, const Piece (&P)[PPI]) {
#pragma unroll
                for (int i = 0; i < PPI; ++i) {
                    int r16, hh, rt16, c; pgeo(i, r16, hh, rt16, c);
                    const int cb = ((stg * KC + c * 64) >> 6) & 3;           // 64-chunk within the q4_K block
                    const uint32_t s0 = P[i].s0, s1 = P[i].s1, s2 = P[i].s2;
                    uint32_t sca, mna, scb, mnb;
                    if (cb < 2) {   // sub-blocks 0..3: 6-bit fields in bytes 0..3 (scales) and 4..7 (mins)
                        sca = (s0 >> (16 * cb)) & 0x3F;     mna = (s1 >> (16 * cb)) & 0x3F;
                        scb = (s0 >> (16 * cb + 8)) & 0x3F; mnb = (s1 >> (16 * cb + 8)) & 0x3F;
                    } else {        // sub-blocks 4..7: low 4 bits from bytes 8..11, high 2 bits from the tops of bytes 0..7
                        const int sa = 16 * (cb - 2), sb = sa + 8;
                        sca = ((s2 >> sa) & 0x0F) | (((s0 >> (sa + 6)) & 0x3) << 4);
                        mna = ((s2 >> (sa + 4)) & 0x0F) | (((s1 >> (sa + 6)) & 0x3) << 4);
                        scb = ((s2 >> sb) & 0x0F) | (((s0 >> (sb + 6)) & 0x3) << 4);
                        mnb = ((s2 >> (sb + 4)) & 0x0F) | (((s1 >> (sb + 6)) & 0x3) << 4);
                    }
                    const float d    = (float) sycl::bit_cast<sycl::half>((uint16_t) (P[i].sd & 0xFFFF));
                    const float dmin = (float) sycl::bit_cast<sycl::half>((uint16_t) (P[i].sd >> 16));
                    fda[i] = d * (float) sca; fma_[i] = dmin * (float) mna;
                    fdb[i] = d * (float) scb; fmb[i]  = dmin * (float) mnb;
                }
            };
            // dequant units [u0, u0+NUN) of a stage (one piece): each unit = one dword of nibbles = 8 weights ->
            // 2 low-nibble + 2 high-nibble VNNI dwords. With DQ_BW the 16 lanes (= the 16 rows of a tile) write
            // them with sub-group block stores: dword j of lane l lands at base + j*16 + l, exactly [k/2][row%16].
            auto stage_units = [&](auto nun_tag, int u0, int buf, const Piece (&P)[PPI]) {
                constexpr int NUN = decltype(nun_tag)::value;
                const int i = u0 / PDW, dw0 = u0 % PDW;
                int r16, hh, rt16, c; pgeo(i, r16, hh, rt16, c);
                const int kp0 = (hh * PB + dw0 * 4) / 2;
                uint32_t lo[2 * NUN], hi[2 * NUN];
#pragma unroll
                for (int n = 0; n < NUN; ++n) {
                    const uint32_t x  = P[i].q[dw0 + n];
                    const uint32_t tl = x & 0x0F0F0F0Fu, th = (x >> 4) & 0x0F0F0F0Fu;   // 4 low / 4 high nibbles as bytes
#pragma unroll
                    for (int pp = 0; pp < 2; ++pp) {
#if DQ_DBG == 1
                        lo[2 * n + pp] = tl >> pp; hi[2 * n + pp] = th >> pp;
#else
                        const sycl::half l0 = (sycl::half) sycl::fma(fda[i], (float) (uint8_t) (tl >> (16 * pp)),     -fma_[i]);
                        const sycl::half l1 = (sycl::half) sycl::fma(fda[i], (float) (uint8_t) (tl >> (16 * pp + 8)), -fma_[i]);
                        const sycl::half h0 = (sycl::half) sycl::fma(fdb[i], (float) (uint8_t) (th >> (16 * pp)),     -fmb[i]);
                        const sycl::half h1 = (sycl::half) sycl::fma(fdb[i], (float) (uint8_t) (th >> (16 * pp + 8)), -fmb[i]);
                        lo[2 * n + pp] = (uint32_t) sycl::bit_cast<uint16_t>(l0) | ((uint32_t) sycl::bit_cast<uint16_t>(l1) << 16);
                        hi[2 * n + pp] = (uint32_t) sycl::bit_cast<uint16_t>(h0) | ((uint32_t) sycl::bit_cast<uint16_t>(h1) << 16);
#endif
                    }
                }
                const int base = buf * BUF_DW + rt16 * RT_DW + c * 32 * 16;
#if DQ_BW
                sycl::vec<uint32_t, 2 * NUN> vl, vh;
#pragma unroll
                for (int n = 0; n < 2 * NUN; ++n) { vl[n] = lo[n]; vh[n] = hi[n]; }
#ifdef __SYCL_DEVICE_ONLY__
                sycl::detail::sub_group::store<2 * NUN>(sp + base + kp0 * 16, vl);
                sycl::detail::sub_group::store<2 * NUN>(sp + base + (kp0 + 16) * 16, vh);
#else
                (void) vl; (void) vh; (void) base; (void) kp0;
#endif
#else
#pragma unroll
                for (int n = 0; n < 2 * NUN; ++n) {
                    sp[base + (kp0 + n) * 16 + r16]      = lo[n];
                    sp[base + (kp0 + 16 + n) * 16 + r16] = hi[n];
                }
#endif
            };
            using one_t = std::integral_constant<int, 1>;
            using piece_t = std::integral_constant<int, PDW>;

            Cmat C[NA][NB];
#pragma unroll
            for (int a = 0; a < NA; ++a)
#pragma unroll
                for (int j = 0; j < NB; ++j) { mx::joint_matrix_fill(sg, C[a][j], 0.f); }

            const bool sg_active = !CHECKED || t0 < N;          // uniform per sub-group
            auto actg = sycl::address_space_cast<address_space::global_space, decorated::no>(act);

            // operand loads (A from global, B from SLM buffer `buf`)
            auto load_A = [&](Amat (&A)[NA], int stg, int kk) {
                const int kg = stg * KC + kk * TK;
#pragma unroll
                for (int a = 0; a < NA; ++a) {
                    if (CHECKED && t0 + (a + 1) * TM > N) {   // only tiles that straddle / pass the end are checked
                        imx::joint_matrix_load_checked(sg, A[a], actg, K, N, K, t0 + a * TM, kg);
                    } else if (DQ_DBG == 4) {
                        mx::joint_matrix_load(sg, A[a], actg + (size_t) (t0 + a * TM) * K + kk * TK, K);
                    } else {
                        mx::joint_matrix_load(sg, A[a], actg + (size_t) (t0 + a * TM) * K + kg, K);
                    }
                }
            };
            auto bbase = [&](int buf) { return sph + buf * (BUF_DW * 2) + (sr * NB) * (RT_DW * 2); };

            Piece Pc[PPI], Pn[PPI];                               // current (stage s+1) and next (stage s+2) global data
            // prologue: stage 0 -> buffer 0, stage 1 in registers
            stage_load(0, Pc);
            stage_scales(0, Pc);
#pragma unroll
            for (int i = 0; i < PPI; ++i) { stage_units(piece_t{}, i * PDW, 0, Pc); }
            stage_load(NS > 1 ? 1 : 0, Pc);
            sycl::group_barrier(it.get_group());

            for (int stg = 0; stg < NS; ++stg) {
                // the last stage dequantizes a clamped duplicate into the idle buffer: no branch in the loop body
                const int sn = stg + 1 < NS ? stg + 1 : stg;
                if (DQ_DBG != 3) {
                    stage_load(stg + 2 < NS ? stg + 2 : NS - 1, Pn);  // clamped, unconditional
                    stage_scales(sn, Pc);
                    if (!DQ_ILV) {
#pragma unroll
                        for (int i = 0; i < PPI; ++i) { stage_units(piece_t{}, i * PDW, (stg + 1) & 1, Pc); }
                    }
                }
                const auto bb = bbase(stg & 1);
#pragma unroll
                for (int kk = 0; kk < KS; ++kk) {
                    if (sg_active) {
                        Amat A[NA]; Bmat B[NB];
                        load_A(A, stg, kk);
#pragma unroll
                        for (int j = 0; j < NB; ++j) { mx::joint_matrix_load(sg, B[j], bb + j * (RT_DW * 2) + kk * 256, 32); }
#pragma unroll
                        for (int a = 0; a < NA; ++a)
#pragma unroll
                            for (int j = 0; j < NB; ++j) {
                                if (DQ_DBG != 2 || (a == 0 && j == 0)) { mx::joint_matrix_mad(sg, C[a][j], A[a], B[j], C[a][j]); }
                            }
                    }
                    if (DQ_ILV && DQ_DBG != 3) {
                        // this step's share of the next stage's dequant units
#pragma unroll
                        for (int u = kk * NU / KS; u < (kk + 1) * NU / KS; ++u) { stage_units(one_t{}, u, (stg + 1) & 1, Pc); }
                    }
                }
                if (DQ_DBG != 3) {
#pragma unroll
                    for (int i = 0; i < PPI; ++i) { Pc[i] = Pn[i]; }
                }
                sycl::group_barrier(it.get_group());
            }

            if (sg_active) {
                auto dg = sycl::address_space_cast<address_space::global_space, decorated::no>(dst);
#pragma unroll
                for (int a = 0; a < NA; ++a)
#pragma unroll
                    for (int j = 0; j < NB; ++j) {
                        const int row = rw0 + (sr * NB + j) * TN;
                        if (CHECKED && t0 + (a + 1) * TM > N) {
                            imx::joint_matrix_store_checked(sg, C[a][j], dg, ldd, mx::layout::row_major, N, M, t0 + a * TM, row);
                        } else {
                            mx::joint_matrix_store(sg, C[a][j], dg + (size_t) (t0 + a * TM) * ldd + row, ldd, mx::layout::row_major);
                        }
                    }
            }
          };
          if ((int) (it.get_group(0) % ntt) >= nfull) { body(std::true_type{}); } else { body(std::false_type{}); }
        });
    });
}
} // namespace dqg

bool dq_gemm_q4k_supported(int64_t M, int64_t N, int64_t K) {
    return K > 0 && K % 256 == 0 && K % dqg::KC == 0 && M > 0 && M % dqg::WG_R == 0 && N >= 1 &&
           M < (1ll << 24) && N < (1ll << 24) && K < (1ll << 24);
}

sycl::event dq_gemm_q4k(sycl::queue & q, const void * w, const sycl::half * act, float * dst,
                        int64_t M, int64_t N, int64_t K, int64_t ldd) {
    const int nfull = (int) (N / dqg::WG_T);
    const int tail  = (int) (N - (int64_t) nfull * dqg::WG_T);
    return dqg::launch(q, (const uint8_t *) w, act, dst, (int) M, (int) N, (int) K, (int) ldd, nfull, nfull + (tail > 0 ? 1 : 0));
}
