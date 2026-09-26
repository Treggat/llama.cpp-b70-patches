//
// Flash-attention decode / verify on the Xe2 matrix units - see fattn-xmx.hpp
//
// Kernel: b70-secret-sauce/xmx-probe/xmx_fa.cpp, VAR 1 (NSG 8, TPS 8, GRF 256, QS 2 for R > 24), host harness removed.
//
#include "fattn-xmx.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#if GGML_SYCL_XMX
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#endif

static int ggml_sycl_fattn_xmx_env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e && *e ? atoi(e) : def;
}

bool ggml_sycl_fattn_xmx_env() {
    static const bool v = ggml_sycl_fattn_xmx_env_int("GGML_SYCL_XMX_FA", 0) != 0;   // default off
    return v;
}

// kernel geometry (xmx_fa.cpp): 8 sub-groups of 16 lanes, 8 tokens per sub-group per iteration
static constexpr int XFA_D    = 256;             // head dim, K and V
static constexpr int XFA_SG   = 16;
static constexpr int XFA_NSG  = 8;
static constexpr int XFA_TPS  = 8;
static constexpr int XFA_T    = XFA_NSG * XFA_TPS;   // tokens per iteration (64)
static constexpr int XFA_DVS  = XFA_D / XFA_NSG;     // dv columns per sub-group in the PV product (32)
static constexpr int XFA_NTV  = XFA_DVS / 16;        // 16-wide dv tiles per sub-group (2)
static constexpr int XFA_MAXR = 48;                  // query rows (GQA ratio x tokens) per KV head
static_assert(XFA_TPS % 8 == 0 && XFA_T % 16 == 0 && XFA_DVS % 16 == 0, "bad NSG/TPS");

static int ggml_sycl_fattn_xmx_min_cols() {
    // default 2: at 1 token the kernel measured only ~5% faster than TILE per layer (472 vs 496 us at kv 65536)
    static const int v = std::max(1, ggml_sycl_fattn_xmx_env_int("GGML_SYCL_XMX_FA_MIN_COLS", 2));
    return v;
}

static int ggml_sycl_fattn_xmx_max_cols() {
    static const int v = ggml_sycl_fattn_xmx_env_int("GGML_SYCL_XMX_FA_MAX_COLS", 8);
    return v;
}

bool ggml_sycl_fattn_xmx_can_use(int device, const ggml_tensor * dst) {
#if !GGML_SYCL_XMX
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#else
    if (!ggml_sycl_fattn_xmx_env() || dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    if (!Q || !K || !V || !mask || sinks) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 ||
        mask->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (Q->ne[0] != XFA_D || K->ne[0] != XFA_D || V->ne[0] != XFA_D || dst->ne[0] != XFA_D) {
        return false;
    }
    // one sequence (a unified KV cache); the mask is broadcast over the heads
    if (Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    const int64_t nb = Q->ne[1];
    const int64_t kv = K->ne[1];
    if (nb < ggml_sycl_fattn_xmx_min_cols() || nb > ggml_sycl_fattn_xmx_max_cols()) {
        return false;
    }
    if (K->ne[2] <= 0 || Q->ne[2] % K->ne[2] != 0 || V->ne[2] != K->ne[2] || V->ne[1] != kv) {
        return false;
    }
    if ((Q->ne[2] / K->ne[2]) * nb > XFA_MAXR) {
        return false;
    }
    // whole 64-token iterations only (llama.cpp pads the KV length to 256); no tail masking
    if (kv <= 0 || kv % XFA_T != 0 || kv > INT32_MAX / 2 || mask->ne[0] < kv || mask->ne[1] < nb) {
        return false;
    }
    if (dst->ne[1] != Q->ne[2] || dst->ne[2] != nb) {
        return false;
    }
    // element-contiguous rows; the K/V row loads are 2D block loads from global: 64 B aligned base and pitches
    if (Q->nb[0] != sizeof(float) || K->nb[0] != sizeof(sycl::half) || V->nb[0] != sizeof(sycl::half) ||
        mask->nb[0] != sizeof(sycl::half) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (Q->nb[1] % sizeof(float) || Q->nb[2] % sizeof(float) || dst->nb[1] % sizeof(float) ||
        dst->nb[2] % sizeof(float) || mask->nb[1] % sizeof(sycl::half)) {
        return false;
    }
    for (const ggml_tensor * t : { K, V }) {
        if (t->nb[1] % 64 != 0 || t->nb[2] % 64 != 0 || ((uintptr_t) t->data) % 64 != 0) {
            return false;
        }
    }
    return ggml_sycl_fattn_xmx_device_ok(device);
#endif
}

bool ggml_sycl_fattn_xmx_can_use(ggml_backend_sycl_context & ctx, const ggml_tensor * dst) {
    return ggml_sycl_fattn_xmx_can_use(ctx.device, dst);
}

#if GGML_SYCL_XMX

namespace mx   = sycl::ext::oneapi::experimental::matrix;
namespace imx  = sycl::ext::intel::experimental::matrix;
namespace oexp = sycl::ext::oneapi::experimental;

bool ggml_sycl_fattn_xmx_device_ok(int device) {
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
                for (const auto & c : dev.get_info<oexp::info::device::matrix_combinations>()) {
                    // a size of 0 means "any up to max_*"
                    const bool m_ok = c.msize == 8 || (c.msize == 0 && c.max_msize >= 8);
                    const bool n_ok = c.nsize == 16 || (c.nsize == 0 && c.max_nsize >= 16);
                    const bool k_ok = c.ksize == 16 || (c.ksize == 0 && c.max_ksize >= 16);
                    if (c.atype == mx::matrix_type::fp16 && c.btype == mx::matrix_type::fp16 &&
                        c.ctype == mx::matrix_type::fp32 && m_ok && n_ok && k_ok) {
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

using sycl::access::address_space;
using sycl::access::decorated;
using half = sycl::half;

struct xfa_params {
    const float * Q; const half * K; const half * V; const half * mask;
    float * Opart; sycl::float2 * ML; float * dst;
    int kv, nb, nh, nhkv, ratio, R;
    int64_t q_s1, q_s2;         // Q strides (floats): token, head
    int64_t k_s1, k_s2;         // K strides (halves): token, head
    int64_t v_s1, v_s2;         // V strides (halves)
    int64_t m_s1;               // mask row (query token) stride (halves)
    int64_t d_s1, d_s2;         // dst strides (floats): head, token
    int chunk, nchunks;
    float scale;
    int qs, Rw;                 // query-row splits per (KV head, chunk) and rows per split (R = qs * Rw)
};

// One work-group of NSG sub-groups per (KV head, KV chunk, query split). Per iteration of T = 64 tokens:
//   1. S^T[tok, row] = K Q^T      A = K tile (8 tok x 16 d, row-major from global), B = scaled Q^T (VNNI in SLM)
//   2. online softmax per row     f32, exp2; P (f16) and the per-row O rescale -> SLM
//   3. O[row, dv] = O*alpha + P V A = P (SLM), B = V tile (16 tok x 16 dv, row-major from global);
//                                 sub-group s owns dv columns [s*32, s*32+32) for all rows
// Partial (O, m, l) per work-group -> Opart / ML, merged by xfa_launch_combine.
template <int MT8, int NT16>
void xfa_launch_main(sycl::queue & q, const xfa_params p) {
    constexpr int NSG = XFA_NSG, SG = XFA_SG, TPS = XFA_TPS, T = XFA_T, D = XFA_D, DVS = XFA_DVS, NTV = XFA_NTV;
    constexpr int R16 = NT16 * 16, R8 = MT8 * 8;
    constexpr int SST = R16 + 4;                  // S^T row stride: +4 floats makes the column reads of step 2 conflict-free
    constexpr int RPS = (R8 + NSG - 1) / NSG;     // softmax rows per sub-group
    const int ngroups = p.nhkv * p.nchunks * p.qs;
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<half, 1>  Qp(sycl::range<1>(R16 * D), h);     // Q^T, VNNI packed: (d, r) -> (d/2)*(2*R16) + 2r + (d&1)
        sycl::local_accessor<float, 1> St(sycl::range<1>(T * SST), h);     // S^T [tok][row]
        sycl::local_accessor<half, 1>  Ps(sycl::range<1>(R8 * T), h);      // P   [row][tok]
        sycl::local_accessor<float, 1> Al(sycl::range<1>(R8), h);          // per-row rescale of O for this iteration
        sycl::local_accessor<float, 1> Mr(sycl::range<1>(R8), h);          // running row max (log2 domain)
        sycl::local_accessor<float, 1> Lr(sycl::range<1>(R8), h);          // running row sum
        h.parallel_for(sycl::nd_range<1>((size_t) ngroups * NSG * SG, NSG * SG),
            oexp::properties{ sycl::ext::intel::experimental::grf_size<256> },
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int s     = sg.get_group_linear_id();
            const int lane  = sg.get_local_linear_id();
            const int tid   = it.get_local_linear_id();
            const int g     = it.get_group_linear_id();
            const int split = g % p.qs;          // query-row split: adjacent groups share the same K/V chunk (L2 reuse)
            const int gh    = g / p.qs;
            const int hk    = gh % p.nhkv;
            const int ch    = gh / p.nhkv;
            const int rbase = split * p.Rw;
            const int c0    = ch * p.chunk;
            const int c1    = sycl::min(p.kv, c0 + p.chunk);
            const float LOG2E = 1.4426950408889634f;

            half  * qp = Qp.get_multi_ptr<decorated::no>().get();
            float * st = St.get_multi_ptr<decorated::no>().get();
            half  * ps = Ps.get_multi_ptr<decorated::no>().get();
            float * al = Al.get_multi_ptr<decorated::no>().get();
            float * mr = Mr.get_multi_ptr<decorated::no>().get();
            float * lr = Lr.get_multi_ptr<decorated::no>().get();

            // ---- stage Q^T (scaled, f16, VNNI packed) ----
            for (int i = tid; i < R16 * D; i += NSG * SG) {
                const int r = i / D, d = i % D;
                float v = 0.f;
                if (r < p.Rw) {
                    const int rg = rbase + r;
                    const int t = rg / p.ratio, hq = hk * p.ratio + rg % p.ratio;
                    v = p.Q[t * p.q_s1 + hq * p.q_s2 + d] * p.scale;
                }
                qp[(d >> 1) * (2 * R16) + 2 * r + (d & 1)] = (half) v;
            }
            for (int r = tid; r < R8; r += NSG * SG) { mr[r] = -1e30f; lr[r] = 0.f; }

            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16> O[MT8][NTV];
#pragma unroll
            for (int mt = 0; mt < MT8; ++mt)
#pragma unroll
                for (int nv = 0; nv < NTV; ++nv) mx::joint_matrix_fill(sg, O[mt][nv], 0.f);

            const half * Kh = p.K + hk * p.k_s2;
            const half * Vh = p.V + hk * p.v_s2;
            auto lQp = sycl::address_space_cast<address_space::local_space, decorated::no>(qp);
            auto lSt = sycl::address_space_cast<address_space::local_space, decorated::no>(st);
            auto lPs = sycl::address_space_cast<address_space::local_space, decorated::no>(ps);

            sycl::group_barrier(it.get_group());

            for (int tok0 = c0; tok0 < c1; tok0 += T) {
                // ---- 1. S^T = K Q^T for this sub-group's TPS tokens ----
                {
                    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16> S[TPS / 8][NT16];
#pragma unroll
                    for (int mt = 0; mt < TPS / 8; ++mt)
#pragma unroll
                        for (int nt = 0; nt < NT16; ++nt) mx::joint_matrix_fill(sg, S[mt][nt], 0.f);
                    for (int kk = 0; kk < D / 16; ++kk) {
                        mx::joint_matrix<sycl::sub_group, half, mx::use::b, 16, 16, mx::layout::ext_intel_packed> B[NT16];
#pragma unroll
                        for (int nt = 0; nt < NT16; ++nt)
                            mx::joint_matrix_load(sg, B[nt], lQp + kk * 8 * (2 * R16) + nt * 32, 2 * R16);
#pragma unroll
                        for (int mt = 0; mt < TPS / 8; ++mt) {
                            mx::joint_matrix<sycl::sub_group, half, mx::use::a, 8, 16, mx::layout::row_major> A;
                            mx::joint_matrix_load(sg, A,
                                sycl::address_space_cast<address_space::global_space, decorated::no>(
                                    Kh + (int64_t) (tok0 + s * TPS + mt * 8) * p.k_s1 + kk * 16), p.k_s1);
#pragma unroll
                            for (int nt = 0; nt < NT16; ++nt) mx::joint_matrix_mad(sg, S[mt][nt], A, B[nt], S[mt][nt]);
                        }
                    }
#pragma unroll
                    for (int mt = 0; mt < TPS / 8; ++mt)
#pragma unroll
                        for (int nt = 0; nt < NT16; ++nt)
                            mx::joint_matrix_store(sg, S[mt][nt], lSt + (s * TPS + mt * 8) * SST + nt * 16, SST, mx::layout::row_major);
                }
                sycl::group_barrier(it.get_group());

                // ---- 2. online softmax: sub-group s owns rows s, s+NSG, ... (unrolled so the row chains interleave) ----
#pragma unroll
                for (int ri = 0; ri < RPS; ++ri) {
                    const int r = s + ri * NSG;
                    if (r >= R8) continue;
                    if (r >= p.Rw) {   // padding row: P = 0, no rescale
                        for (int j = lane; j < T; j += SG) ps[r * T + j] = (half) 0.f;
                        if (lane == 0) al[r] = 1.f;
                        continue;
                    }
                    const int tq = (rbase + r) / p.ratio;
                    float x[T / SG];
                    float mx_ = -INFINITY;
#pragma unroll
                    for (int i = 0; i < T / SG; ++i) {
                        const int j = lane + i * SG;
                        x[i] = (st[j * SST + r] + (float) p.mask[tq * p.m_s1 + tok0 + j]) * LOG2E;
                        mx_ = sycl::fmax(mx_, x[i]);
                    }
                    mx_ = sycl::reduce_over_group(sg, mx_, sycl::maximum<float>());
                    const float m_old = mr[r];
                    const float m_new = sycl::fmax(m_old, mx_);
                    float sum = 0.f;
                    for (int i = 0; i < T / SG; ++i) {
                        const float pe = sycl::exp2(x[i] - m_new);
                        sum += pe;
                        ps[r * T + lane + i * SG] = (half) pe;
                    }
                    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
                    if (lane == 0) {
                        const float a = sycl::exp2(m_old - m_new);
                        al[r] = a;
                        mr[r] = m_new;
                        lr[r] = lr[r] * a + sum;
                    }
                }
                sycl::group_barrier(it.get_group());

                // ---- 3. O = O*alpha + P V for this sub-group's dv columns ----
#pragma unroll
                for (int mt = 0; mt < MT8; ++mt) {
                    float a[8];
                    bool any = false;
#pragma unroll
                    for (int i = 0; i < 8; ++i) { a[i] = al[mt * 8 + i]; any |= a[i] != 1.f; }
                    if (any) {   // uniform: the running max of these rows moved this iteration
#pragma unroll
                        for (int nv = 0; nv < NTV; ++nv)
                            imx::joint_matrix_apply(sg, O[mt][nv], [&](float & v, size_t row, size_t) { v *= a[row]; });
                    }
                }
#pragma unroll
                for (int kk = 0; kk < T / 16; ++kk) {
                    mx::joint_matrix<sycl::sub_group, half, mx::use::b, 16, 16, mx::layout::row_major> B[NTV];
#pragma unroll
                    for (int nv = 0; nv < NTV; ++nv)
                        mx::joint_matrix_load(sg, B[nv],
                            sycl::address_space_cast<address_space::global_space, decorated::no>(
                                Vh + (int64_t) (tok0 + kk * 16) * p.v_s1 + s * DVS + nv * 16), p.v_s1);
#pragma unroll
                    for (int mt = 0; mt < MT8; ++mt) {
                        mx::joint_matrix<sycl::sub_group, half, mx::use::a, 8, 16, mx::layout::row_major> A;
                        mx::joint_matrix_load(sg, A, lPs + mt * 8 * T + kk * 16, T);
#pragma unroll
                        for (int nv = 0; nv < NTV; ++nv) mx::joint_matrix_mad(sg, O[mt][nv], A, B[nv], O[mt][nv]);
                    }
                }
                // no third barrier: the next iteration's step 1 only writes St, which nobody reads any more; Ps/Al are
                // rewritten only after the next step-1 barrier, which every sub-group reaches after finishing step 3.
            }

            // ---- partial results ----
            float * op = p.Opart + (int64_t) g * R8 * D;
#pragma unroll
            for (int mt = 0; mt < MT8; ++mt)
#pragma unroll
                for (int nv = 0; nv < NTV; ++nv)
                    mx::joint_matrix_store(sg, O[mt][nv],
                        sycl::address_space_cast<address_space::global_space, decorated::no>(op + mt * 8 * D + s * DVS + nv * 16),
                        D, mx::layout::row_major);
            sycl::group_barrier(it.get_group());
            for (int r = tid; r < R8; r += NSG * SG) p.ML[(int64_t) g * R8 + r] = sycl::float2(mr[r], lr[r]);
        });
    });
}

// combine the chunks: one work-group of 256 work-items (one per dv) per (KV head, real row)
template <int R8>
void xfa_launch_combine(sycl::queue & q, const xfa_params p) {
    constexpr int D = XFA_D;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::nd_range<1>((size_t) p.nhkv * p.R * D, D), [=](sycl::nd_item<1> it) {
            const int dv = it.get_local_linear_id();
            const int gi = it.get_group_linear_id();
            const int hk = gi % p.nhkv, r = gi / p.nhkv;
            const int split = r / p.Rw, rl = r % p.Rw;
            float M = -1e30f;
            for (int c = 0; c < p.nchunks; ++c) M = sycl::fmax(M, p.ML[((int64_t) (c * p.nhkv + hk) * p.qs + split) * R8 + rl].x());
            float L = 0.f, acc = 0.f;
            for (int c = 0; c < p.nchunks; ++c) {
                const int64_t gg = (int64_t) (c * p.nhkv + hk) * p.qs + split;
                const sycl::float2 ml = p.ML[gg * R8 + rl];
                const float w = sycl::exp2(ml.x() - M);
                L   += w * ml.y();
                acc += w * p.Opart[(gg * R8 + rl) * D + dv];
            }
            const int t = r / p.ratio, hq = hk * p.ratio + r % p.ratio;
            // a row whose every key is masked has L == 0: write 0 rather than 0/0
            p.dst[t * p.d_s2 + hq * p.d_s1 + dv] = L > 0.f ? acc / L : 0.f;
        });
    });
}

template <int MT8, int NT16>
void xfa_run(sycl::queue & q, const xfa_params & p) {
    xfa_launch_main<MT8, NT16>(q, p);
    xfa_launch_combine<MT8 * 8>(q, p);
}

void xfa_dispatch(sycl::queue & q, const xfa_params & p) {
    const int R8 = (p.Rw + 7) / 8, R16 = (p.Rw + 15) / 16;
#define XFA_CASE(a, b) if (R8 == a && R16 == b) { xfa_run<a, b>(q, p); return; }
    XFA_CASE(1, 1) XFA_CASE(2, 1) XFA_CASE(3, 2) XFA_CASE(4, 2) XFA_CASE(5, 3) XFA_CASE(6, 3)
#undef XFA_CASE
    GGML_ABORT("XMX FA: unsupported rows per work-group %d", p.Rw);
}

} // namespace

void ggml_sycl_fattn_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int nb    = (int) Q->ne[1];
    const int nh    = (int) Q->ne[2];
    const int nhkv  = (int) K->ne[2];
    const int kv    = (int) K->ne[1];
    const int ratio = nh / nhkv;
    const int R     = nb * ratio;
    GGML_ASSERT(R <= XFA_MAXR && kv % XFA_T == 0);

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    // query-token splits per (head, chunk): default 2 for R > 24 with an even token count (nb 6/8), measured
    // 704->614 us (nb 6) and 762->637 us (nb 8) at kv 65536; the split work-groups are adjacent so the second
    // K/V read mostly hits L2
    static const int qs_env = ggml_sycl_fattn_xmx_env_int("GGML_SYCL_XMX_FA_QS", 0);
    int qs = (R > 24 && nb % 2 == 0) ? 2 : 1;
    if (qs_env > 0 && nb % qs_env == 0 && (R / qs_env) <= XFA_MAXR) {
        qs = qs_env;
    }
    const int Rw  = R / qs;
    const int R8p = (Rw + 7) / 8 * 8;   // rows per work-group, padded to the 8-row accumulator

    // split-KV: about 4 work-groups per Xe core in total (ncu = EUs; the B70: 256 -> 32 chunks per KV head),
    // chunk a multiple of the 64-token iteration (2048 at kv 65536, 1024 at 32768)
    const int id  = ctx.device;
    const int ncu = std::max(8, ggml_sycl_info().devices[id].nsm * 16);
    static const int chunk_env = ggml_sycl_fattn_xmx_env_int("GGML_SYCL_XMX_FA_CHUNK", 0);
    int chunk;
    if (chunk_env > 0) {
        chunk = chunk_env;
    } else {
        const int want = std::max(1, (ncu / 8) * 4 / nhkv);
        chunk = std::max(XFA_T, ((kv + want - 1) / want + XFA_T - 1) / XFA_T * XFA_T);
    }
    chunk = (chunk + XFA_T - 1) / XFA_T * XFA_T;
    const int nchunks = (kv + chunk - 1) / chunk;

    const size_t ngroups = (size_t) nchunks * nhkv * qs;
    ggml_sycl_pool_alloc<float>        opart(ctx.pool(), ngroups * R8p * XFA_D);
    ggml_sycl_pool_alloc<sycl::float2> ml(ctx.pool(), ngroups * R8p);

    xfa_params p;
    p.Q       = (const float *) Q->data;
    p.K       = (const half *) K->data;
    p.V       = (const half *) V->data;
    p.mask    = (const half *) mask->data;
    p.Opart   = opart.get();
    p.ML      = ml.get();
    p.dst     = (float *) dst->data;
    p.kv      = kv;
    p.nb      = nb;
    p.nh      = nh;
    p.nhkv    = nhkv;
    p.ratio   = ratio;
    p.R       = R;
    p.q_s1    = Q->nb[1] / sizeof(float);
    p.q_s2    = Q->nb[2] / sizeof(float);
    p.k_s1    = K->nb[1] / sizeof(half);
    p.k_s2    = K->nb[2] / sizeof(half);
    p.v_s1    = V->nb[1] / sizeof(half);
    p.v_s2    = V->nb[2] / sizeof(half);
    p.m_s1    = mask->nb[1] / sizeof(half);
    p.d_s1    = dst->nb[1] / sizeof(float);
    p.d_s2    = dst->nb[2] / sizeof(float);
    p.chunk   = chunk;
    p.nchunks = nchunks;
    p.scale   = scale;
    p.qs      = qs;
    p.Rw      = Rw;

    xfa_dispatch(*ctx.stream(), p);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

#else  // !GGML_SYCL_XMX

bool ggml_sycl_fattn_xmx_device_ok(int) { return false; }

void ggml_sycl_fattn_xmx(ggml_backend_sycl_context &, ggml_tensor *) {
    GGML_ABORT("XMX flash-attention path not built (configure with -DGGML_SYCL_XMX=ON)");
}

#endif
