#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <optional>
#include <unordered_map>
#include <vector>

#include "fattn-onednn.hpp"
#include "fattn-tile.hpp"
#include "convert.hpp"

// set minimum query length to treat as prefill (32)
#define GGML_SYCL_FA_ONEDNN_MIN_Q 32

bool ggml_sycl_fattn_onednn_binds_kv(const ggml_tensor * K, const ggml_tensor * V) {
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        return false;
    }
    auto bindable = [](const ggml_tensor * t) {
        return t->nb[0] == sizeof(sycl::half) && t->nb[1] % sizeof(sycl::half) == 0 &&
               t->nb[2] % sizeof(sycl::half) == 0 && t->nb[3] % sizeof(sycl::half) == 0;
    };
    return bindable(K) && bindable(V);
}

bool ggml_sycl_flash_attn_ext_onednn_supported(const ggml_tensor * dst, bool use_shape_limit) {
#if !GGML_SYCL_DNNL
    GGML_UNUSED(dst);
    GGML_UNUSED(use_shape_limit);
    return false;
#else
    if (!g_ggml_sycl_fa_onednn) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    // F16 KV: native SDPA at any KV length.
    // Non-F16: dequant to F16 then SDPA at prefill lengths. Only the
    // standard quantized KV cache types (Q4_0-Q8_0) and F32 are accepted
    // because their to_fp16_sycl conversion is verified. BF16 and IQ*
    // are excluded: BF16 needs a strided conversion kernel that does not
    // exist yet; IQ types are model-weight-only quants with no dequant
    // registration and are never used as KV caches.
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        auto kt = K->type, vt = V->type;
        bool k_ok = kt == GGML_TYPE_F32 || kt == GGML_TYPE_Q4_0 || kt == GGML_TYPE_Q4_1 ||
                    kt == GGML_TYPE_Q5_0 || kt == GGML_TYPE_Q5_1 || kt == GGML_TYPE_Q8_0;
        bool v_ok = vt == GGML_TYPE_F32 || vt == GGML_TYPE_Q4_0 || vt == GGML_TYPE_Q4_1 ||
                    vt == GGML_TYPE_Q5_0 || vt == GGML_TYPE_Q5_1 || vt == GGML_TYPE_Q8_0;
        if (!k_ok || !v_ok) {
            return false;
        }
        if (use_shape_limit && (Q->ne[1] < 32 || K->ne[1] < 1024)) {
            return false;
        }
        for (const ggml_tensor * t : {K, V}) {
            if (t->type == GGML_TYPE_F16 && t->nb[1] % (t->ne[0] * 2) != 0) {
                return false;
            }
        }
    }
    // This is the improved SPDA gate. Rather than gating Alchemist GPUs from all SPDA features, we instead target only the failing shapes.
    // If the GPU being assessed isn't in the grouping below, it has full access to all SPDA shapes. Otherwise, if it's an Alchemist GPU, we block only the shapes with head sizes that fail.
    // It is much easier to compare the device to a small list of failing cases than to define all the passing ones.
    const gpu_arch arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    bool support_spda = !(arch == gpu_arch::intel_gpu_dg2_g10 ||
                         arch == gpu_arch::intel_gpu_dg2_g11 ||
                         arch == gpu_arch::intel_gpu_dg2_g12);

    if (!support_spda && K->ne[0] == 64) {
        return false;
    }
    // Optional KV-length ceiling (GGML_SYCL_FA_ONEDNN_MAX_KV, 0 = unlimited). Escape hatch:
    // very long sequences make the fused SDPA slow enough to risk the xe driver watchdog on
    // some stacks; past the cap we fall back to the native FA kernel instead.
    if (g_ggml_sycl_fa_onednn_max_kv > 0 && K->ne[1] > g_ggml_sycl_fa_onednn_max_kv) {
        return false;
    }
    // gate for the following cases
    // 1. if the oneDNN graph Add node has no input --> skip
    // 2. types other than f16 need different logical_tensor declaration
    // 3. the mask must be shape [1, 1, q, seq]
    // 4. sinks: excludes attention sink (Xiao et al., 2024) that can't be modeled by oneDNN graph
    if (!mask || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1 || sinks) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    // K and V must share head_dim: the SDPA graph uses a single `d` for both.
    const int64_t d = K->ne[0];
    if (V->ne[0] != d || Q->ne[3] != 1) {
        return false;
    }
    // GQA must divide evenly.
    if (K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    // Prefill only.
    if (use_shape_limit && Q->ne[1] < GGML_SYCL_FA_ONEDNN_MIN_Q) {
        return false;
    }
    return true;
#endif
}

void ggml_sycl_fa_mask_ensure(ggml_backend_sycl_context & ctx, const ggml_tensor * mask_c) {
    if (!mask_c || mask_c->op_params[0] != GGML_KQ_MASK_HINT_CAUSAL || mask_c->op_params[3] != 1) {
        return;
    }
    // test hook (negative control of the GGML_SYCL_FA_MASK_POISON test): PX_NO_ENSURE=1 skips the rebuild
    static const bool no_ensure = getenv("PX_NO_ENSURE") != nullptr;
    if (no_ensure) {
        return;
    }
    ggml_tensor * mask = const_cast<ggml_tensor *>(mask_c);   // the flag lives in the host-side tensor struct
    GGML_ASSERT(mask->type == GGML_TYPE_F16);
    const int64_t n_used = mask->op_params[1];
    const int64_t n_rows = mask->op_params[2];
    const int64_t ne0 = mask->ne[0], ne1 = mask->ne[1], ne2 = mask->ne[2], ne3 = mask->ne[3];
    const size_t  nb1 = mask->nb[1], nb2 = mask->nb[2], nb3 = mask->nb[3];
    char * base = (char *) mask->data;
    const int64_t n = ne0 * ne1 * ne2 * ne3;
    const sycl::half zero = sycl::half(0.0f), ninf = sycl::half(-INFINITY);
    ctx.stream()->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        int64_t i = ix[0];
        const int64_t i0 = i % ne0; i /= ne0;
        const int64_t i1 = i % ne1; i /= ne1;
        const int64_t i2 = i % ne2; const int64_t i3 = i / ne2;
        const int64_t last = i1 < n_rows ? n_used - n_rows + i1 : -1;
        *(sycl::half *) (base + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * sizeof(sycl::half)) = i0 <= last ? zero : ninf;
    });
    mask->op_params[3] = 2;
}

#if GGML_SYCL_DNNL

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"   // graph API lives only under oneapi/dnnl/, not at the include root

using namespace dnnl;
using namespace dnnl::graph;

// strided src (f16 or f32) -> contiguous f16 [ne0,ne1,ne2,ne3] (ne0 innermost). nb* are BYTE strides.
template <typename src_t>
static void cont_to_f16_sycl(const char * src, sycl::half * dst,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        size_t nb1, size_t nb2, size_t nb3, dpct::queue_ptr stream) {
    const int64_t n = ne0 * ne1 * ne2 * ne3;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        const int64_t gid = ix[0];
        int64_t       i   = gid;
        const int64_t i0 = i % ne0; i /= ne0;
        const int64_t i1 = i % ne1; i /= ne1;
        const int64_t i2 = i % ne2; const int64_t i3 = i / ne2;
        const src_t * p = (const src_t *) (src + i1 * nb1 + i2 * nb2 + i3 * nb3) + i0;
        dst[gid] = (sycl::half) (*p);
    });
}

// oneDNN SDPA out (f16 contiguous [mb,H,q,d]) -> ggml dst (f32 [head_dim,H,n_tok,mb], contiguous).
static void permute_sdpa_out_sycl(const sycl::half * out, float * dst,
        int64_t mb, int64_t H, int64_t q, int64_t d, dpct::queue_ptr stream) {
    const int64_t n = mb * H * q * d;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        const int64_t gid = ix[0];
        int64_t       i   = gid;
        const int64_t e = i % d; i /= d;
        const int64_t t = i % q; i /= q;
        const int64_t h = i % H; const int64_t b = i / H;
        dst[e + h * d + t * d * H + b * d * H * q] = (float) out[gid];
    });
}

struct sdpa_partition {
    compiled_partition          cp;
    std::vector<logical_tensor> ins;
    logical_tensor              out;
    size_t id_q = 0, id_k = 0, id_v = 0, id_scale = 0, id_mask = 0;
    // LOCAL (prefillx) implicit bottom-right causal mask: host-scalar inputs
    size_t id_kvlen = SIZE_MAX, id_qlen = SIZE_MAX, id_ninf = SIZE_MAX;
    bool   ok = false;
};

// LOCAL (prefillx): GGML_SYCL_FA_CAUSAL=1 (default 0). When the KQ mask carries the GGML_KQ_MASK_HINT_CAUSAL hint
// (llama sets it on a prefill ubatch whose mask is exactly bottom-right causal over the first n_used KV cells), the
// SDPA runs with oneDNN's implicit bottom-right causal mask over K/V cut to n_used cells instead of adding the
// explicit mask: no n_q x n_kv mask read per head, and the key loop of each query tile stops at the diagonal. The
// same scores, max and exp terms as the explicit mask (a 0 added is exact, a -inf term contributes 0); only
// oneDNN's kernel choice / tiling for the causal variant can change the summation order (rounding level).
static bool ggml_sycl_fa_causal_env() {
    static const bool v = [] {
        const char * e = getenv("GGML_SYCL_FA_CAUSAL");
        return e && atoi(e) != 0;
    }();
    return v;
}

// n_used from the mask hint when the causal variant applies to this op, else 0
static int64_t ggml_sycl_fa_causal_n_used(const ggml_tensor * mask, int64_t n_q, int64_t n_kv) {
    if (!mask || !ggml_sycl_fa_causal_env()) {
        return 0;
    }
    const int32_t * p = mask->op_params;
    if (p[0] != GGML_KQ_MASK_HINT_CAUSAL) {
        return 0;
    }
    const int64_t n_used = p[1];
    const int64_t n_rows = p[2];
    if (n_rows != n_q || mask->ne[1] < n_q || n_used < n_q || n_used > n_kv) {
        return 0;
    }
    return n_used;
}

// Build + compile the contiguous-input GQA SDPA graph (MatMul->Divide->Add->SoftMax->MatMul), f32 out.
// Mirrors the hardware-verified scratch/onednn_sdpa_probe.cpp build_gqa (partitions=1, sdp_primitive_kernel_t).
static sdpa_partition build_sdpa(const engine & eng, int H, int Hkv, int q, int seq, int d,
                                 const std::array<int64_t, 5> & k_str, const std::array<int64_t, 5> & v_str,
                                 bool causal = false) try {
    using ltype = logical_tensor::layout_type;
    using dt    = logical_tensor::data_type;
    using ldims = logical_tensor::dims;
    const dt    fi = dt::f32, t = dt::f16;
    const int   rep = H / Hkv;
    const ldims q_sz = {1, Hkv, rep, q, d}, kv_sz = {1, Hkv, 1, seq, d}, s_sz = {1, Hkv, rep, q, seq},
                sc = {1, 1, 1, 1, 1}, msk = {1, 1, 1, q, seq}, o_sz = {1, Hkv, rep, q, d};
    const ldims k_st(k_str.begin(), k_str.end()), v_st(v_str.begin(), v_str.end());
    int64_t        id = 0;
    sdpa_partition E;

    auto query  = logical_tensor(id++, t,  q_sz, ltype::strided);
    auto key    = logical_tensor(id++, t,  kv_sz, k_st);
    auto score  = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto bmm1   = op(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);          // key is [.., seq, d]
    bmm1.add_inputs({query, key}); bmm1.add_outputs({score});

    auto scale  = logical_tensor(id++, t,  sc,   ltype::strided);
    auto scaled = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto sdiv   = op(id++, op::kind::Divide, "scale_div");     // score / (1/kq_scale) == score * kq_scale
    sdiv.add_inputs({score, scale}); sdiv.add_outputs({scaled});

    auto mask   = logical_tensor(id++, t,  msk,  ltype::strided);
    auto masked = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto madd   = op(id++, op::kind::Add, "mask_add");
    madd.add_inputs({scaled, mask}); madd.add_outputs({masked});

    // LOCAL (prefillx) implicit bottom-right causal mask (oneDNN's sdpa_bottom_right_causal_mask pattern):
    // keep score[r][c] iff r + seq_len_kv - seq_len_q >= c, else -inf
    using ptype = logical_tensor::property_type;
    auto idx_row  = logical_tensor(id++, dt::s32, s_sz, ltype::strided);
    auto gi_row   = op(id++, op::kind::GenIndex, "gen_index_row");
    gi_row.set_attr<int64_t>(op::attr::axis, -2);
    gi_row.add_inputs({scaled}); gi_row.add_outputs({idx_row});
    auto kvlen    = logical_tensor(id++, dt::s32, 0, ltype::strided, ptype::host_scalar);
    auto r_add    = logical_tensor(id++, dt::s32, s_sz, ltype::strided);
    auto cadd     = op(id++, op::kind::Add, "causal_add");
    cadd.add_inputs({idx_row, kvlen}); cadd.add_outputs({r_add});
    auto qlen     = logical_tensor(id++, dt::s32, 0, ltype::strided, ptype::host_scalar);
    auto r_sub    = logical_tensor(id++, dt::s32, s_sz, ltype::strided);
    auto csub     = op(id++, op::kind::Subtract, "causal_sub");
    csub.add_inputs({r_add, qlen}); csub.add_outputs({r_sub});
    auto idx_col  = logical_tensor(id++, dt::s32, s_sz, ltype::strided);
    auto gi_col   = op(id++, op::kind::GenIndex, "gen_index_col");
    gi_col.set_attr<int64_t>(op::attr::axis, -1);
    gi_col.add_inputs({scaled}); gi_col.add_outputs({idx_col});
    auto keep     = logical_tensor(id++, dt::boolean, s_sz, ltype::strided);
    auto cge      = op(id++, op::kind::GreaterEqual, "causal_ge");
    cge.add_inputs({r_sub, idx_col}); cge.add_outputs({keep});
    auto ninf     = logical_tensor(id++, fi, 0, ltype::strided, ptype::host_scalar);
    auto cmasked  = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto csel     = op(id++, op::kind::Select, "causal_select");
    csel.add_inputs({keep, scaled, ninf}); csel.add_outputs({cmasked});

    auto probs  = logical_tensor(id++, t,  s_sz, ltype::strided);
    auto smax   = op(id++, op::kind::SoftMax, "softmax");
    smax.set_attr<int64_t>(op::attr::axis, -1);
    smax.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    smax.add_inputs({causal ? cmasked : masked}); smax.add_outputs({probs});

    auto value  = logical_tensor(id++, t,  kv_sz, v_st);
    // f16 output is REQUIRED to hit sdp_primitive_kernel_t (the systolic micro-kernel); an f32 output
    // falls to larger_partition_kernel_t which materializes N^2 (confirmed: scratch/onednn_sdpa_kernel_probe.cpp).
    // converted to the f32 ggml dst in the permute below.
    auto output = logical_tensor(id++, t,  o_sz, ltype::strided);   // f16 contiguous [mb,Hkv,rep,q,d]
    auto bmm2   = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value}); bmm2.add_outputs({output});

    dnnl::graph::graph g(eng.get_kind());
    g.add_op(bmm1); g.add_op(sdiv);
    if (causal) {
        g.add_op(gi_row); g.add_op(cadd); g.add_op(csub); g.add_op(gi_col); g.add_op(cge); g.add_op(csel);
    } else {
        g.add_op(madd);
    }
    g.add_op(smax); g.add_op(bmm2);
    g.finalize();

    auto parts = g.get_partitions();
    if (parts.size() != 1 || !parts[0].is_supported()) {
        GGML_LOG_WARN("%s: oneDNN did not fuse the SDPA graph%s; falling back\n", __func__, causal ? " (causal)" : "");
        return E;   // ok stays false -> caller falls back to TILE
    }
    E.ins      = parts[0].get_input_ports();
    E.out      = parts[0].get_output_ports()[0];
    E.cp       = parts[0].compile(E.ins, {E.out}, eng);
    E.out      = E.cp.query_logical_tensor(E.out.get_id());
    E.id_q     = query.get_id(); E.id_k = key.get_id(); E.id_v = value.get_id();
    E.id_scale = scale.get_id(); E.id_mask = mask.get_id();
    if (causal) {
        E.id_mask  = SIZE_MAX;
        E.id_kvlen = kvlen.get_id(); E.id_qlen = qlen.get_id(); E.id_ninf = ninf.get_id();
    }
    E.ok       = true;
    return E;
}
catch (const std::exception & e) {
    // compile() can reject a stride set the partitioner never inspects; memoise the failure so the
    // fallback costs one build rather than one per call.
    GGML_LOG_WARN("%s: oneDNN SDPA partition build failed (%s); falling back to TILE kernel\n", __func__, e.what());
    return {};
}

void ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int64_t d   = K->ne[0];   // head_dim
    const int64_t seq = K->ne[1];   // n_kv
    const int64_t Hkv = K->ne[2];   // n_head_kv
    const int64_t H   = Q->ne[2];   // n_head
    const int64_t q   = Q->ne[1];   // n_tok
    const int64_t mb  = Q->ne[3];   // batch (== 1, gated)

    float kq_scale = 1.0f;
    memcpy(&kq_scale, (const float *) dst->op_params + 0, sizeof(float));

    dpct::queue_ptr stream = ctx.stream();
    dnnl::engine    eng    = ctx.engine_dnnl(stream);
    dnnl::stream    strm   = ctx.stream_dnnl(stream);

    const ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);

    // Q: always f32 -- copy to dense f16.
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Qf_pool;
    sycl::half * Qf_ptr = (sycl::half *) extra.Q_buffer_ptr;
    if (!Qf_ptr) {
        Qf_pool.emplace(ctx.pool(), (size_t) H * q * d);
        Qf_ptr = Qf_pool->get();
    }
    cont_to_f16_sycl<float>((const char *) Q->data, Qf_ptr, d, q, H, mb, Q->nb[1], Q->nb[2], Q->nb[3], stream);

    // K/V: bind the f16 cache in place. llama.cpp permutes it to [token][head][dim], so its head
    // plane is strided rather than dense, which is what an explicit stride vector expresses.
    // Quantized and f32 KV still stage a dense copy -- the layout the k_str/v_str defaults describe.
    sycl::half * K_ptr = nullptr;
    sycl::half * V_ptr = nullptr;
    std::array<int64_t, 5> k_str{ Hkv * seq * d, seq * d, seq * d, d, 1 };
    std::array<int64_t, 5> v_str = k_str;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Kf_pool;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Vf_pool;
    // Helper: hand out reserved space, or fall back to the pool.
    auto stage_k = [&](size_t n) { if (extra.K_buffer_ptr) { return (sycl::half *) extra.K_buffer_ptr; }
                                  Kf_pool.emplace(ctx.pool(), n); return Kf_pool->get(); };
    auto stage_v = [&](size_t n) { if (extra.V_buffer_ptr) { return (sycl::half *) extra.V_buffer_ptr; }
                                  Vf_pool.emplace(ctx.pool(), n); return Vf_pool->get(); };

    auto elem_strides = [](const ggml_tensor * t) {
        const int64_t s1 = (int64_t) (t->nb[1] / t->nb[0]);
        const int64_t s2 = (int64_t) (t->nb[2] / t->nb[0]);
        const int64_t s3 = (int64_t) (t->nb[3] / t->nb[0]);
        // dims are {mb=1, Hkv, rep=1, seq, d}; the size-1 dims at 0 and 2 never advance an address.
        return std::array<int64_t, 5>{ s3, s2, s2, s1, 1 };
    };

    if (ggml_sycl_fattn_onednn_binds_kv(K, V)) {
        K_ptr = (sycl::half *) K->data;
        V_ptr = (sycl::half *) V->data;
        k_str = elem_strides(K);
        v_str = elem_strides(V);
    } else if (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) {
        K_ptr = stage_k((size_t) Hkv * seq * d);
        V_ptr = stage_v((size_t) Hkv * seq * d);
        cont_to_f16_sycl<sycl::half>((const char *) K->data, K_ptr, d, seq, Hkv, mb, K->nb[1], K->nb[2], K->nb[3], stream);
        cont_to_f16_sycl<sycl::half>((const char *) V->data, V_ptr, d, seq, Hkv, mb, V->nb[1], V->nb[2], V->nb[3], stream);
    } else if (ggml_is_quantized(K->type)) {
        // Quantized K/V: dequant to dense F16 using pool, same lifetime as F16 path.
        K_ptr = stage_k((size_t) ggml_nelements(K));
        {
            const char * K_data = (const char *)K->data;
            const bool k_non_dense = ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
            const bool k_gemma = k_non_dense &&
                ((int64_t)K->nb[2] < (int64_t)K->ne[1] * (int64_t)K->nb[1]);
            if (ggml_is_contiguously_allocated(K) && !k_non_dense) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(K->type, dst);
                to_fp16(K_data, K_ptr, ggml_nelements(K), stream);
            } else {
                const size_t bs = ggml_blck_size(K->type);
                const size_t ts = ggml_type_size(K->type);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(K->type);
                int64_t s01, s02, s03;
                if (k_gemma) {
                    const int64_t blk_per_row = (int64_t)K->ne[0] / bs;
                    s01 = (int64_t)Hkv * blk_per_row;
                    s02 = blk_per_row;
                    s03 = (int64_t)K->ne[1] * s01;
                } else {
                    s01 = (int64_t)K->nb[1] / ts;
                    s02 = (int64_t)K->nb[2] / ts;
                    s03 = (int64_t)K->nb[3] / ts;
                }
                to_fp16(K_data, K_ptr,
                        K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                        s01, s02, s03, stream);
            }
        }
        // Quantized V: always dequant separately. Even when K and V share
        // the same underlying allocation (V is a view of K with the same
        // data pointer), their logical values differ because the quantized
        // elements at different positions/offsets represent different K/V
        // data. Master's F16 path also never aliases K and V.
        V_ptr = stage_v((size_t) ggml_nelements(V));
        {
            const char * V_data = (const char *)V->data;
            const bool v_non_dense = ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;
            const bool v_gemma = v_non_dense &&
                ((int64_t)V->nb[2] < (int64_t)V->ne[1] * (int64_t)V->nb[1]);
            if (ggml_is_contiguously_allocated(V) && !v_non_dense) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(V->type, dst);
                to_fp16(V_data, V_ptr, ggml_nelements(V), stream);
            } else {
                const size_t bs = ggml_blck_size(V->type);
                const size_t ts = ggml_type_size(V->type);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(V->type);
                int64_t s01, s02, s03;
                if (v_gemma) {
                    const int64_t blk_per_row = (int64_t)V->ne[0] / bs;
                    s01 = (int64_t)V->ne[2] * blk_per_row;
                    s02 = blk_per_row;
                    s03 = (int64_t)V->ne[1] * s01;
                } else {
                    s01 = (int64_t)V->nb[1] / ts;
                    s02 = (int64_t)V->nb[2] / ts;
                    s03 = (int64_t)V->nb[3] / ts;
                }
                to_fp16(V_data, V_ptr,
                        V->ne[0], V->ne[1], V->ne[2], V->ne[3],
                        s01, s02, s03, stream);
            }
        }
    } else {
        // F32: strided copy to dense F16 via cont_to_f16_sycl<float>.
        K_ptr = stage_k((size_t) ggml_nelements(K));
        cont_to_f16_sycl<float>((const char *) K->data, K_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                                K->nb[1], K->nb[2], K->nb[3], stream);
        V_ptr = stage_v((size_t) ggml_nelements(V));
        cont_to_f16_sycl<float>((const char *) V->data, V_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3],
                                V->nb[1], V->nb[2], V->nb[3], stream);
    }

    // divide-by-(1/scale) reproduces ggml's score *= kq_scale on the proven probe graph.
    //
    // The scale must not be uploaded with an async memcpy from a stack local: on the in-order
    // queue that copy waits behind the K/V staging kernels, and once those take long enough
    // (n_kv >= ~26k on B70) the host frame is recycled before the copy runs, feeding the SDPA a
    // garbage scale (output collapses to a repeated token). Write the scalar from a kernel
    // instead -- the value is captured into the command, so no host memory has to outlive the
    // call, and the enqueue stays async.
    const sycl::half scale_h = (sycl::half) (1.0f / kq_scale);
    std::optional<ggml_sycl_pool_alloc<sycl::half>> scbuf;
    sycl::half * scale_dev = (sycl::half *) extra.scale_buffer_ptr;
    if (!scale_dev) {
        scbuf.emplace(ctx.pool(), 1);
        scale_dev = scbuf->get();
    }
    stream->single_task([=]() { *scale_dev = scale_h; });

    // f16 contiguous SDPA out [mb,H,q,d]
    std::optional<ggml_sycl_pool_alloc<sycl::half>> outf_pool;
    sycl::half * outf_ptr = (sycl::half *) extra.out_buffer_ptr;
    if (!outf_ptr) {
        outf_pool.emplace(ctx.pool(), (size_t) H * q * d);
        outf_ptr = outf_pool->get();
    }

    // compile once per (device, shape, KV strides), reuse across layers/calls. Stride 2 always
    // repeats stride 1 and stride 4 is always 1, so the key covers every entry that can differ.
    static std::unordered_map<std::string, sdpa_partition> cache;
    auto get_partition = [&](int64_t n_seq, bool causal) -> sdpa_partition & {
        char keyb[256];
        snprintf(keyb, sizeof(keyb), "%d:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%d", ggml_sycl_get_device(),
                 (long long) H, (long long) Hkv, (long long) q, (long long) n_seq, (long long) d,
                 (long long) k_str[0], (long long) k_str[1], (long long) k_str[3],
                 (long long) v_str[0], (long long) v_str[1], (long long) v_str[3], (int) causal);
        auto it = cache.find(keyb);
        if (it == cache.end()) {
            // LOCAL (prefillx) GGML_SYCL_FA_ONEDNN_LOG=1: host time of each partition build + compile
            static const bool log_build = getenv("GGML_SYCL_FA_ONEDNN_LOG") != nullptr;
            const auto t0 = std::chrono::steady_clock::now();
            it = cache.emplace(keyb, build_sdpa(eng, (int) H, (int) Hkv, (int) q, (int) n_seq, (int) d, k_str, v_str,
                                                causal)).first;
            if (log_build) {
                GGML_LOG_INFO("%s: SDPA partition q=%lld seq=%lld causal=%d built in %.1f ms (ok=%d, %zu cached)\n",
                              __func__, (long long) q, (long long) n_seq, (int) causal,
                              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                              (int) it->second.ok, cache.size());
            }
        }
        return it->second;
    };
    // LOCAL (prefillx) GGML_SYCL_FA_CAUSAL: the implicit-causal partition over the first n_used cells (the K/V
    // pointers and strides stay those of the n_kv-cell tensors; only the sequence dimension shrinks), else the
    // explicit-mask partition over all n_kv cells
    const int64_t n_used = ggml_sycl_fa_causal_n_used(mask, q, seq);
    sdpa_partition * Ep = nullptr;
    if (n_used > 0) {
        sdpa_partition & Ec = get_partition(n_used, true);
        if (Ec.ok) {
            Ep = &Ec;
        }
    }
    if (Ep == nullptr) {
        ggml_sycl_fa_mask_ensure(ctx, mask);   // the explicit-mask partition reads the mask
        Ep = &get_partition(seq, false);
    }
    sdpa_partition & E = *Ep;
    if (!E.ok) {
        // oneDNN can decline a shape or a stride set that _supported() never sees; build_sdpa warns per key.
        ggml_sycl_flash_attn_ext_tile(ctx, dst);
        return;
    }

    auto id2ptr = [&](size_t r) -> void * {
        if (r == E.id_q)     return Qf_ptr;
        if (r == E.id_k)     return K_ptr;
        if (r == E.id_v)     return V_ptr;
        if (r == E.id_scale) return scale_dev;
        if (r == E.id_mask)  return (void *) mask->data;
        return nullptr;
    };
    std::vector<tensor> ti;
    ti.reserve(E.ins.size());
    int32_t kvlen_v = (int32_t) n_used;
    int32_t qlen_v  = (int32_t) q;
    float   ninf_v  = -INFINITY;
    for (auto & lt : E.ins) {
        const size_t lid = lt.get_id();
        if (lid == E.id_kvlen) {
            ti.push_back(tensor::make_scalar_tensor(lt, &kvlen_v));
        } else if (lid == E.id_qlen) {
            ti.push_back(tensor::make_scalar_tensor(lt, &qlen_v));
        } else if (lid == E.id_ninf) {
            ti.push_back(tensor::make_scalar_tensor(lt, &ninf_v));
        } else {
            ti.emplace_back(lt, eng, id2ptr(lid));
        }
    }
    tensor to(E.out, eng, outf_ptr);
    E.cp.execute(strm, ti, {to});

    permute_sdpa_out_sycl(outf_ptr, (float *) dst->data, mb, H, q, d, stream);
    // Single device needs no sync: the dnnl stream wraps this same in-order queue, so the SDPA
    // serializes with the staging kernels before it and the permute/pool reuse after it. The
    // garbage output formerly blamed on the missing sync here was the scale use-after-return
    // fixed above. Keep the conservative wait for multi-GPU, where other devices' streams can
    // race the pool:
    if (ggml_sycl_info().device_count > 1) {
        stream->wait_and_throw();
    }
}
catch (const std::exception & e) {
    if (g_ggml_sycl_graph_recording) {
        // oneDNN SDPA cannot be recorded into a SYCL command graph (it depends on events from outside
        // the graph): give the whole ggml graph back to the recorder, which runs it eagerly with oneDNN
        throw;
    }
    // any oneDNN/SYCL failure is non-fatal: fall back to the existing kernel (strictly additive).
    GGML_LOG_WARN("%s: oneDNN SDPA failed (%s); falling back to TILE kernel\n", __func__, e.what());
    ggml_sycl_fa_mask_ensure(ctx, dst->src[3]);
    ggml_sycl_flash_attn_ext_tile(ctx, dst);
}

#endif // GGML_SYCL_DNNL
