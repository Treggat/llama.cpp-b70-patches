#pragma once

#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_sycl_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_sycl_try_gdn_cache_fusion)
void ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache);

// LOCAL (GGML_SYCL_GDN_GATHER): the input state read straight from the recurrent-state cache, row idx[seq] (the
// GET_ROWS in front of the op folded in); row = the cache's row stride in floats
struct ggml_sycl_gated_delta_net_state_gather {
    const float *   states;
    const int32_t * idx;
    int64_t         row;
};

void ggml_sycl_op_gated_delta_net_fused_cache_gather(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                                     ggml_sycl_gated_delta_net_fused_cache cache,
                                                     ggml_sycl_gated_delta_net_state_gather gather);

// LOCAL (prefill-gdn): chunked prefill kernel (gated_delta_net_chunked.cpp), GGML_SYCL_GDN_CHUNKED=1 (default off),
// d_k = d_v = 128, scalar gate, n_tokens >= GGML_SYCL_GDN_CHUNKED_MIN (default 128). launch returns false (nothing
// submitted) when the layout does not fit; the caller then runs the token-sequential kernel.
bool ggml_sycl_gdn_chunked_enabled(int64_t S_v, bool kda, int64_t n_tokens, int K);
bool ggml_sycl_gdn_chunked_launch(ggml_backend_sycl_context & ctx, const float * q, const float * k, const float * v,
                                  const float * g, const float * b, const float * s_in, float * dst, float * state,
                                  int64_t H, int64_t n_tokens, int64_t n_seqs, int64_t sq1, int64_t sq2, int64_t sq3,
                                  int64_t sv1, int64_t sv2, int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3,
                                  int64_t neqk1, int64_t rq3, float scale, int64_t slot_stride, int K,
                                  const int32_t * s_idx, int64_t s_row);
