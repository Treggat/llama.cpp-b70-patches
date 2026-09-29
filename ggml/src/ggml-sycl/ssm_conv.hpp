#pragma once

#include "common.hpp"

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst);

// LOCAL (verifystep) GGML_SYCL_CONV_STEP: the qwen35 conv-state step of a verify / decode batch (n_t <= 16) in one launch:
// CONCAT(conv state, new qkv columns) -> conv_input, its n_snap rollback-snapshot CPYs, and (conv_out != nullptr) the
// SSM_CONV + SiLU of the unfused fused-conv path (same ssm_conv_element<4> code). One work-item per channel.
struct ggml_sycl_conv_step_snap { float * dst[16]; int off[16]; };
void ggml_sycl_conv_step(ggml_backend_sycl_context & ctx, const char * cs, int64_t cs_nb0, int64_t cs_nb1, const char * qx,
                         int64_t q_nb0, int64_t q_nb1, float * ci, int C, int n_t, const ggml_sycl_conv_step_snap & snap,
                         int n_snap, const float * w, float * conv_out);
