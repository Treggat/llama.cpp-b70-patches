//
// q4_K x q8_1 matmul on the Xe2 matrix units (joint_matrix / DPAS) for up to 16 activation columns
// (3..16 by default, GGML_SYCL_XMX_Q4K_MIN_COLS / _MAX_COLS).
//
// MMVQ runs q4_K at memory bandwidth for one column but pays for every extra column (a speculative-decoding
// verify pass runs 2..6). This path is flat in the column count: the activations are the DPAS A tile
// (8 columns x 32), the weights the B tile (32 x 16 rows), so every lane of the accumulator owns one weight row
// and its scales in registers. It reads the MMVQ reorder (SoA) layout in place (no repack, no side buffers) and
// quantizes the activations with MMVQ's own q8_1 routine (quantize_q8_1_impl: same int8 quants, same half d),
// so the integer dot products are the ones MMVQ computes and the result matches it up to float summation order:
//
//   out[c][n] = sum_blocks ( d * sum_g sc_g * d8[c][g] * dot(q_g, u_g)  -  dmin * sum_g m_g * d8[c][g] * sum(u_g) )
//
// Built only with -DGGML_SYCL_XMX=ON (defines GGML_SYCL_XMX=1); runs only when GGML_SYCL_XMX_Q4K=1 and the
// runtime reports aspect::ext_intel_matrix with an s8 x s8 8x16x32 combination (the oneAPI 2026.1 runtime on
// the Arc Pro B70; the 2025.3 runtime reports none).
//
#ifndef GGML_SYCL_MMQ_XMX_Q4K_HPP
#define GGML_SYCL_MMQ_XMX_Q4K_HPP

#include "common.hpp"

// true when this backend was built with -DGGML_SYCL_XMX=ON
bool ggml_sycl_xmx_q4k_built();

// value of the GGML_SYCL_XMX_Q4K env switch (default 0), read once
bool ggml_sycl_xmx_q4k_env();

// "glue" kernels between XMX matmuls (gated GLU, wide rms_norm, f32 residual add) can run in the XMX kernels' 256-GRF
// mode so the GPU does not switch register modes around them (~6.5 us per switch on the B70). True when op (one of
// the bits below) should launch at 256-GRF for an activation of `nrows` columns/tokens; see GGML_SYCL_XMX_GLUE.
enum { GGML_SYCL_XMX_GLUE_GLU = 1, GGML_SYCL_XMX_GLUE_NORM = 2, GGML_SYCL_XMX_GLUE_ADD = 4 };
bool ggml_sycl_xmx_glue_grf256(int op, int64_t nrows);

// true when the device has s8 matrix hardware for this kernel; probed once per device and cached
bool ggml_sycl_xmx_q4k_device_ok(int device);

// true when src0 x src1 -> dst is a q4_K mul_mat this path can serve on ctx.device (type, shape, column window,
// env, device); no side effects. The caller must still make sure src0 is in the reorder layout.
bool ggml_sycl_xmx_q4k_can_use(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               const ggml_tensor * dst);

// which launch sequence serves the op (GGML_SYCL_XMX_Q4K_PATH, default 0): 0 = fused quantize + matmul launched
// directly by ggml_sycl_mul_mat_xmx_q4k; 1 = legacy, through ggml_sycl_op_mul_mat with the op below (A/B only)
int ggml_sycl_xmx_q4k_path();

// GGML_SYCL_XMX_Q4K_DIRECT=1 (default 0): 1..8 columns run the register-fed DPAS kernel of mmq-xmx-direct.cpp
// (bit-identical to the joint_matrix kernel, no SLM staging) instead of the joint_matrix kernel
bool ggml_sycl_xmx_q4k_direct_env();


// the whole mul_mat (1..16 columns) for a src0 already in the reorder layout: quantizes src1 straight into the
// kernel's operands and runs the matmul on ctx.stream(); two launches
void ggml_sycl_mul_mat_xmx_q4k(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst);

// legacy (GGML_SYCL_XMX_Q4K_PATH=1) per-device mul_mat op in the ggml_sycl_op_mul_mat<> driver's signature, 1..8
// columns; src1_ddq_i must be the SoA q8_1 activations (quantize_and_reorder_q8_1_soa) and src0 must be reordered
void ggml_sycl_op_mul_mat_xmx_q4k(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                  ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                  const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low,
                                  const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_row_size,
                                  const queue_ptr & stream);

#endif // GGML_SYCL_MMQ_XMX_Q4K_HPP
