//
// q6_K x q8_1 matmul on the Xe2 matrix units (joint_matrix / DPAS) for up to 8 activation columns
// (3..8 by default, GGML_SYCL_XMX_Q6K_MIN_COLS / _MAX_COLS; weights under GGML_SYCL_XMX_Q6K_MIN_MB = 8 stay on MMVQ,
// and weights under GGML_SYCL_XMX_Q6K_BIG_MB = 64 below GGML_SYCL_XMX_Q6K_SMALL_MIN_COLS = 5 columns). Sibling of
// mmq-xmx-q4k.hpp.
//
// Same scheme as the q4_K path: the activations are the DPAS A tile (8 columns x 32), the weights the B tile
// (32 x 16 rows), every accumulator lane owns one weight row and its scales in registers. It reads the MMVQ q6_K
// reorder (SoA) layout in place (ql | qh | scales | d, no repack) and the SoA q8_1 activations MMVQ gets, so the
// integer dot products per 16-element scale group are the ones MMVQ computes (vec_dot_q6_K_q8_1_impl_mmvq) and
// the result matches it up to float summation order:
//
//   out[c][n] = sum_blocks d * sum_g32 d8[c][g32] * ( sc[2 g32] * dot(q - 32, u)_lo16 + sc[2 g32 + 1] * dot(q - 32, u)_hi16 )
//
// The 16-element scale groups are finer than the 32-deep s8 DPAS, so each 32-group runs two DPAS on the same
// weight tile with two activation tiles, one with the upper and one with the lower 16 quants zeroed.
//
// Built only with -DGGML_SYCL_XMX=ON; runs only when GGML_SYCL_XMX_Q6K=1 and the device passes the same matrix
// capability check as the q4_K path (ggml_sycl_xmx_q4k_device_ok).
//
#ifndef GGML_SYCL_MMQ_XMX_Q6K_HPP
#define GGML_SYCL_MMQ_XMX_Q6K_HPP

#include "common.hpp"

// value of the GGML_SYCL_XMX_Q6K env switch (default 0), read once
bool ggml_sycl_xmx_q6k_env();

// true when src0 x src1 -> dst is a q6_K mul_mat this path can serve on ctx.device (type, shape, column window,
// env, device); no side effects. The caller must still make sure src0 is in the reorder layout.
bool ggml_sycl_xmx_q6k_can_use(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               const ggml_tensor * dst);

// which launch sequence serves the op (GGML_SYCL_XMX_Q6K_PATH, default 0): 0 = fused quantize + matmul launched
// directly by ggml_sycl_mul_mat_xmx_q6k (both 256-GRF, no q8_1 SoA buffer); 1 = legacy, through ggml_sycl_op_mul_mat
// with the op below (A/B only)
int ggml_sycl_xmx_q6k_path();

// GGML_SYCL_XMX_Q6K_DIRECT=1 (default 0): the register-fed DPAS kernel of mmq-xmx-direct.cpp instead of the
// joint_matrix kernel (bit-identical at the same split-K; GGML_SYCL_XMX_Q6K_DIRECT_KS: 0 = the joint_matrix kernel's
// split-K, n > 0 = n, default: its own table)
bool ggml_sycl_xmx_q6k_direct_env();


// the whole mul_mat for a src0 already in the reorder layout: quantizes src1 straight into the kernel's operands with
// MMVQ's own q8_1 routine and runs the matmul on ctx.stream(); two launches
void ggml_sycl_mul_mat_xmx_q6k(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst);

// legacy (GGML_SYCL_XMX_Q6K_PATH=1) per-device mul_mat op in the ggml_sycl_op_mul_mat<> driver's signature;
// src1_ddq_i must be the SoA q8_1 activations (quantize_and_reorder_q8_1_soa) and src0 must be reordered
void ggml_sycl_op_mul_mat_xmx_q6k(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                  ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                  const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low,
                                  const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_row_size,
                                  const queue_ptr & stream);

#endif // GGML_SYCL_MMQ_XMX_Q6K_HPP
