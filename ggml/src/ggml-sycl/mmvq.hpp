//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_MMVQ_HPP
#define GGML_SYCL_MMVQ_HPP

#include "common.hpp"


void ggml_sycl_op_mul_mat_vec_q(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

// Requires standard (non-reorder) block layout for src0.
// Returns false if src0_type isn't handled; caller should fall back.
bool ggml_sycl_mul_mat_vec_q_id(
    enum ggml_type     src0_type,
    const void *       vx_base,             // start of stacked expert weights
    const void *       vy,                  // pre-quantized src1 (Q8_1)
    const int32_t *    ids_dev,             // device-side int32, length n_experts_used
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    size_t             expert_weight_stride, // bytes between experts in vx_base
    size_t             dst_row_stride,       // bytes between dst rows
    size_t             src1_row_stride,      // 0 = shared src1, else per-expert stride in bytes
    dpct::queue_ptr    stream);

// Reorder (SoA) variant of the fused MoE expert GEMV.
// vx_base: each expert slice (stride expert_weight_stride == src0->nb[2]) is a self-contained reorder/SoA layout.
// vy: src1 quantized with quantize_and_reorder_q8_1_soa (per-row SoA). Returns false if src0_type isn't handled.
bool ggml_sycl_mul_mat_vec_q_id_reorder(
    enum ggml_type     src0_type,
    const void *       vx_base,
    const void *       vy,
    const int32_t *    ids_dev,
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    dpct::queue_ptr    stream);

// Fused dense-FFN GEMV: writes glu(gate . y, up . y) instead of the two mat-vec results.
// vx / vgate must share shape, stride and reorder layout. Returns false if unhandled.
bool ggml_sycl_mul_mat_vec_q_glu_reorder(
    enum ggml_type     src0_type,
    enum ggml_glu_op   glu_op,
    const void *       vx,
    const void *       vgate,
    const void *       vy,
    float *            dst,
    int                ncols,                // K, shared by both weights
    int                nrows,                // output rows, i.e. weight ne[1]
    int                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    int                stride_col_y_bytes,   // bytes between activation columns in vy
    int                stride_col_dst,       // floats between output columns in dst
    dpct::queue_ptr    stream);


// Fused dense-FFN GEMV + GLU over the standard (non-reorder) layout; the gate and up
// weights may carry different block types (q5_K / iq4_xs, mixed included).
// vy: src1 quantized with plain quantize_q8_1 (padded rows). stride_col_y is in
// block_q8_1 units. Returns false if the pair or batch is unhandled; caller falls back.
bool ggml_sycl_mul_mat_vec_q_glu_plain(
    enum ggml_type     gate_type,
    enum ggml_type     up_type,
    enum ggml_glu_op   glu_op,
    const void *       vgate,
    const void *       vup,
    const void *       vy,
    float *            dst,
    int                ncols,                // K, shared by both weights
    int                nrows,                // output rows, i.e. weight ne[1]
    int                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    int                stride_col_y,         // block_q8_1 units between activation columns
    int                stride_col_dst,       // floats between output columns in dst
    dpct::queue_ptr    stream);

// LOCAL: split-K reorder MMVQ for weights with few rows (GGML_SYCL_SMALLROW=1, default off). The row-per-sub-group
// MMVQ kernels leave the device mostly idle on e.g. 48 x 5120 (ssm_alpha / ssm_beta) or 1024 x 5120 (attn_k / attn_v):
// each row is one serial chain over K. Here a work-group of `ks` sub-groups shares one row, each sub-group takes every
// ks-th q-block, and the partial sums are added in a fixed order through local memory (deterministic).
// Optional epilogue, applied to the finished dot product (the float ops the unfused graph nodes perform):
//   op 1: sigmoid(v)                          (qwen35 beta)
//   op 2: softplus(v + b[row]) * a[row]       (qwen35 alpha -> ADD ssm_dt -> SOFTPLUS -> MUL ssm_a)
struct ggml_sycl_mmvq_epilogue {
    int           op = 0;
    const float * b  = nullptr;
    const float * a  = nullptr;
};

bool ggml_sycl_smallrow_enabled();
// q4_K / q6_K in the reorder layout, nrows <= GGML_SYCL_SMALLROW_MAX_ROWS_Q4K (256) / _Q6K (1024), columns in
// [GGML_SYCL_SMALLROW_MIN_COLS (2), 8]
bool ggml_sycl_smallrow_can_use(enum ggml_type type, int64_t nrows, int64_t ncols_dst);
// same without the column window (fused epilogues also serve 1 column)
bool ggml_sycl_smallrow_shape_ok(enum ggml_type type, int64_t nrows, int64_t ncols, int64_t ncols_dst);

bool ggml_sycl_mul_mat_vec_q_reorder_splitk(
    enum ggml_type                  src0_type,
    const void *                    vx,
    const void *                    vy,                  // quantize_and_reorder_q8_1_soa activations
    float *                         dst,
    int                             ncols,               // K
    int                             nrows,
    int                             ncols_dst,           // 1..8
    int                             stride_col_y_bytes,
    int                             stride_col_dst,
    const ggml_sycl_mmvq_epilogue & ep,
    dpct::queue_ptr                 stream);

// two same-type, same-shape weights over the same activations in one launch (qwen35 ssm_alpha + ssm_beta)
bool ggml_sycl_mul_mat_vec_q_reorder_splitk2(
    enum ggml_type                  src0_type,
    const void *                    vx0,
    float *                         dst0,
    const ggml_sycl_mmvq_epilogue & ep0,
    const void *                    vx1,
    float *                         dst1,
    const ggml_sycl_mmvq_epilogue & ep1,
    const void *                    vy,
    int                             ncols,
    int                             nrows,
    int                             ncols_dst,
    int                             stride_col_y_bytes,
    int                             stride_col_dst,
    dpct::queue_ptr                 stream);

#endif // GGML_SYCL_MMVQ_HPP
