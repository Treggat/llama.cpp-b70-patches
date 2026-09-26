//
// oneDNN int4 weight-decompression matmul for q4_K weights at small column counts.
//
// The MMVQ kernels run q4_K at memory bandwidth for one column but their cost grows with every extra column
// (about 2x at 6 columns on the Arc Pro B70), which is what a speculative-decoding verify pass pays. oneDNN's
// matmul with u4 weights + f16 group scales is flat in the column count at the same bandwidth. This path repacks
// a q4_K tensor once, in place, into the layout that primitive wants and keeps the q4_K math exact:
//
//   q4_K:  w = d*sc[g] * q - dmin*mn[g]            (g = 32-element sub-block)
//   here:  y = X * (S[g] .* Q)^T  -  Xg * MN^T     (S = d*sc as f16 per group, MN = dmin*mn decoded on the fly)
//
// Layout after the repack (same byte size as the tensor, like the MMVQ reorder):
//   [ u4 quants, K nibbles per row in k order ][ packed 6-bit scales, 12 B per block ][ dm, 4 B per block ]
// plus a side buffer of f16 scales [K/32][N] owned by the tensor's extra.
//
#ifndef GGML_SYCL_MMQ_DNN_U4_HPP
#define GGML_SYCL_MMQ_DNN_U4_HPP

#include "common.hpp"

// true when this backend was built with oneDNN and the env switch is on
bool ggml_sycl_dnn_u4_enabled();

// true when src0 is a q4_K weight this path can serve (type, shape, buffer); does not repack
bool ggml_sycl_dnn_u4_can_use(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                              const ggml_tensor * dst);

// true once src0's bytes are in the u4 layout: from then on only this path may read them
bool ggml_sycl_dnn_u4_is_repacked(const ggml_tensor * src0);

// repack src0 once (no-op when already done); false when it cannot be done, in which case the caller must not
// use the u4 path for this tensor
bool ggml_sycl_dnn_u4_prepare(ggml_backend_sycl_context & ctx, const ggml_tensor * src0);

// called at the start of every graph compute: drops the per-graph activation cache
void ggml_sycl_dnn_u4_graph_begin(const queue_ptr & stream);

// per-device mul_mat op in the ggml_sycl_op_mul_mat<> driver's signature
void ggml_sycl_op_mul_mat_dnn_u4(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                 ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                 const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low,
                                 const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_row_size,
                                 const queue_ptr & stream);

#endif // GGML_SYCL_MMQ_DNN_U4_HPP
