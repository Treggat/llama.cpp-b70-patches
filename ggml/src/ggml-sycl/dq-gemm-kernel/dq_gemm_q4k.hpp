// dq_gemm_q4k: q4_K x f16 -> f32 GEMM for llama.cpp PREFILL on Intel Xe2 (Arc Pro B70), SYCL.
// The q4_K weights are dequantized to f16 INSIDE the kernel (once per work-group tile, into SLM) and fed to the f16
// DPAS (joint_matrix 8x16x16) with f32 accumulation over the whole K loop. Numerics = "dequantize to f16, then f16
// GEMM with f32 accumulation" (the current ggml-sycl prefill path); only the summation order differs.
//
// Naming follows ggml / the integration side: M = weight rows (ne01), N = token columns (ne11), K = ne00.
//
//   w   : ggml-sycl q4_K REORDER layout of the whole tensor, nb = M*K/256 blocks, block i = row*(K/256) + kb:
//           qs     at w + 0          (nb * 128 B, block i's 128 B contiguous, same nibble order as block_q4_K.qs)
//           scales at w + nb*128     (nb * 12 B, same packing as block_q4_K.scales)
//           dm     at w + nb*140     (nb x half2: d, dmin)
//   act : f16, N rows of K contiguous: act[n*K + k]
//   dst : f32, dst[n*ldd + m], ldd >= M. Overwritten (beta = 0).
// Asynchronous: submits ONE kernel on q (a ragged last token tile is handled inside it);
// no host sync, no allocations.
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>

// true when the kernel handles this shape: K % 256 == 0, M % 128 == 0, N >= 1, 16-byte-aligned pointers and ldd % 4 == 0
bool dq_gemm_q4k_supported(int64_t M, int64_t N, int64_t K);

sycl::event dq_gemm_q4k(sycl::queue & q, const void * w, const sycl::half * act, float * dst,
                        int64_t M, int64_t N, int64_t K, int64_t ldd);
