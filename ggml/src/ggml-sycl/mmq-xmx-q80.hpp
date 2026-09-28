//
// q8_0 x q8_1 matmul on the Xe2 matrix units (joint_matrix / DPAS) for 1..8 activation columns (LOCAL)
// (GGML_SYCL_XMX_Q80=1, default off; GGML_SYCL_XMX_Q80_MIN_COLS / _MAX_COLS, default 1 / 8; weights under
// GGML_SYCL_XMX_Q80_MIN_MB = 64 stay on MMVQ; _KS split-K override; _CHECK=1 sampled double-precision check).
// Target: a large q8_0 LM head at speculative-verify column counts (248320 x 5120, 1.35 GB): MMVQ 2.50 ms at 1
// column, 2.95 ms at 8; this path 2.25 / 2.30 ms (~587 GB/s, flat in the column count).
//
// Same scheme as mmq-xmx-q4k.hpp: the activations are the DPAS A tile (8 columns x 32, the q8_1 quants MMVQ gets),
// the weights the B tile (32 x 16 rows, staged through a private SLM slice per sub-group), every accumulator lane
// owns one weight row. It reads the MMVQ q8_0 reorder layout in place (qs [N][K] | d [N*K/32]). Per 32-group the
// integer dot is MMVQ's (vec_dot_q8_0_q8_1: sum of q * u), scaled by d * d8; only the float summation order differs.
//
// Built only with -DGGML_SYCL_XMX=ON; the device check is the q4_K path's (s8 8x16x32 DPAS).
//
#ifndef GGML_SYCL_MMQ_XMX_Q80_HPP
#define GGML_SYCL_MMQ_XMX_Q80_HPP

#include "common.hpp"

// GGML_SYCL_XMX_WIDE=1 (LOCAL longdraft, default off), read once: the XMX q6_K / q8_0 matmuls and the XMX flash
// attention serve 9..16 columns (verify batches of up to 16 tokens); their 1..8-column kernels are unchanged
bool ggml_sycl_xmx_wide();

// LOCAL (wideverify): GGML_SYCL_XMX_WIDE=1 and GGML_SYCL_XMX_DIRECT_WIDE != 0 (default 1): the register-fed q4_K / q6_K
// kernels (their DIRECT switches) also serve 9..16 columns
bool ggml_sycl_xmx_direct_wide();

// true when src0 x src1 -> dst is a q8_0 mul_mat this path can serve on ctx.device (type, shape, column window,
// size, env, device); no side effects. The caller must still make sure src0 is in the reorder layout.
bool ggml_sycl_xmx_q80_can_use(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               const ggml_tensor * dst);

// the whole mul_mat for a src0 already in the reorder layout: quantizes src1 with MMVQ's q8_1 routine straight into
// the kernel's operands, then the matmul, both on ctx.stream() in the 256-register mode
void ggml_sycl_mul_mat_xmx_q80(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst);

#endif // GGML_SYCL_MMQ_XMX_Q80_HPP
