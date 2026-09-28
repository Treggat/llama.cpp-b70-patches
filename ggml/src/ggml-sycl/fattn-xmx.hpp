//
// Flash-attention decode / speculative-verify on the Xe2 matrix units (joint_matrix / DPAS).
//
// Serves GGML_OP_FLASH_ATTN_EXT with a small query batch (2..8 tokens by default, GGML_SYCL_XMX_FA_MIN_COLS /
// _MAX_COLS) over a long f16 KV cache: head dim 256 (K and V), GQA ratio * tokens <= 48, f16 mask, no sinks,
// ALiBi or softcap, KV length a multiple of 64. One work-group of 8 sub-groups per (KV head, KV chunk) handles all
// query rows that share the KV head, so K and V are read from memory once per chunk regardless of the query batch
// (the TILE kernel pads GQA 6 to 8 columns and reads K/V twice for 5..8 tokens). S^T = K Q^T and O += P V run on
// DPAS with f32 accumulators; online softmax in f32 (exp2); split-KV partials are merged by a second kernel.
//
// Kernel: b70-secret-sauce/xmx-probe/xmx_fa.cpp (VAR 1, NSG 8, TPS 8, 256-GRF mode, QS 2 for 6/8 tokens), host
// harness removed. Built only with -DGGML_SYCL_XMX=ON; runs only when GGML_SYCL_XMX_FA=1 and the runtime reports
// aspect::ext_intel_matrix with an f16 x f16 -> f32 8x16x16 combination.
//
#ifndef GGML_SYCL_FATTN_XMX_HPP
#define GGML_SYCL_FATTN_XMX_HPP

#include "common.hpp"

// value of the GGML_SYCL_XMX_FA env switch (default 0), read once
bool ggml_sycl_fattn_xmx_env();

// value of the GGML_SYCL_XMX_FA_Q8 env switch (default 0, LOCAL): the same kernel structure for a q8_0 K/V cache
// (2D int8 block loads, dequantized to f16 in registers), 1..8 query tokens (GGML_SYCL_XMX_FA_Q8_MIN_COLS / _MAX_COLS)
bool ggml_sycl_fattn_xmx_q8_env();

// true when the device has f16 matrix hardware for this kernel; probed once per device and cached
bool ggml_sycl_fattn_xmx_device_ok(int device);

// true when this FLASH_ATTN_EXT node can run on the XMX kernel on this device (env, build, device, shapes, types)
bool ggml_sycl_fattn_xmx_can_use(int device, const ggml_tensor * dst);
bool ggml_sycl_fattn_xmx_can_use(ggml_backend_sycl_context & ctx, const ggml_tensor * dst);

void ggml_sycl_fattn_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_XMX_HPP
