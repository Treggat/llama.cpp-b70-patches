#ifndef GGML_SYCL_MMQ_XMX_DIRECT_HPP
#define GGML_SYCL_MMQ_XMX_DIRECT_HPP

// Register-fed DPAS kernels for the XMX q4_K / q6_K paths (mmq-xmx-direct.cpp). Built as their own small shared
// library (ggml-sycl-xmxd): the kernels call the OpenCL extension builtin intel_sub_group_i8_i8_matrix_mad_k32
// (cl_intel_subgroup_matrix_multiply_accumulate), which IGC resolves when it compiles the image. libggml-sycl is
// device-linked with -fsycl-allow-device-image-dependencies (from the oneAPI CMake config), under which the SYCL
// runtime treats every undefined device function as an import from another image and refuses to launch; a separate
// library keeps the builtin a plain IGC builtin.

#include <sycl/sycl.hpp>
#include <cstdint>

// q4_K, 1..8 columns: qs [N][K/2], sc [N*KB][12], dm [N*KB] half2 (the reorder layout); xa [G][16][32] int8
// (group-major activation quants), d8 [16][G] half, us [16][G] int32; out column i of row r at out[i*ldd + r]
sycl::event ggml_sycl_xmx_q4k_direct_launch(const uint8_t * qs, const uint8_t * sc, const uint32_t * dm, const int8_t * xa,
                                            const sycl::half * d8, const int32_t * us, float * out, int N, int K, int M,
                                            int64_t ldd, int ks, sycl::queue & q);

// q6_K, 1..8 columns: ql [N*KB][128], qh [N*KB][64], sc [N*KB][16], dd [N*KB] half; xa [G][2][8][32] int8,
// d8 [G][8] half
sycl::event ggml_sycl_xmx_q6k_direct_launch(const uint8_t * ql, const uint8_t * qh, const int8_t * sc, const uint16_t * dd,
                                            const int8_t * xa, const sycl::half * d8, float * out, int N, int K, int M,
                                            int64_t ldd, int ks, sycl::queue & q);

// LOCAL (wideverify): 9..16 columns (two A tiles on every register-built B operand). q4_K: the same operand layouts as
// the 1..8 kernel (xa [G][16][32] already holds 16 columns). q6_K: xa [G][2][16][32] int8 (half-group tiles of 16
// columns each), d8 [G][16] half. Per column the float operations and their order are the two-tile joint_matrix
// kernels' (xmx_q4k_launch<16> / xmx_q6k_launch_wide<16>), so the outputs are bit-identical to them at the same ks.
sycl::event ggml_sycl_xmx_q4k_direct16_launch(const uint8_t * qs, const uint8_t * sc, const uint32_t * dm, const int8_t * xa,
                                              const sycl::half * d8, const int32_t * us, float * out, int N, int K, int M,
                                              int64_t ldd, int ks, sycl::queue & q);
sycl::event ggml_sycl_xmx_q6k_direct16_launch(const uint8_t * ql, const uint8_t * qh, const int8_t * sc, const uint16_t * dd,
                                              const int8_t * xa, const sycl::half * d8, float * out, int N, int K, int M,
                                              int64_t ldd, int ks, sycl::queue & q);

// LOCAL (verifystep): two q4_K weights of the same shape (gate, up) against one activation, 1..8 columns, one launch;
// bit-identical gate / up values to two ggml_sycl_xmx_q4k_direct_launch calls at the same ks. glu: out = silu(gate) * up
// into outG (outU unused), else gate -> outG, up -> outU.
sycl::event ggml_sycl_xmx_q4k_direct_pair_launch(const uint8_t * gqs, const uint8_t * gsc, const uint32_t * gdm,
                                                 const uint8_t * uqs, const uint8_t * usc, const uint32_t * udm,
                                                 const int8_t * xa, const sycl::half * d8, const int32_t * us, float * outG,
                                                 float * outU, int N, int K, int M, int64_t ldd, int ks, bool glu,
                                                 sycl::queue & q);

#endif // GGML_SYCL_MMQ_XMX_DIRECT_HPP
