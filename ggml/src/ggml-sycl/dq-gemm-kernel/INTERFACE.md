# dq_gemm_q4k interface (for dq-gemm-integrate)

Files: `dq_gemm_q4k.hpp` (declarations) + `dq_gemm_q4k.cpp` (kernel + launcher, SYCL only, no ggml includes).
Compile `dq_gemm_q4k.cpp` as a normal SYCL source (it needs the 2026.1 toolchain/runtime: joint_matrix).
Tunables are `DQ_*` macros at the top of the .cpp; the defaults are the best measured variant.

```cpp
bool        dq_gemm_q4k_supported(int64_t M, int64_t N, int64_t K);   // K % 256 == 0, M % 128 == 0, N >= 1
sycl::event dq_gemm_q4k(sycl::queue & q, const void * w, const sycl::half * act, float * dst,
                        int64_t M, int64_t N, int64_t K, int64_t ldd);
```
- M = weight rows (ne01), N = token columns (ne11, any N >= 1; a ragged last token tile is handled inside the same single launch),
  K = ne00.
- `w`: ggml-sycl q4_K REORDER layout of the whole tensor, nb = M*K/256, block i = row*(K/256)+kb:
  qs at w+0 (nb*128 B), scales at w+nb*128 (nb*12 B), dm at w+nb*140 (nb x half2 d, dmin).
- `act`: f16, act[n*K + k] (contiguous rows, stride K). 16-byte aligned (64-byte preferred: 2D block loads).
- `dst`: f32, dst[n*ldd + m], overwritten. ldd >= M, ldd*4 must be a multiple of 16 bytes (2D block stores).
- Asynchronous: submits exactly 1 kernel for any N, no waits, no allocations; returns its event.
  Token tile = 256 (the last partial tile runs in the same launch; activation rows >= N are never read).
- Numerics: weights dequantized exactly like ggml-sycl's dequantize_block_q4_K (f32 d*sc*q - dmin*m, fma, then
  rounded to f16 once), f16 x f16 products with f32 accumulation over all of K.
- Pointer type: `w` is `const void *` (the tensor data); the kernel reads 16-byte vectors of qs, dwords of scales/dm.

Status: see the header of `dq_gemm_q4k.cpp` and the report; the harness `dq_gemm.cpp` includes the .cpp directly.
