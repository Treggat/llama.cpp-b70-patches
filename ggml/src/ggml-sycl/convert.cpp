#include "convert.hpp"
#include <type_traits>
#include <vector>
#include "dequantize.hpp"
#include "presets.hpp"

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k,
                             const sycl::nd_item<3> &item_ct1) {
    const int64_t i = 2 * (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2));

    if (i >= k) {
        return;
    }

    const int64_t ib = i/qk; // block index
    const int64_t iqs = (i%qk)/qr; // quant index
    const int64_t iybs = i - i%qk; // y block start index
    const int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x();
    y[iybs + iqs + y_offset] = v.y();
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_sycl(const void *__restrict__ vx,
                                  dst_t *__restrict__ y, const int64_t k,
                                  dpct::queue_ptr stream) {
    const int64_t num_blocks = (k + 2*SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / (2*SYCL_DEQUANTIZE_BLOCK_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, num_blocks) *
                    sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                dequantize_block<qk, qr, dequantize_kernel>(vx, y, k, item_ct1);
            });
    }
}

template <typename dst_t>
static void dequantize_row_q2_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q2_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q2_K(vx, y, item_ct1);
                             });
    }

#endif
}

template <typename dst_t>
static void dequantize_row_q2_K_sycl_reorder(const void *vx, dst_t *y, const int64_t k,
                                             dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q2_K_reorder(vx, y, item_ct1, nb);
        });
}

template <typename dst_t>
static void dequantize_row_q3_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q3_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q3_K(vx, y, item_ct1);
                             });
    }
#endif
}

template <typename dst_t>
static void dequantize_row_q3_K_sycl_reorder(const void *vx, dst_t *y, const int64_t k,
                                             dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q3_K_reorder(vx, y, item_ct1, nb);
        });
}

template <typename dst_t>
static void dequantize_row_q4_0_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb32 = k / 32;
    const int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q4_0(vx, y, nb32, item_ct1);
                             });
    }
}

template <typename dst_t>
static void dequantize_row_q4_0_sycl_reorder(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    int constexpr WARP_K = WARP_SIZE * QK4_0;
    const int n_warp = (k + WARP_K - 1) / WARP_K;
    GGML_ASSERT(k % 2 == 0);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) *
        sycl::range<3>(1, 1, WARP_SIZE),
        sycl::range<3>(1, 1, WARP_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            dequantize_block_q4_0_reorder(vx, y, k, item_ct1);
        });

}

template <typename dst_t>
static void dequantize_row_q8_0_sycl_reorder(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    int constexpr WARP_K = WARP_SIZE * QK8_0;
    const int n_warp = (k + WARP_K - 1) / WARP_K;
    GGML_ASSERT(k % QK8_0 == 0);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) *
        sycl::range<3>(1, 1, WARP_SIZE),
        sycl::range<3>(1, 1, WARP_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            dequantize_block_q8_0_reorder(vx, y, k, item_ct1);
        });

}

template <typename dst_t>
static void dequantize_row_q4_1_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb32 = k / 32;
    const int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q4_1(vx, y, nb32, item_ct1);
                             });
    }
}


template <typename dst_t>
static void dequantize_row_q4_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(12), cgh);
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q4_K(vx, y, get_pointer(scale_local_acc), item_ct1);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_q4_K_sycl_reorder(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    const size_t  local_size  = 32;
    const size_t  global_size = nb * local_size;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(12), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
                         [=](sycl::nd_item<1> item_ct1) {
                             dequantize_block_q4_K_reorder(vx, y, get_pointer(scale_local_acc), item_ct1, nb);
                         });
    });
}

template <typename dst_t>
static void dequantize_row_q5_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q5_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q5_K(vx, y, item_ct1);
                             });
    }

#endif
}

template <typename dst_t>
static void dequantize_row_q5_K_sycl_reorder(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(K_SCALE_SIZE), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
            [=](sycl::nd_item<3> item_ct1) {
                dequantize_block_q5_K_reorder(vx, y, get_pointer(scale_local_acc), item_ct1, nb);
            });
    });
}

template <typename dst_t>
static void dequantize_row_q6_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q6_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q6_K(vx, y, item_ct1);
                             });
    }

#endif
}

template <typename dst_t>
static void dequantize_row_q6_K_sycl_reorder(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) { dequantize_block_q6_K_reorder(vx, y, item_ct1, nb); });
}

// LOCAL (prefillx) GGML_SYCL_Q6K_DQ_FAST=1 (default 0): q6_K (reorder layout) -> f16 for the prefill GEMM, 8
// consecutive outputs per work-item (8-byte ql / qh loads, one 16-byte store) and 8 blocks per 256-item work-group,
// instead of 4 strided 2-byte stores per item in 64-item groups. Every output is the same expression as
// dequantize_block_q6_K_reorder (d * sc * (q - 32) with the same operand types), so the result is bit-identical.
static void dequantize_row_q6_K_sycl_reorder_fast(const void * vx, sycl::half * yy, const int64_t k,
                                                  dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    constexpr int64_t items_per_block = 32;   // 2 halves (ip) x 4 quarters x 4 groups of 8
    constexpr int64_t wg              = 256;
    const int64_t     n_items         = nb * items_per_block;
    stream->parallel_for(sycl::nd_range<1>(((n_items + wg - 1) / wg) * wg, wg), [=](sycl::nd_item<1> it) {
        const int64_t gid = it.get_global_id(0);
        if (gid >= n_items) {
            return;
        }
        const int64_t ib  = gid / items_per_block;
        const int     r   = (int) (gid % items_per_block);
        const int     ip  = r / 16;          // 0..1: which 128-value half of the block
        const int     qq  = (r / 4) % 4;     // 0..3: which 32-value quarter of the half
        const int     il0 = (r % 4) * 8;     // 0, 8, 16, 24: first of 8 consecutive values

        const uint8_t *   base_ptr   = static_cast<const uint8_t *>(vx);
        const uint8_t *   ql_ptr     = base_ptr + ib * (QK_K / 2);
        const uint8_t *   qh_ptr     = base_ptr + (QK_K / 2) * nb + (QK_K / 4) * ib;
        const uint8_t *   scales_ptr = base_ptr + (QK_K / 2) * nb + (QK_K / 4) * nb + (QK_K / 16) * ib;
        const ggml_half * d          = (const ggml_half *) (base_ptr + ((QK_K / 2) + (QK_K / 4) + (QK_K / 16)) * nb) + ib;

        const int is = 8 * ip + il0 / 16;
        const int8_t * sc = reinterpret_cast<const int8_t *>(scales_ptr + is);

        const sycl::vec<uint8_t, 8> qlv =
            *reinterpret_cast<const sycl::vec<uint8_t, 8> *>(ql_ptr + 64 * ip + 32 * (qq & 1) + il0);
        const sycl::vec<uint8_t, 8> qhv = *reinterpret_cast<const sycl::vec<uint8_t, 8> *>(qh_ptr + 32 * ip + il0);

        sycl::vec<sycl::half, 8> out;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const uint8_t ql = qlv[e];
            const uint8_t qh = qhv[e];
            sycl::half v;
            if (qq == 0) {
                v = *d * sc[0] * ((int8_t) ((ql & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
            } else if (qq == 1) {
                v = *d * sc[2] * ((int8_t) ((ql & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
            } else if (qq == 2) {
                v = *d * sc[4] * ((int8_t) ((ql >> 4) | (((qh >> 4) & 3) << 4)) - 32);
            } else {
                v = *d * sc[6] * ((int8_t) ((ql >> 4) | (((qh >> 6) & 3) << 4)) - 32);
            }
            out[e] = v;
        }
        *reinterpret_cast<sycl::vec<sycl::half, 8> *>(yy + ib * QK_K + 128 * ip + 32 * qq + il0) = out;
    });
}

template <typename dst_t>
static void dequantize_row_q6_K_sycl_reorder_sel(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    static const bool fast = [] {
        const char * e = getenv("GGML_SYCL_Q6K_DQ_FAST");
        return e && atoi(e) != 0;
    }();
    if constexpr (std::is_same_v<dst_t, sycl::half>) {
        if (fast && (k % QK_K) == 0 && ((uintptr_t) vx) % 8 == 0 && ((uintptr_t) y) % 16 == 0) {
            dequantize_row_q6_K_sycl_reorder_fast(vx, y, k, stream);
            // GGML_SYCL_Q6K_DQ_FAST=2 (test): also run the reference kernel and compare bit for bit (syncs)
            static const bool verify = [] { const char * e = getenv("GGML_SYCL_Q6K_DQ_FAST"); return e && atoi(e) == 2; }();
            if (verify) {
                sycl::half * ref = sycl::malloc_device<sycl::half>(k, *stream);
                dequantize_row_q6_K_sycl_reorder(vx, ref, k, stream);
                std::vector<uint16_t> a(k), b(k);
                stream->memcpy(a.data(), y, k * sizeof(uint16_t));
                stream->memcpy(b.data(), ref, k * sizeof(uint16_t)).wait();
                sycl::free(ref, *stream);
                int64_t bad = 0;
                for (int64_t i = 0; i < k; ++i) {
                    bad += a[i] != b[i];
                }
                static int n_log = 0;
                if (bad != 0 || n_log++ < 4) {
                    fprintf(stderr, "GGML_SYCL_Q6K_DQ_FAST=2: %lld values, %lld differ\n", (long long) k, (long long) bad);
                }
                GGML_ASSERT(bad == 0);
            }
            return;
        }
    }
    dequantize_row_q6_K_sycl_reorder(vx, y, k, stream);
}


template <typename dst_t>
static void dequantize_row_iq1_s_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq1_s(
                                     vx, y, item_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq1_m_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq1_m(
                                     vx, y, item_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_xxs_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_xxs(
                                     vx, y, item_ct1, iq2xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_xs_sycl(const void *vx, dst_t *y, const int64_t k,
                                       dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_xs(
                                     vx, y, item_ct1, iq2xs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_s_sycl(const void *vx, dst_t *y, const int64_t k,
                                      dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_s(vx, y, item_ct1);
                             });
        });
    }
}


template <typename dst_t>
static void dequantize_row_iq3_xxs_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq3_xxs(
                                     vx, y, item_ct1, iq3xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq3_s_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq3_s(
                                     vx, y, item_ct1, kmask_iq2xs, iq3s_grid);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq4_xs_sycl(const void *vx, dst_t *y, const int64_t k,
                                       dpct::queue_ptr stream) {
    const int64_t nb = (k + QK_K - 1) / QK_K;
#if QK_K == 64
    dequantize_row_iq4_nl_sycl(vx, y, k, stream);
#else
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_block_iq4_xs(vx, y, item_ct1);
                      });
            });
      }
#endif
}

template <typename dst_t>
static void dequantize_row_iq4_nl_sycl(const void *vx, dst_t *y, const int64_t k,
                                       dpct::queue_ptr stream) {
    const int64_t nb = (k + QK_K - 1) / QK_K;
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_block_iq4_nl(vx, y, item_ct1);
                      });
            });
      }
}

template <typename dst_t>
static void dequantize_row_mxfp4_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_mxfp4(vx, y, item_ct1);
        });
}

template <typename dst_t>
static void dequantize_row_nvfp4_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nb = k / QK_NVFP4;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            dequantize_block_nvfp4(vx, y, k);
        });
}


template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_nc(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i00 = 2 * (int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2));

    if (i00 >= ne00) {
        return;
    }

    const int64_t i01 = item_ct1.get_group(1);
    const int64_t i02 = item_ct1.get_group(0) % ne02;
    const int64_t i03 = item_ct1.get_group(0) / ne02;

    const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

    const int64_t ib = ibx0 + i00/qk; // block index
    const int64_t iqs = (i00%qk)/qr; // quant index
    const int64_t iybs = i00 - i00%qk; // y block start index
    const int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    #ifdef GGML_SYCL_F16
        sycl::half2 v;
    #else
        sycl::float2 v;
    #endif

    dequantize_kernel(vx, ib, iqs, v);

    const int64_t iy0 = ((i03*ne02 + i02)*ne01 + i01)*ne00 + iybs + iqs;
    y[iy0 + 0]        = ggml_sycl_cast<dst_t>(v.x());
    y[iy0 + y_offset] = ggml_sycl_cast<dst_t>(v.y());
}


template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_nc_sycl(const void *    vx,
                                  dst_t *         y,
                                  const int64_t   ne00,
                                  const int64_t   ne01,
                                  const int64_t   ne02,
                                  const int64_t   ne03,
                                  const int64_t   s01,
                                  const int64_t   s02,
                                  const int64_t   s03,
                                  dpct::queue_ptr stream) {
    const dpct::dim3 num_blocks((ne00 + 2 * SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / (2 * SYCL_DEQUANTIZE_BLOCK_SIZE), ne01,
                                ne02 * ne03);
    stream->parallel_for(sycl::nd_range<3>(num_blocks * sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) {
                             GGML_UNUSED(item_ct1);
                             dequantize_block_nc<qk, qr, dequantize_kernel>(vx, y, ne00, ne01, ne02, s01, s02, s03);
                         });
}
template <typename src_t, typename dst_t>
static void convert_unary_nc(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t ne00, const int64_t ne01,
                          const int64_t ne02, const int64_t s01, const int64_t s02, const int64_t s03,
                          const sycl::nd_item<3> & item_ct1) {

    const int64_t work_group_size = item_ct1.get_local_range(2);
    const int64_t global_id       = item_ct1.get_local_id(2) + work_group_size * item_ct1.get_group(2);

    const int64_t i01 = item_ct1.get_group(1);
    const int64_t i02 = item_ct1.get_group(0) % ne02;
    const int64_t i03 = item_ct1.get_group(0) / ne02;

    // make each work-item deal with more elements since sycl global range can not exceed max int
    const src_t * x = static_cast<const src_t *>(vx);
    const int64_t ix = i03 * s03 + i02 * s02 + i01 * s01;
    const int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00;

#pragma unroll
    for (int64_t i00 = global_id; i00 < ne00; i00 += work_group_size * item_ct1.get_group_range(2)) {
        y[iy + i00] = static_cast<dst_t>(x[ix + i00]);
    }
}

template <typename src_t, typename dst_t>
static void convert_unary_nc_sycl(const void * __restrict__ vx, dst_t * __restrict__ y,
                                  const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
                                  const int64_t s01, const int64_t s02, const int64_t s03, dpct::queue_ptr queue) {
    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    sycl::range<3> global_size(ne02 * ne03, ne01, ceil_div(ne00, SYCL_DEQUANTIZE_BLOCK_SIZE));

    // decrease global range when it exceeds the max int
    // TODO: Downsample logic is separated from the kernel, a rewrite is desirable
    int64_t        downsized_workgroup = downsample_sycl_global_range(global_size[0], SYCL_DEQUANTIZE_BLOCK_SIZE);
    sycl::range<3> workgroup_size(1, 1, downsized_workgroup);

    queue->parallel_for(sycl::nd_range<3>(global_size * workgroup_size, workgroup_size), [=](sycl::nd_item<3> item_ct1) {
        convert_unary_nc<src_t>(vx, y, ne00, ne01, ne02, s01, s02, s03, item_ct1);
    });
}

template <typename src_t, typename dst_t>
static void convert_unary_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr queue) {
    convert_unary_nc_sycl<src_t>(vx, y, k, 1, 1, 1, k, k, k, queue);
}


to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_sycl<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            if (dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q4_0_sycl_reorder;
            } else {
                return dequantize_block_sycl<QK4_0, QR4_0, dequantize_q4_0>;
            }
        case GGML_TYPE_Q4_1:
            return dequantize_block_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            if (dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q8_0_sycl_reorder;
            } else {
                return dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0>;
            }
        case GGML_TYPE_Q2_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q2_K_sycl_reorder;
            } else {
                return dequantize_row_q2_K_sycl;
            }
        case GGML_TYPE_Q3_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q3_K_sycl_reorder;
            } else {
                return dequantize_row_q3_K_sycl;
            }
        case GGML_TYPE_Q4_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q4_K_sycl_reorder;
            } else {
                return dequantize_row_q4_K_sycl;
            }
        case GGML_TYPE_Q5_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q5_K_sycl_reorder;
            } else {
                return dequantize_row_q5_K_sycl;
            }
        case GGML_TYPE_Q6_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q6_K_sycl_reorder_sel;
            } else {
                return dequantize_row_q6_K_sycl;
            }
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_sycl;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_sycl;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_sycl;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_sycl;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_sycl;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_sycl;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_sycl;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_sycl;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_sycl;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_sycl;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_sycl;
        case GGML_TYPE_F32:
            return convert_unary_sycl<float>;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            return convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        default:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
            return nullptr;
    }
}

to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor *dst) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_sycl<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            if (dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q4_0_sycl_reorder;
            } else {
                return dequantize_row_q4_0_sycl;
            }
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_sycl;
        case GGML_TYPE_Q5_0:
            return dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            if (dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q8_0_sycl_reorder;
            } else {
                return dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0>;
            }
        case GGML_TYPE_Q2_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q2_K_sycl_reorder;
            } else {
                return dequantize_row_q2_K_sycl;
            }
        case GGML_TYPE_Q3_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q3_K_sycl_reorder;
            } else {
                return dequantize_row_q3_K_sycl;
            }
        case GGML_TYPE_Q4_K:
            if (dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q4_K_sycl_reorder;
            } else {
                return dequantize_row_q4_K_sycl;
            }
        case GGML_TYPE_Q5_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q5_K_sycl_reorder;
            } else {
                return dequantize_row_q5_K_sycl;
            }
        case GGML_TYPE_Q6_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q6_K_sycl_reorder;
            } else {
                return dequantize_row_q6_K_sycl;
            }
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_sycl;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_sycl;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_sycl;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_sycl;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_sycl;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_sycl;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_sycl;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_sycl;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_sycl;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_sycl;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_sycl;
        case GGML_TYPE_F16:
            return convert_unary_sycl<sycl::half>;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            return convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        default:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
            return nullptr;
    }
}


#ifdef GGML_SYCL_HAS_BF16
to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * /*dst*/) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_sycl<float>;
        case GGML_TYPE_F16:
            return convert_unary_sycl<sycl::half>;
        case GGML_TYPE_BF16:
            return convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
        default:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
            return nullptr;
    }
}
#endif

to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_nc_sycl<float>;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            return convert_unary_nc_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        case GGML_TYPE_Q1_0:
            return dequantize_block_nc_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_nc_sycl<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_nc_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_nc_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_nc_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_nc_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        default:
            return nullptr;
    }
}
