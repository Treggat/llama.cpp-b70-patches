#ifndef GGML_SYCL_FATTN_ONEDNN_HPP
#define GGML_SYCL_FATTN_ONEDNN_HPP

#include "common.hpp"

// Static-only check: fused-XMX oneDNN Graph SDPA path==flash-attn op
// (f16 KV, no softcap/ALiBi, single stream, tuned head_dim, prefill-sized q.)
bool ggml_sycl_flash_attn_ext_onednn_supported(const ggml_tensor * dst, bool use_shape_limit = true);

// True when the oneDNN path binds an F16 KV cache in place instead of staging a dense copy of
// it. Depends only on the types and strides of K and V, so the answer holds for every call.
bool ggml_sycl_fattn_onednn_binds_kv(const ggml_tensor * K, const ggml_tensor * V);

// Run flash attention through oneDNN's fused xmx SDPA
// execute the cached SDPA partition, write the f32 dst. Falls back to the TILE kernel on any failure.
void ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// LOCAL (prefillx): a KQ mask carrying the GGML_KQ_MASK_HINT_CAUSAL hint with op_params[3] == 1 was not uploaded by the
// scheduler (llama LLAMA_KQ_MASK_HINT=3); rebuild its data on the device from the hint before a kernel reads it
// (op_params[3] = 2 afterwards). No-op for any other mask.
void ggml_sycl_fa_mask_ensure(ggml_backend_sycl_context & ctx, const ggml_tensor * mask);

// LOCAL (prefill2) GGML_SYCL_FA_STATS=1 (default 0): per-process counters of the flash-attention dispatch and of the
// KQ-mask hint as the SYCL backend sees it, printed to stderr at exit and whenever the call count reaches a power of
// two (>= 16). =2 additionally runs the explicit-mask oneDNN partition after every implicit-causal call and compares
// the two f16 outputs bit for bit (test mode, synchronous, slow).
enum ggml_sycl_fa_stat_id {
    FA_ST_CALLS = 0,          // FLASH_ATTN_EXT calls
    FA_ST_K_ONEDNN, FA_ST_K_XMX, FA_ST_K_TILE, FA_ST_K_VEC, FA_ST_K_MKL,   // kernel picked by the dispatcher
    FA_ST_MASK_NONE,          // no mask
    FA_ST_MASK_UNTAGGED,      // mask without the causal hint
    FA_ST_MASK_TAGGED,        // mask with the causal hint (uploaded)
    FA_ST_MASK_NOTUPLOADED,   // hinted mask whose upload the scheduler skipped (op_params[3] 1 or 2 at entry)
    FA_ST_CAUSAL,             // oneDNN implicit-causal partition executed
    FA_ST_EXPL_TAGGED,        // oneDNN explicit-mask partition although the mask carried the hint
    FA_ST_EXPL_UNTAGGED,      // oneDNN explicit-mask partition, mask untagged
    FA_ST_CAUSAL_NOFUSE,      // causal partition declined by oneDNN (fell back to explicit)
    FA_ST_FALLBACK_TILE,      // oneDNN partition not ok -> TILE
    FA_ST_FALLBACK_EXC,       // oneDNN exception -> TILE
    FA_ST_REGEN,              // not-uploaded mask regenerated on the device
    FA_ST_CMP_CALLS,          // =2: causal calls compared against the explicit partition
    FA_ST_CMP_DIFF_CALLS,     // =2: ... with at least one output value differing in its bits
    FA_ST_CMP_DIFF_VALUES,    // =2: total differing values
    FA_ST_CMP_VALUES,         // =2: total compared values
    FA_ST_COUNT
};
int  ggml_sycl_fa_stats_mode();
void ggml_sycl_fa_stat(ggml_sycl_fa_stat_id id, int64_t n = 1);
void ggml_sycl_fa_stat_maxdiff(double d);
void ggml_sycl_fa_stats_tick();   // after a call: periodic print

#endif // GGML_SYCL_FATTN_ONEDNN_HPP
