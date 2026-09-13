# Changelog

All notable changes to this fork (areslp/1Cat-vLLM) are recorded here. Entries
are grouped by branch; experimental behaviour stays behind env flags that
default off unless stated otherwise.

## Unreleased (branch `exp/sub-block-chunks`)

### Added

- `--prefix-cache-retention-interval` / `CacheConfig.prefix_cache_retention_interval`
  (default `None`, byte-identical): sparse retention of Mamba `align`-mode
  state snapshots, ported from upstream vLLM (#43447, #45845, default 0 there
  since #55353). Every 1568-token boundary used to leave a hashed snapshot per
  Mamba group in the LRU tail, so one 221K prefill popped ~570 of the 683
  blocks and evicted every other request's cached attention prefix (the
  "221K healthy / 237K cliff" of 1CatAI/1Cat-vLLM#490). `0` keeps only the
  prompt-end boundary state, `N` (multiple of the scheduler block size) also
  keeps one snapshot per `N` tokens; non-retained snapshots carry no hash and
  are reused before any other free block (`BlockPool(reuse_unhashed_first)`,
  `FreeKVCacheBlockQueue.prepend_n`, enabled only with the knob set). With
  `0`: resend of a 221K prompt after another 221K prefill hits 1.0 (was 0,
  296 s), two 245,760-token contexts run concurrently with hit 0.995, TTFT
  4.3 / 6.7 s, 9.3 / 14.0 tok/s, zero preemptions; greedy outputs after a miss
  and after a hit identical. Test: `tests/v1/core/test_prefix_cache_retention.py`.
- `VLLM_FLASH_V100_PREFILL_D256_BM32_ANY_PAGE` (default off): the SM70 D256
  BM32 phase paged-prefill kernel accepts any KV page size that is a multiple
  of its 16-token page slot. With MTP the align-mode attention block becomes
  816 instead of 784 and previously fell off the page-784-only fast path
  (104 ms vs 44 ms per 784-token chunk call at 104K context); MTP4 prefill of
  a 131K prompt next to a resident 240K decoder drops from 314.7 s to 207.9 s.
  Page 784 keeps its existing gate and code path. Test:
  `tests/kernels/attention/test_sm70_flash_v100_paged_prefill_any_page.py`.
- `VLLM_FLASH_V100_PREFILL_PREFIX_DECODE_ROWS` (default off, commit d8afc4722):
  small-query rows (1 <= q <= 16) of a mixed prefill+decode batch run on the
  paged decode kernels instead of the per-sequence paged prefill kernel,
  removing ~0.9 s per step for a 240K resident decoder (1CatAI/1Cat-vLLM#490).
- `VLLM_1CAT_PREFILL_PACE_STEPS` (default 0, commit d8afc4722): while another
  running request decodes, a request still in prompt prefill is scheduled a
  chunk only every N engine steps.
- `VLLM_1CAT_ALLOW_SUB_BLOCK_PREFILL` (default off, commit 67c36d64b): allow a
  prefill token budget below the Mamba align block size.
