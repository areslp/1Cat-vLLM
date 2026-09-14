# Changelog

All notable changes to this fork (areslp/1Cat-vLLM) are recorded here. Entries
are grouped by branch; experimental behaviour stays behind env flags that
default off unless stated otherwise.

## Unreleased (branch `exp/sub-block-chunks`)

### Fixed

- Speculative decoding: a batch that is uniform by shape only (every request
  scheduled with exactly `1 + num_speculative_tokens` tokens but no draft
  tokens, e.g. a prefill chunk of 8 tokens with DFlash2 K=7 or 5 tokens with
  MTP4) is no longer dispatched to the FULL cudagraph captured for the
  speculative query length (`vllm/v1/worker/gpu/cudagraph_utils.py`:
  `is_speculative_uniform_batch`, guard in `model_runner.execute_model`).
  That graph replays the verify kernels, whose spec-state metadata (Mamba
  slot page ids, slot selectors, verify block tables) is only rebuilt for
  requests carrying draft tokens, so the replay ran on the previous
  occupant's metadata: the GDN layers wrote the finished request's stale
  slot pages, the running state fell one chunk behind, and once the stale
  pages belonged to another group the next chunk's first attention layer
  went NaN (out-of-vocabulary token id 248320 until max_tokens). Such a
  batch now runs as a regular piecewise/eager batch; real verify batches
  are unchanged.
- `--prefix-cache-retention-interval`: blocks freed without a prefix-cache
  hash are put at the front of the free queue only when the next scheduler
  step starts (`BlockPool.flush_pending_front` from
  `KVCacheCoordinator.new_step_starts`), never within the scheduling pass
  that freed them. Correction: this was first recorded as the fix for the
  DFlash2 garbage output above; it only changed which pages the stale spec
  metadata pointed at (same-group pages, finite garbage) and hid the
  symptom. Kept because it is harmless; the real fix is the dispatch guard.

### Added

- `VLLM_SM70_CG_DISPATCH_DEBUG` (default off): the MRV2 runner logs one line
  per step with `num_reqs`, `uniform_tok_count` and the chosen cudagraph mode,
  used to attribute the two-decoder step cost on TP4 (clean 2 x 8 verify steps
  replay the FULL graph at 47 ms; the ~11% non-uniform steps run eagerly).
  CPU regression test pins the dispatch decision for a 2-request verify batch.

- SM70 DFlash2 verify: the per-request grouped-verify skeleton (request
  metadata rows, seq_lens views, output slices) is built once per step and
  reused across the 16 full-attention layers (`_dflash2_per_request_skeleton`),
  and the Triton small-query metadata kernel is compiled at boot for reqs
  1..max_num_seqs (`_warmup_sm70_dflash2_smallq_metadata_kernel`) so the first
  verify of a new batch shape no longer pays a JIT spike. Native calls and
  numerics unchanged. Production note: `VLLM_FLASH_V100_DFLASH2_BATCHED_GROUPED_VERIFY=1`
  (request-major batched verifier, bitwise-identical to per-request) takes the
  TP4 two-decoder step from 0.124 to 0.114 s and is now exported by the serve
  script.

- `VLLM_FLASH_V100_GROUPED_VERIFY_MULTI_KV_HEAD`: multi-request verify batches
  on a rank that holds a single KV head (the TP4 layout of Qwen3.8-27B) now
  take the per-request one-pass grouped verifier as well, one native call per
  request on its contiguous q x 6 x 256 slice (E5M2 q 8/16, E4M3 FP32 q 2..8),
  instead of falling past the B=1 native gate to the per-row XQA scan. TP4 two
  decoders 240K+16K: 0.225 -> 0.124 s/step. Log line "multi-request verify
  batch takes the grouped per-request single-KV-head route". The two-KV-head
  branch and the flag-unset behaviour are unchanged.

- `VLLM_FLASH_V100_GROUPED_VERIFY_MULTI_KV_HEAD` (default off): on a TP rank
  that holds several KV heads (TP2 of the 24/4-head Qwen3.8-27B), the SM70
  one-pass grouped verifiers (E4M3 FP32 q2..8 and the legacy E5M2 DFlash2
  q8/q16 entry) run once per KV head on strided KV views instead of falling
  back to one context scan per verify row. Covers the single-request verify
  step and the verify rows of a mixed prefill+decode batch. At 240K on
  2x V100 the DFlash2 q7 step drops from ~0.20 s to ~0.075 s on fp8_e5m2
  (23 -> 77-107 tok/s solo; 7 -> 19 tok/s while another request prefills)
  and from 0.43 s to 0.12 s on fp8_e4m3. Test:
  `tests/kernels/attention/test_sm70_grouped_e4m3_fp32_multi_kv_head.py`.
  Verify batches holding several requests (request-major, one span per
  request) take the same route per request: two decoders at 240K + 16K go
  from 0.274 s to 0.148 s per step (A 12.5 -> 21.4 tok/s, B 11.4 -> 17.6).
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
