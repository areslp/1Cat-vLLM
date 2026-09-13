# Handoff: `exp/sub-block-chunks`

Tracked summary for the branch. The full brief with every measurement,
retraction and process trap lives on the development host, git-excluded:
`oh-my-gpu:/home/l/work/1Cat-vLLM/logs/handoff/HANDOFF.md` plus the
task-lifecycle contract `packet.yaml` (validates `READY`).

## 2026-09-14 — prefix-cache retention interval (this commit)

**Task intent.** Item 2 of the #490 follow-up: the report's "221K healthy /
237K cliff" is not the kernel but the prefix cache. In `mamba_cache_mode=align`
every Mamba cache group snapshots its recurrent state into a fresh block at
each 1568-token boundary and frees the superseded snapshot, hashed, into the
LRU tail: one 221K prefill pops ~570 of the 683 blocks (1 attention + 3
snapshots per boundary), so the next long prefill evicts the previous
context's attention prefix while only its final snapshot survives, and the
resend recomputes everything. Upstream vLLM hit the same problem and fixed it
with `--prefix-cache-retention-interval` (vllm-project/vllm #43447, #45845,
Marconi junctions #37898/#47782, default 0 since #55353); this commit ports
that knob (the owner's choice over a fork-only evict-first flag).

**Changed scope.**

- `vllm/config/cache.py`, `vllm/engine/arg_utils.py`: the knob
  (`None` = dense, the default; `0` = keep only the prompt-end boundary
  state; `N` = also one snapshot per `N` tokens, `N` a multiple of the
  scheduler block size), excluded from the compile hash.
- `vllm/v1/kv_cache_interface.py`, `vllm/v1/core/kv_cache_utils.py`: carried
  on `KVCacheConfig`; `FreeKVCacheBlockQueue.prepend_n`.
- `vllm/v1/core/block_pool.py`: `reuse_unhashed_first` (set only with the
  knob): unhashed frees go to the queue front, hashed ones to the back. The
  default free order is unchanged.
- `vllm/v1/core/single_type_kv_cache_manager.py`: `cache_blocks` takes
  `retention_interval` / `replay_boundaries`; `MambaManager.retention_block_mask`
  leaves unretained boundary states unhashed and skips them in
  `cached_blocks_this_step` and the offload hand-off list.
- `vllm/v1/core/kv_cache_coordinator.py`: validation, `_replay_boundaries`
  (`num_prompt_tokens - 1` and `num_prompt_tokens`, plus one block earlier for
  EAGLE/MTP groups), plumbing into both coordinators.
- `tests/v1/core/test_prefix_cache_retention.py`: free-queue order with the
  gate off/on, mask cases, and a full-attention + Mamba(align) end to end
  (dense 6 snapshots / partial-prefix sibling hit 48; interval 0 -> 1 / 0;
  interval 32 -> 3 / 32; fewer block pops with retention).
- Not ported: Marconi shared-prefix junctions (a request sharing only part of
  a cached prompt gets hit 0 at interval 0) and upstream's internal prefill
  checkpoints (#52789, needs kernel work on V100).

**Validation.**

- `pytest tests/v1/core/test_prefix_cache_retention.py`: 11 passed.
  `test_prefix_caching.py`, `test_single_type_kv_cache_manager.py`,
  `test_mamba_align_chunk_split.py`, `test_kv_cache_utils.py`: 143 passed,
  16 failed, the same 16 that fail on this host without the change (15 need
  Hugging Face models, 1 pre-existing `SimpleNamespace` failure).
  `ruff check` / `ruff format --check` clean.
- End to end (Qwen3.8-27B-QUASAR-NVFP4, TP2 on 2x V100-PCIE-32GB,
  fp8_e5m2 KV, decode-rows flag on, no MTP, zero preemptions):
  interval 0 -> resend of a 221,184-token prompt after another 221K prefill
  hits 221088/221183 in 1.5 s (dense: hit 0, 296 s); two distinct
  245,760-token contexts warmed then fired concurrently: hit 0.995, TTFT
  4.3 / 6.7 s, decode 9.3 / 14.0 tok/s; a prompt sharing the first 110.6K
  tokens then diverging: hit 0 (121 s). Interval 25088 (16 blocks): the same
  resend hits 1.0 and the shared-prefix prompt hits 0.891 (16.7 s). Dense:
  0.974 (4.4 s) but the cliff. In all arms the greedy 64-token output after a
  cache miss and after a hit is identical.
- Not run: MTP with the knob set (the EAGLE-group boundary is unit-tested
  only); any repeat (n=1).

**Remaining risks / follow-ups.**

- Shared-document-different-question workloads lose their Mamba resume point
  at interval 0; use a positive interval (25088 costs ~27 blocks per 221K
  context) or port Marconi junctions.
- Capacity per TP2 server with the knob: about 4 x 262K contexts (171 blocks
  each of 683); dense keeps only 1.
- Recommended production settings on this hardware:
  `--prefix-cache-retention-interval 0`,
  `VLLM_FLASH_V100_PREFILL_PREFIX_DECODE_ROWS=1`,
  `VLLM_FLASH_V100_PREFILL_D256_BM32_ANY_PAGE=1` whenever MTP is on,
  `VLLM_1CAT_PREFILL_PACE_STEPS=4`.

**Commit readiness.** Default behaviour byte-identical with the knob unset;
nothing posted to #490; PR #616 does not include this change.

## 2026-09-13 — BM32 any-page kernel path (commit 57670d4a6)

**Task intent.** Close 1CatAI/1Cat-vLLM#490 on SM70: a resident 240K-context
decoder collapsed to 0.5–0.6 tok/s while another request chunk-prefilled.
Commit d8afc4722 fixed the kernel routing of the decoder's row and added a
prefill pacing knob. This commit removes the second cost that only MTP users
pay: with MTP the Mamba page grows by `num_spec` conv slots, the align-mode
attention block becomes 816 instead of 784, and every paged-prefill chunk call
fell off the kernel's page-784-only fast path (2.3x slower).

**Changed scope.**

- `flash-attention-v100/kernel/fused_mha_forward_paged.cu`: the D256 BM32
  phase body takes `page_block_size` as a kernel argument (it was a
  compile-time 784 used only to map each 16-token page slot); the dispatch
  gate accepts any page size that is a multiple of 16 when
  `VLLM_FLASH_V100_PREFILL_D256_BM32_ANY_PAGE=1`. Page 784 keeps the existing
  gate. The M < 32 low-smem software-pipeline gates and the fp8 bridge
  workspace still assume 784 (not generalised).
- `vllm/envs.py`: documents the new knob (default off).
- `tests/kernels/attention/test_sm70_flash_v100_paged_prefill_any_page.py`:
  pages 816/896, q=32/784, flag off/on, against page 784 and a dense reference.

**Validation.**

- Extension rebuilt on oh-my-gpu (`uv pip install --no-build-isolation -e
  flash-attention-v100`, TORCH_CUDA_ARCH_LIST=7.0, ccache), import ok.
- Standalone (GPU 2, q=784 at 104K context, fp16, 12/2 heads): page 784
  43.99 ms unchanged; page 816 and 896 104.4 ms -> 43.99 ms with the flag;
  all outputs bit-identical to the dense reference and to page 784.
- `pytest tests/kernels/attention/test_sm70_flash_v100_paged_prefill_any_page.py`:
  8 passed. `tests/kernels/attention/test_sm70_flash_v100_*.py` and
  `test_sm70_e4m3_scalar_fp32.py`: 147 passed with the flag off and on.
- End to end (Qwen3.8-27B-QUASAR-NVFP4, TP2 on 2x V100-PCIE-32GB, fp16 KV,
  MTP4 via `VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1`, decode-rows flag on,
  resident decoder at 240,000 tokens, fixed prompts, zero preemptions):
  partner 16K prefill 21.3 s -> 19.6 s, partner 131K 314.7 s -> 207.9 s; with
  `VLLM_1CAT_PREFILL_PACE_STEPS=4` the 131K prefill 400.1 s -> 293.8 s while
  the resident decoder runs 9.86 tok/s (was 7.38). Output text and 512+8
  logged tokens per arm identical across boots; logprob shifts within the
  measured cross-boot noise floor.
- Not run: lint beyond `ruff check`/`ruff format` on the Python files; no
  clang-format pass on the `.cu` change.

**Remaining risks / follow-ups.**

- Two-request MTP decode steps cost 0.18 s at 240K versus ~0.10 s solo; this
  bounds prefill pacing at ~21–25 tok/s for the resident decoder. Unattributed.
- 0.15–0.23 s residual per mixed step with all flags on (drafting plus eager
  overhead). Unattributed.
- fp8_e5m2 in-server behaviour of the decode-rows path is unit-tested only.
- Every measurement is n=1; two or three resident decoders and longer
  generations are unexercised.
- Recommended production settings on this hardware:
  `VLLM_FLASH_V100_PREFILL_PREFIX_DECODE_ROWS=1`,
  `VLLM_FLASH_V100_PREFILL_D256_BM32_ANY_PAGE=1` whenever MTP is on,
  `VLLM_1CAT_PREFILL_PACE_STEPS=4`. Do not enable
  `VLLM_1CAT_ALLOW_SUB_BLOCK_PREFILL` for production.

**Commit readiness.** Default behaviour byte-identical with the flags unset;
nothing here is proposed upstream as-is; nothing has been posted to #490.
