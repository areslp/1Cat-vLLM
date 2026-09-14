# Handoff: `exp/sub-block-chunks`

Tracked summary for the branch. The full brief with every measurement,
retraction and process trap lives on the development host, git-excluded:
`oh-my-gpu:/home/l/work/1Cat-vLLM/logs/handoff/HANDOFF.md` plus the
task-lifecycle contract `packet.yaml` (validates `READY`).

## 2026-09-14 — TP4 round 12b: verify skeleton cache, metadata-kernel warmup

**Task intent.** Attribute and cut host-side cost of multi-request verify
steps on TP4; settle the NCCL question.

**Changed scope.** `vllm/v1/attention/backends/flash_attn_v100.py`
(`_dflash2_per_request_skeleton`, per-step cache consumed by
`_smallq_grouped_per_request`; native calls unchanged),
`vllm/v1/worker/gpu/model_runner.py` (`_warmup_sm70_dflash2_smallq_metadata_kernel`
from `_warmup_sm70_aux_kernels`), CPU plumbing tests in
`tests/v1/attention/test_sm70_flash_v100_policy.py`. Serve script exports
`VLLM_FLASH_V100_DFLASH2_BATCHED_GROUPED_VERIFY=1` and NCCL tunables (defaults unchanged).

**Validation.** GPU tests 194 pass (grouped verify, smallq metadata, multi-KV-head,
policy) in the stop window; production TP4 boot with the patch + batched env:
smoke and garbage probe clean, route "request-major B2/q8/H6/Hkv1", two decoders
240K+16K 0.114 s/step (per-request 0.124), solo 240K 46 ms unchanged, aux warmup
lists `dflash2_smallq_metadata`. NCCL microbench: P2P on gives no all-reduce gain;
`NCCL_P2P_LEVEL=PHB` hangs all four GPUs — `NCCL_P2P_DISABLE=1` is mandatory.

**Remaining risks / follow-ups.** ~70-80 ms host-side time per multi-request
step (verifier wall 116 ms vs GPU 33 ms) is unattributed (not in the attention
files); eight other Triton kernels still JIT once during inference.

**Commit readiness.** Numerics unchanged; flag-gated paths only. Committed via
/cpm on 2026-09-14 and deployed on TP4.

## 2026-09-14 — TP4: per-request grouped verify for single-KV-head ranks

**Task intent.** Owner moved production to TP4 on all four V100s; multi-request
verify batches had fallen to the per-row XQA scan (0.225 s/step).

**Changed scope.** `vllm/v1/attention/backends/flash_attn_v100.py`
(`_smallq_grouped_per_request`: single-KV-head branch, one native call per
request, distinct log/route tag), tests in
`tests/kernels/attention/test_sm70_grouped_e4m3_fp32_multi_kv_head.py` and
`tests/v1/attention/test_sm70_flash_v100_policy.py`. Run script gained TP/PP/GPUS
tunables; 2-card unit `qwen38-27b-vllm-tp2.service` kept (disabled).

**Validation.** 155 kernel+policy tests pass on GPU; production TP4: smoke and
garbage probe clean, two decoders 240K+16K 0.124 s/step (was 0.225; TP2 0.148),
solo 240K 46 ms/step (TP2 73). Profile: target_forward 33 ms, draft 7, sample 3.

**Remaining risks / follow-ups.** E4M3 single-head branch not run in-server;
two-decoder phase looks CPU-bound (verifier wall 110 ms vs gpu 27 ms); NCCL P2P
within PHB pairs untested (hard-coded `NCCL_P2P_DISABLE=1`); first-verify
Triton JIT spike; mixed prefill+decode rows still need two KV heads.

**Commit readiness.** Flag-gated; default behaviour unchanged. Committed via
/cpm on 2026-09-14; production TP4 already runs this tree.

## 2026-09-14 — speculative cudagraph dispatch guard

**Task intent.** Find the exact site of the retention-0 x DFlash2 garbage
output (token 248320) that the deferred front insertion only hid.

**Changed scope.** `vllm/v1/worker/gpu/cudagraph_utils.py`
(`is_speculative_uniform_batch`), `vllm/v1/worker/gpu/model_runner.py`
(guard before `dispatch_cg_and_sync_dp`, one info_once log),
`tests/v1/worker/test_gpu_cudagraph_uniform_guard.py`, CHANGELOG correction.
Root cause: a prefill chunk of exactly 1 + K tokens per request has the shape
of a verify batch, `get_uniform_token_count` returns K + 1 and the FULL
speculative cudagraph is replayed while the spec-state metadata buffers were
last written for the previous occupant of the request slot; the replayed GDN
kernels write that request's stale slot pages, the running state falls one
chunk behind, and when the stale pages now belong to another group the next
chunk's first attention layer produces NaN.

**Validation.** Unit test 6/6; `tests/v1/worker` cudagraph/v2-runner tests 30
passed, 1 failure identical on clean HEAD; dev-pair e2e: immediate-reuse repro
config + fix 2/2 clean (9/9 bad before), production config + fix 2/2 clean;
grouped verify routes still active, pair timing unchanged.

**Remaining risks / follow-ups.** MTP4 (5-token chunks) not exercised
in-server; the `_pending_front` deferral is redundant now (owner's call to
remove); production is exposed until redeployed.

**Commit readiness.** Default behaviour identical for real verify batches;
only shape-only uniform batches change path. Committed via /cpm on
2026-09-14; production redeploy (fp8_e5m2 KV, 2 slots, DFlash2 q7) follows
in the same session (see logs/handoff/HANDOFF.md deployment notes).

## 2026-09-14 — grouped verify for multi-request verify batches (this commit)

**Task intent.** Round 9 left verify steps with several requests (two
decoders both speculating) on the per-row XQA path ("MTP verifier XQA path
active (rows=16)"). The request-major batch has one uniform query span per
request, so each request gets the single-request view (its block-table row,
its per-token lengths) and the one-pass grouped verifier runs per request and
per KV head, without host synchronisation (CUDA-graph replay safe).

**Changed scope.** `vllm/v1/attention/backends/flash_attn_v100.py`:
`_smallq_grouped_per_request`, tried first in `_call_flash_attn_smallq_decode_paged`
when `num_reqs >= 2`; E5M2 DFlash2 targets (q 8/16) and E4M3 (q 2..8); all
requests must pass the gates or the batch stays on the per-row path.

**Validation.** Two decoders (A 240K + B 16K, 256 tokens each, DFlash2 q7,
fp8_e5m2, 2 slots): 22.5 s -> 14.5 s wall, 0.274 -> 0.148 s per step, A 12.5
-> 21.4 tok/s, B 11.4 -> 17.6; outputs differ from the baseline only at
near-tie tokens (A ':' -0.73 vs '_bytes' -0.74, B ' validate' -1.14 vs
' logger' -1.35); garbage probe clean; SM70 attention/spec tests 233/233 with
the flag off and on.

**Remaining risks / follow-ups.** n=1; three decoders and E4M3 batches not
run in-server; padded MTP verify batches stay per-row by design.

**Commit readiness.** Default behaviour unchanged with the flag unset. Not
deployed (owner to decide).

## 2026-09-14 — retention interval: defer front insertion by one step (commit 3040730392)

**Task intent.** Deploying DFlash2 q7 on the 2x V100 service with
`--prefix-cache-retention-interval 0` produced, for a multimodal request that
followed two long prefix-cache requests, an answer made of the out-of-vocab
token id 248320 repeated to max_tokens (0 accepted drafts). Bisect: DFlash2 +
retention 0 fails with and without the per-KV-head flag; DFlash2 + dense
retention and MTP4 + retention 0 are clean. The retention knob handed a block
freed in the current scheduling pass straight back out in the same pass, and
the Mamba align + DFlash2 path still used that block in the step in flight.

**Changed scope.**

- `vllm/v1/core/block_pool.py`: unhashed frees under `reuse_unhashed_first`
  wait in `_pending_front`; `flush_pending_front()` prepends them.
- `vllm/v1/core/kv_cache_coordinator.py`: `new_step_starts()` flushes first.
- `tests/v1/core/test_prefix_cache_retention.py`: pending order asserted, the
  end-to-end helper calls `new_step_starts()` per step like the scheduler.

**Validation.** `pytest tests/v1/core/test_prefix_cache_retention.py
test_prefix_caching.py test_single_type_kv_cache_manager.py`: 85 passed.
Reproduction probe (two 15.5K prefix requests then a fresh 3-image prompt,
DFlash2 q7, retention 0, flag on, GPUs 0,1): 5/5 bad before, 4/4 clean after.
Sequential cliff probe with retention 0 (warm A, resend A, warm B, resend A)
re-run after the fix: see logs/cliff7_r0d_seq.json on the host.

**Remaining risks / follow-ups.** The exact freeing site under speculative
decoding is not identified; the deferral is the containment. Pending blocks
are not counted as free for one step.

**Commit readiness.** Dense default (knob unset) untouched.

## 2026-09-14 — per-KV-head grouped verify on TP2 (commit 75319e2d39)

**Task intent.** Long-context speculative decoding on 2x V100. The fork's
one-pass grouped verifiers only accept the TP4 layout (6 query heads and one
KV head per rank), so on TP2 every MTP/DFlash2 verify row scanned the 240K
context separately: MTP4 30 tok/s, DFlash2 q7 23 tok/s (fp8_e5m2). Both
native entries address the paged KV through runtime strides, so the rank's
query and KV are sliced per KV head and the entry is called once per head.

**Changed scope.**

- `vllm/v1/attention/ops/sm70_e4m3_grouped.py`: `grouped_e4m3_fp32_kv_head_views`
  and `run_grouped_e4m3_fp32_per_kv_head` (gate every head view, then run).
- `vllm/v1/attention/backends/flash_attn_v100.py`: the per-KV-head route in
  the single-request E4M3 FP32 path, the legacy E5M2 DFlash2 verifier, and
  the verify rows of a mixed prefill+decode batch
  (`_run_prefill_prefix_decode_rows_grouped`).
- `vllm/envs.py`: `VLLM_FLASH_V100_GROUPED_VERIFY_MULTI_KV_HEAD` (default off).
- `tests/kernels/attention/test_sm70_grouped_e4m3_fp32_multi_kv_head.py`.

**Validation.** Kernel tests 10/10 on GPU 0 (per-head result bitwise equal to
a contiguous single-head call, FP64 oracle, CUDA-graph replay, gating);
existing E4M3 grouped tests unchanged. End to end at 240K (TP2, fp8, ROWS +
ANY_PAGE + pacing 4): DFlash2 q7 fp8_e5m2 solo 23 -> 77-107 tok/s
(0.20 -> 0.073 s per step), during a 16K prefill 7 -> 19 tok/s, after it
19 -> 97-115; fp8_e4m3 8.7 -> 56-66 tok/s. Outputs coherent; the e5m2 and
e4m3 arms with the flag are identical over 512 tokens. tok/s differs with
the branch the fixed prompt takes at token 12 (a copy of corpus text drafts
at 8/8), so compare step times across arms.

**Remaining risks / follow-ups.** Verify batches with several requests still
scan per row (log "MTP verifier XQA path active (rows=24)"). MTP4 on fp8_e4m3
cannot serve a long request on this branch (draft-layer E4M3 XQA guard,
with and without `VLLM_FLASH_V100_E4M3_BATCH_XQA`). DFlash2 shrinks the KV
pool to 634K tokens (2.42 x 262K). All numbers n=1.

**Commit readiness.** Default behaviour unchanged with the flag unset.

## 2026-09-14 — prefix-cache retention interval (commit 59f54f8bea)

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
