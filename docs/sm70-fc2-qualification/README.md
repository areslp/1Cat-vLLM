# Frozen-main SM70 residual qualification

Upstream is `1CatAI/1Cat-vLLM` at `fc2f145aebee0d2e7cbfc2b520c383cb64644e57`; the tested twelve-commit candidate is `c64c4824e830301b3f0fec2da47a202e49ede20e`. This evidence commit adds documentation and synthetic data only. The implementation is the candidate, not this evidence commit. Existing draft PR903 is retained; there is no upstream merge.

The local M1 tile override no longer shadows upstream native admission at M1/M5. Native GDN, PLE, compact-head and live-prefill routes remain first. Retained guarded behavior covers GDN/shortconv metadata and state, PLE layout/output fallbacks, shared-gate/NVFP4 scratch, MTP selected/padded rows and transformed-logits fallback, grouped segmented page4 attention, finite warmup and optional pacing. Engine draft policy is captured once and participates in relevant graph hashing. Pacing is disabled. E7-on is outside this validation; complete source is preserved privately. Later PR1007 merged after the frozen upstream snapshot and is not qualified by these results.

## Complete test results

Exact-source CPU scope: 400 passed, 80 CUDA/JIT-dependent skips. Exact-source SM70 GPU scope: 480 passed, zero skips, 43.34 seconds. Official hooks passed on the complete 56-file implementation diff, including mypy, configuration/environment metadata and layering. Public numerical/state/changed-input/graph fixtures are in the twenty implementation test files below. Follow repository `uv` instructions and install a source-complete build, including the attention wrapper. Run `.venv/bin/python -m pytest --noconftest -q` on these paths; hardware-dependent cases must be run on SM70 to reproduce the 480-pass GPU scope.

- `tests/kernels/attention/test_sm70_grouped_e4m3_fp32_multi_kv_head.py`
- `tests/kernels/attention/test_sm70_segmented_page4.py`
- `tests/kernels/moe/test_sm70_mtp_upstream_dispatch.py`
- `tests/kernels/moe/test_sm70_unquantized_moe_config.py`
- `tests/models/qwen4_exp/test_production_fusion_equivalence.py`
- `tests/models/qwen4_exp/test_sm70_gdn_ba_verify_dispatch.py`
- `tests/models/qwen4_exp/test_sm70_gdn_mixed_qkv_priority.py`
- `tests/models/qwen4_exp/test_sm70_gdn_verify_fused.py`
- `tests/models/qwen4_exp/test_sm70_ple_ngram_dispatch.py`
- `tests/test_jit_monitor.py`
- `tests/v1/attention/test_gdn_spec_state_contract.py`
- `tests/v1/core/test_prefill_pacing.py`
- `tests/v1/spec_decode/test_eagle_prefill_moe_rows.py`
- `tests/v1/spec_decode/test_sm70_draft_policy.py`
- `tests/v1/spec_decode/test_sm70_mtp_head_scope.py`
- `tests/v1/worker/test_gpu_cudagraph_2req_verify_dispatch.py`
- `tests/v1/worker/test_gpu_cudagraph_prefill_state_priority.py`
- `tests/v1/worker/test_gpu_cudagraph_uniform_guard.py`
- `tests/v1/worker/test_gpu_warmup_blocks.py`
- `tests/v1/worker/test_sm70_runner_warmup_owner.py`

## Paired endpoint observations

`observations.json` contains all synthetic prompts and whitelisted warm/formal observations, including output text, usage, request/token hashes, phase wall times and MTP counters. It excludes host paths, private logs, weight files and compiler binaries. Four PCIe V100 32GB GPUs, source-complete native builds and unchanged FP16/TP4/MTP4/NVFP4 settings are recorded in `environment.json`. The exact local weight conversion is not distributed; identical endpoint results require the same weights/tokenizer. Public numerical fixtures and aggregate recomputation are independently reproducible.

Both arms completed 28 warm cohorts, then all twelve formal cohorts. Prefix state was reset before each benchmark cohort. No compiler/autotuner/loader selection gates were installed during final timing; warm collectors were removed. All mathematical cache artifact ledgers stayed unchanged in formal timing. The 76 matched helper identities have zero selected-configuration differences. Source-dependent binary hashes are not claimed identical.

The formal comparison has 49 requests and 14,336 output tokens per arm. All paired request bytes, input token hashes/counts and output budgets match. Generated text differs in every formal cohort; MTP work is not fixed by fixing the output budget. Ten candidate cohorts are faster, two slower. Three short C1, two short C4, two short C8 and one of each other cohort are descriptive observations, not a statistical or per-patch speedup proof. `main_INPUT_CONCURRENCY_REPEAT` requests 128 output tokens per request; `decode_INPUT_CONCURRENCY_REPEAT` requests 1024.

| Group | Main wall s | Candidate wall s | Candidate/main |
| --- | ---: | ---: | ---: |
| main_1024_1_0 | 2.042757 | 2.157572 | 1.056206 |
| main_1024_1_1 | 2.109966 | 2.001246 | 0.948473 |
| main_1024_1_2 | 2.145716 | 1.876715 | 0.874633 |
| main_1024_4_0 | 5.134317 | 3.999671 | 0.779007 |
| main_1024_4_1 | 5.275247 | 4.017056 | 0.761492 |
| main_1024_8_0 | 7.638087 | 5.666986 | 0.741938 |
| main_1024_8_1 | 7.670725 | 5.790916 | 0.754937 |
| main_8192_8_0 | 26.080970 | 23.156713 | 0.887878 |
| main_32768_4_0 | 47.156202 | 45.061125 | 0.955572 |
| main_65536_1_0 | 24.775833 | 23.781089 | 0.959850 |
| decode_1024_1_0 | 13.213959 | 13.417845 | 1.015430 |
| decode_1024_8_0 | 41.829155 | 27.406028 | 0.655190 |

The +5.62% short C1 observation has 51→57 draft rounds and 33.313→31.865 ms server decode wall per round. Exact symmetric decomposition gives +195.53 ms work and −78.21 ms per-round cost, totaling +117.33 ms decode. The +1.54% long C1 observation has 388→408 rounds and 33.110→31.984 ms per round. These are phase wall/work figures, not GPU kernel times. The earliest divergent kernel has not been identified. Both arms have the same four pre-existing basic-gold failures and 10/14 passes; tool/JSON pass, with no new basic-gold failure. Earlier exact-source warm coverage also passed retrieval at 8K/32K/64K and exercised 64K eight-way concurrency; those are not extra formal timing samples here.

Two earlier attempts failed due to private observer/loading restrictions. They are retained as failed controls, not model numerical failures. The final run removed those restrictions. Earlier same-source fixed-cache M1 comparison on source96 has 26 identical-output/work pairs and median new/old wall ratio0.996915, effectively a tie; that separate experiment does not qualify this final source's long/concurrent behavior.

## Recompute and replay

Recompute all twelve pairs with no GPU or model load:

```sh
.venv/bin/python docs/sm70-fc2-qualification/summarize.py
```

Replay a cohort against an ordinary source-complete serving endpoint with the recorded configuration and served model name `flash-next`:

```sh
.venv/bin/python docs/sm70-fc2-qualification/replay.py --base-url "$BASE_URL" --group main_1024_4_0 --out replay.json --reset-prefix-cache
```

The prefix-reset endpoint requires benchmark dev mode. For this benchmark, complete the same finite warmup and reset prefix state before both warm and formal cohorts. If prefix reset is unavailable, record first-use/cache-hit observations explicitly and do not equate them to this reset-prefix matrix. The client validates the actual tokenizer before timing and preserves every output; it does not install compiler or loading controls.

## Ordinary production acceptance

The exact candidate was accepted locally on 2026-10-09 after ordinary startup, without a benchmark worker extension, dev mode, source overlays or compiler/loading restrictions. Four GPU workers loaded the qualified core and attention libraries from the clean candidate. Original production C is `0c49127aece147ab0275c4eed1e59af2edb56003`; its source and service configuration remain available for rollback.

`ordinary-production.json` records the separate acceptance aggregates. SSE, image, video, tool calling and constrained JSON pass. Basic strict gold is 10/14, with the same four failures as C. Eight concurrent strict golds are 4/8, exactly matching C; both unique-copy routing checks pass with no cross-talk. This is retention of the baseline, not universal model correctness.

| Input / concurrency / output per request | C wall s | D wall s | D/C |
| --- | ---: | ---: | ---: |
| 1024 / 1 / 128 | 2.161322 | 2.151931 | 0.995655 |
| 1024 / 4 / 128 | 4.101711 | 4.153103 | 1.012529 |
| 1024 / 8 / 128 | 6.003813 | 5.667853 | 0.944042 |
| 32768 / 1 / 128 | 12.228640 | 12.308495 | 1.006530 |
| 65536 / 1 / 128 | 23.911243 | 23.746250 | 0.993100 |

These five cohorts use the same request bytes and output budgets as the saved C baseline. Prefix reset is unavailable on the ordinary endpoint; recorded prompt-cache hits are zero. Output chains and MTP work can differ. Every wall ratio meets the local 1.05 admission threshold; single observations do not establish statistical equivalence. The full formal main/D comparison above remains the upstream qualification evidence.

## Review boundary

Local hooks and GPU checks are separate from hosted CI. New-head hosted CI and current-main mergeability must be read from PR903; neither is inferred from local passes. This remains a draft with AI assistance disclosed, for human review of every changed line, overlapping later upstream work and results before promotion or merging. Ordinary local deployment acceptance is a separate operational record.
