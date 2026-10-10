# Frozen 5dc integration and reproducible phase measurements

This is a CPU integration of PR903 head
`e5f648f235e5bcbb9efe3476258c492eb7ba7ce1` with frozen upstream
`5dc280bb7f073f76c514a4696c5f434a0080c156`. It does not follow later main.
The merge has no textual conflicts. Git auto-merges `vllm/envs.py`,
`vllm/model_executor/layers/logits_processor.py`,
`vllm/model_executor/models/qwen2_moe.py`, and
`tools/pre_commit/layering_baseline.json`.

## Integration and native boundary

The collective, graph and execution-policy implementation is upstream's.
The retained top1 transport remains construction-scoped; the ordinary top1
fallback now consumes the upstream captured communication policy. The d1a
guard also receives that captured policy instead of reading the migrated
environment alias during execution. Shared-gate
fallback still follows upstream native admission. Graph dispatch retains the
new `GraphExecutionPlan`; padded draft-row adapters still cover decode and
multistep capture and restore their scope on exceptions. CPU tests exercise
these owners, engine interleaving, typed overrides and serialization.

The c73-to-5dc native delta is six files: `custom_all_reduce.cu`,
`custom_all_reduce.cuh`, `custom_all_reduce_policy.h`,
`custom_all_reduce_policy_fields.inc`, `ops.h`, and `torch_bindings.cpp`.
The new `init_custom_ar_configured(..., str[] policy)` binds an 18-field
policy to the communicator that owns the native pointer. Registration,
execution and destruction stay with that DSO. Missing configured ABI
disables the custom provider before allocation and falls back; that does
not qualify an old binary for equivalent performance. Core/MoE policy ABI
55 alone is insufficient evidence for this new collective entry point.

Rebuild the core `_C` extension from this tree before GPU qualification and
check the configured schema and real distributed routes. Core native,
`CMakeLists.txt` and `setup.py` have no residual diff from frozen 5dc.
Attention residual source is unchanged from tested E; its prior build is
only reusable after source, compiler/Torch/CUDA ABI and artifact provenance
are checked. No new native build, CUDA kernel, graph replay, model request
or GPU window was executed for this CPU integration. Old tested trees and
production remain separate and unchanged.

## CPU verification

The final 61-file scope completed **1222 passed, 173 skipped, zero failures**;
one cold-import test was deselected there and passed separately. Skips are
CUDA/SM70/native cases, not GPU passes. `cpu-scope.json` records the files,
`cpu-summary.json` records the environment and failures retained from earlier
attempts, and `tested-source-sha256.json` binds the tested residual sources,
tests and reproduction scripts to file bytes.

The Turing backend-priority file was not collected successfully: its module
imports `vllm._C`, which is absent in this source-only checkout. It remains
unverified here rather than being run against an old native substitute.
The CPU bootstrap selects the real `CpuPlatform`; individual unit tests
use their own mock dependencies and do not establish CUDA numerical parity.

On the Linux CPU test host, use the immutable evidence checkout's
`docs/sm70-c73-cpu/cpu_bootstrap` and this checkout's dependency environment.
The equivalent final pytest command is:

```sh
CPU_BOOTSTRAP=/path/to/evidence/docs/sm70-c73-cpu/cpu_bootstrap
mapfile -t cpu_tests < <(jq -r '.[]' docs/sm70-main5dc-integration/cpu-scope.json)
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH="$CPU_BOOTSTRAP:$PWD:$PWD/flash-attention-v100" \
.venv/bin/python -B -m pytest --noconftest -q \
  -k 'not test_collective_cold_import_resolves_cuda_platform_before_config' \
  "${cpu_tests[@]}" --junitxml=/tmp/main5dc-cpu.xml
```

For the cold-import case, omit the inherited bootstrap so its subprocess can
exercise the CUDA-platform import order without initializing a CUDA context:

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH="$PWD:$PWD/flash-attention-v100" .venv/bin/python -B - <<'PY'
import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform
platforms._current_platform = CpuPlatform()
import pytest
raise SystemExit(pytest.main([
    '--noconftest', '-q',
    'tests/config/test_collective_policy.py::test_collective_cold_import_resolves_cuda_platform_before_config',
]))
PY
```

## Evidence scope

The immutable [c73 evidence](https://github.com/areslp/1Cat-vLLM/tree/b352d3967a166bdc468b21b959ebe98529032d1a/docs/sm70-c73-qualification)
qualifies only c73 plus runtime `f64666c6b368483ff30b8c9872d5194c579a9635`.
It is not GPU evidence for this merge. It contains eight matched groups,
32 requests and 11,264 output tokens per arm, and a descriptive median
end-to-end E/U ratio of 0.886137. C4's prefill-rate estimate is 16.9% lower
and TTFT 9.4% higher, while mean queue time changes from about 0.00021 to
0.07988 seconds. That queue increase is close to the TTFT increase;
scheduling is a priority for investigation, not proof of a slower kernel.

All four C4 outputs and all eight outputs in each short/long C8 comparison
also change between upstream's own warm and timed cohorts. Concurrent
differences therefore cannot be labeled E-specific correctness failures.
First-divergence causality and repeatability remain open. The timed 32K/64K
groups reuse 31,008/63,648 tokens and compute only 1,760/1,888 new KV tokens.
They are not cold full-context prefill measurements.

## Recompute the historical tables on CPU

Read the public evidence at its immutable commit into a separate checkout.
The scripts here accept that checkout's `observations.json` explicitly; they
do not download data, start servers or choose a GPU window.

```sh
EVIDENCE=/path/to/evidence/docs/sm70-c73-qualification
.venv/bin/python docs/sm70-main5dc-integration/summarize.py \
  --input "$EVIDENCE/observations.json"
```

The default labels are `upstream-timed` and `E-timed`. Pass
`--labels upstream-timed-unaligned E-timed-unaligned` only for the earlier
unaligned run. Missing metrics stay unknown, not zero. A matched request
pair requires identical request/token hashes, budgets, completed output
counts and exactly the expected prefill observations. Matching prefix and
computed-prefill work is reported separately and must also be checked.

## Collect a future authorized group

Use a dedicated, idle comparison endpoint with one model and one API server.
Do not run this against production. Build each selected arm normally and
use the immutable serving arguments, feature options, tokenizer and weights.
After normal code/shape warmup, execute the recorded groups in the same
order and preserve separate warm/timed labels. The command below sends
requests; it is a recipe, not a record of execution in this CPU task.

```sh
.venv/bin/python docs/sm70-main5dc-integration/replay.py \
  --observations "$EVIDENCE/observations.json" \
  --base-url http://127.0.0.1:8000 --group main_1024_8_0 \
  --label upstream-timed --out /tmp/comparison/U-main_1024_8_0.json
.venv/bin/python docs/sm70-main5dc-integration/summarize.py \
  --results /tmp/comparison/U-main_1024_8_0.json /tmp/comparison/E-main_1024_8_0.json
```

`replay.py` validates tokenizer count/hash before timing, saves `/metrics`
before the barrier and 0.5 seconds after the last SSE completes, and saves
the raw snapshots beside the output JSON. The half-second emission delay,
tokenization and scrapes are outside group wall time. Use one quiescent
process without restarts; counters are summed over its engine/reason label
series. Exact expected phase counts, no unrelated traffic, no counter
reset and completed output budgets are required. A count mismatch can be
delayed emission or other traffic; preserve it as unqualified and investigate.
The count check alone cannot prove complete traffic isolation.

| Field | Definition and limitation |
| --- | --- |
| Group wall | Client barrier release to completion of all request futures; endpoint wall time. |
| Phase sum/count | `after - before` of each Prometheus histogram `_sum`/`_count`; mean is sum/count. |
| Queue | First server QUEUED event to first SCHEDULED event. |
| Prefill | First SCHEDULED event to first NEW_TOKEN event, including preemption. |
| Decode | First NEW_TOKEN to last NEW_TOKEN, including preemption. |
| Server TTFT | Request arrival to first token recorded by the output processor. It need not equal queue + prefill exactly. |
| Client TTFT | First SSE content/reasoning arrival minus client request start; includes transport. |
| Prefill-rate estimate | `request_prefill_kv_computed_tokens_sum` delta / `request_prefill_time_seconds_sum` delta. Concurrent request clocks overlap. |
| Cached prompt tokens | `prompt_tokens_cached_total` delta; use computed KV tokens as the prefill-work numerator, not full input length. |
| Request drafts | `spec_decode_num_drafts_total` delta; multiple requests contribute in one engine step. |
| Acceptance | Accepted token delta / drafted token delta, with identical draft width verified separately. |
| Request draft cost | Decode phase sum / request drafts. It includes CPU, scheduling and collectives and is not GPU batch-step cost. |
| Engine iterations | `iteration_tokens_total_count` delta includes prefill; it does not give the ordered batch composition or request slots. |

For symmetric work/cost decomposition, let `N` be request drafts and
`c = decode_sum / N`. Work delta is `(N_E-N_U)*(c_E+c_U)/2`; cost delta is
`(c_E-c_U)*(N_E+N_U)/2`. Their sum equals the change in summed decode wall
clocks, not the change in non-overlapping group wall time.

Default replay does not reset prefix cache. `--reset-prefix-cache` is an
explicit different experiment. A full 32K/64K prefill run must warm code
first, reset the idle arm's prefix cache immediately before each measured
C1 prompt, then verify cached-token delta is zero and computed KV work
matches the complete prompt. Keep these results separate from cached cohorts.

## Capture compiler-cache evidence

Give U and E separate task caches. Resolve the active backbone and draft
directories for every rank from ordinary startup logs. Follow the immutable
evidence's `align_choices.py` and README: copying `.best_config` into an
inactive model hash is ineffective, and old AOT/standalone artifacts can
restore stale choices. Rebuild only the selected task's active role trees
after preserving them. Never modify production caches.

After warmup and after the complete timed cohort sequence, run this for each
active role cache and its Triton binary cache, with output outside that root:

```sh
.venv/bin/python docs/sm70-main5dc-integration/cache_snapshot.py \
  --root /tmp/comparison/U/active-backbone --out /tmp/U-backbone-before.json
# Run only the already-authorized timed cohort sequence here.
.venv/bin/python docs/sm70-main5dc-integration/cache_snapshot.py \
  --root /tmp/comparison/U/active-backbone --out /tmp/U-backbone-after.json
diff -u /tmp/U-backbone-before.json /tmp/U-backbone-after.json
```

The snapshot hashes regular file bytes, ignores symlinks and detects a file
changing while read. It is not an atomic filesystem snapshot: collect while
idle and preserve the original directories. Hashing is outside measurement.
Compare added/removed/changed helpers and best configs; audit rank/role
helper source SHA, candidate-set hash, selected parameters and serialized
compiled-result references. Zero file changes does not trace runtime kernel
selection. `triton_cache_hash` is not a binary constraint. Do not force the
historical 92/80 choice counts onto the new 5dc tree if the namespaces differ.

See [GPU_PLAN.md](GPU_PLAN.md) for the unexecuted follow-up. AI assistance
was used. The existing PR stays draft; human line review and hosted CI
remain pending. The author-trust gate is not bypassed.
