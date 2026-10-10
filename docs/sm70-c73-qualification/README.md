# SM70 qualification of the frozen c73 residual patch

The runtime implementation is `f64666c6b368483ff30b8c9872d5194c579a9635`,
rebased onto upstream `c73eb2d1d1009e1ab8b980f9c1dea265ba877015`.
Draft PR903 subsequently adds two test-only fixture/default fixes at
`e5f648f235e5bcbb9efe3476258c492eb7ba7ce1`; runtime and native source are
identical to the implementation exercised here. Production D is a separate,
previously qualified deployment. These records do not qualify later main.

## Native and component results

`native-summary.json` records all ten completed native build targets,
twelve actual native module imports, nine ordinary runtime imports,
123 changed or added native schemas, and policy ABI 55 for both core
and MoE modules. The inspected core and QSA cubins target SM70.
Core native source has no residual diff from c73. Attention residual
source is unchanged from previously qualified D; its matching binaries
were reused. Native token export/reload has two CPU passes, not CUDA passes.

`component-summary.json` lists the executed component cases: MTP 7,
GDN/PLE 58, NVFP4/state 67, and attention 55; all pass with no skips.
These suites include CUDA numerical/state/changed-input/graph tests and
CPU control-flow checks. The total of 187 is not an all-CUDA test count.
M1/M5 MoE routing and graph replay have seven explicit CUDA cases.
The retained PLE admission helper additionally checks the real layer
convolution width; its manually constructed test fixture now supplies that
field. The upstream inline guard derives this geometry from the state length,
dilation and weight shape. This fixture failure was before kernel invocation.
The MTP switch test now covers the upstream default enabled and explicit 0/1.
Original failed attempts are retained privately; no failed case is relabeled.

The first ordinary E text scope completed with MTP4, FP16, TP4 and the
unchanged NVFP4 weights. All four workers used the selected clean source;
their six mapped native modules matched the packaged hashes. The launcher
uses the ordinary server entry point, without overlays or worker extensions.

Basic gold checks pass 10/14. The four failures (JSON boolean, JSON fields,
parentheses and remainder) are the same failures observed on D. Parallel
strict gold checks pass 4/8, also matching D; the two unique-copy checks pass
and no cross-talk is observed. Automatic tool calling and constrained JSON
pass. Three-anchor retrieval at both 32K and 64K passes. These are small
synthetic acceptance checks, not a comprehensive model-quality benchmark.

The subsequent ordinary pure-c73/E scope passed genuine image and video
requests on both arms, using generated red fixtures. Both arms returned red;
the image bytes and one-second video bytes were identical across arms.
The original first-scope result retains its media-client Pillow import
failure, which occurred before any media request; that attempt is not
relabeled as entirely passing.

The first same-c73 performance run had 23 backbone and 28 draft selection
differences. Its records are retained under labels ending in `-unaligned`.
They cannot isolate the residual patch from autotuning differences. The
aligned-choice run uses the unsuffixed `upstream-timed` and `E-timed` labels.

## Aligned-choice endpoint observations

All eight groups complete the same fixed token budgets: 32 requests and
11,264 output tokens per arm. Request bytes/token SHA, real prefill work
and prefix reuse match in every group; no extra requests or preemptions
are observed. The median E/U wall ratio is 0.886137 (11.4% lower latency).
All eight observed E walls are lower. This is one ordered U-to-E run,
without reversed-order replication or a statistical confidence claim.

| Group | Upstream seconds | E seconds | E latency change | Request drafts U/E | Decode cost per request-draft change |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode_1024_8_0 | 39.331 | 26.247 | -33.27% | 2836/2790 | -34.08% |
| main_1024_1_0 | 2.190 | 2.111 | -3.59% | 56/56 | -4.21% |
| main_1024_1_1 | 2.091 | 2.015 | -3.64% | 53/53 | -4.40% |
| main_1024_4_0 | 5.053 | 4.086 | -19.13% | 214/214 | -23.51% |
| main_1024_8_0 | 7.483 | 5.650 | -24.50% | 433/412 | -28.33% |
| main_1024_8_1 | 7.895 | 5.780 | -26.79% | 445/425 | -30.02% |
| main_32768_1_0 | 2.352 | 2.282 | -2.98% | 48/48 | -4.75% |
| main_65536_1_0 | 2.603 | 2.523 | -3.09% | 49/49 | -4.61% |

The 32K and 64K timed requests reuse 31,008 and 63,648 prompt tokens,
respectively, on both arms. They compute only 1,760 and 1,888 new prefill
KV tokens. Their preceding full-context warm requests take approximately
12.731/12.130 seconds and 24.733/23.582 seconds (U/E), including any first-use
work; those are not pure steady-state full-prefill comparisons. Three-anchor
long quality retrieval was checked separately on E.

The C8 groups reduce draft rounds 433 to 412 and 445 to 425; acceptance
rises from 34.35% to 37.20% and 32.53% to 35.29%. Their summed request decode
cost per summed draft falls 28.3% and 30.0%. Symmetric decomposition attributes
-1.818/-1.750 seconds to less work and -12.074/-13.444 seconds to changed cost.
These are overlapping request wall clocks including CPU, scheduling and
collectives, rather than isolated GPU or batch-step timings. C4 has the same
214 total request drafts and lower cost. Four single-request groups have
identical output text and draft work; concurrent outputs can differ.

After ordinary fresh compilation, all 92 backbone and 80 draft parameters
match. The 48/36 rank-specific serialized helpers have one saved compiled
candidate each, with matching config and cubin/metadata references across
arms and no unresolved references. All 196 common generated Python sources
match. Ordinary math-cache snapshots change zero files during timing.
The broader cache inventory still contains unequal inactive tuning JSON;
that raw comparison is retained separately from the active-role review.
No runtime CUDA-call trace or single-kernel speed claim is made.

## Reproduction

Use the repository's virtual environment and test dependencies on SM70.
Build the native extensions from the selected source for your Torch/CUDA
ABI. `environment.json` records the hardware, compiler and serving settings.
`serving-arguments.json` contains the actual ordinary CLI argument list;
replace `MODEL_DIR` with the unchanged local NVFP4 model. Apply the common
explicit environment options in `feature-options.json` to both arms and
the E residual values to the candidate arm. Give the arms separate
`VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` directories.
`scope.json` lists the component files/selectors. The following reproduces
the selected MTP numerical scope:

```sh
CUDA_VISIBLE_DEVICES=0 \
VLLM_SM70_MTP_MOE_FP16_EXACT=1 \
VLLM_SM70_MTP_MOE_TUNED_CONFIG=1 \
.venv/bin/python -m pytest -q tests/kernels/moe/test_sm70_mtp_moe_fp16.py \
  -k 'graph_changed_inputs_routes_and_canaries or native_rejects_unaligned_weight or modular_experts_route_and_graph'
```

Run the other listed component scopes on the same native build. Use the
current PR's PLE test fixture. This report does not ship weights or native
binaries; those require compatible local builds and the fixed NVFP4 model
conversion described in the environment record.

The portable `replay.py` sends the published synthetic request bodies and
validates the actual endpoint tokenizer before replay. It uses the ordinary
HTTP/SSE endpoint and supports the saved group or exact request ID.
`--reset-prefix-cache` is an explicit alternative experiment; the recorded
ordinary cohorts do not reset prefix cache between requests.

```sh
.venv/bin/python docs/sm70-c73-qualification/replay.py \
  --base-url http://127.0.0.1:8000 --group main_1024_8_0 --out /tmp/replay.json
.venv/bin/python docs/sm70-c73-qualification/summarize.py
```

Use repository multimodal dependencies for the media smoke test. The checked-in
1713-byte video is a generated one-second red fixture, not private media.

```sh
.venv/bin/python docs/sm70-c73-qualification/multimodal_smoke.py \
  --url http://127.0.0.1:8000 --out /tmp/image.json
.venv/bin/python docs/sm70-c73-qualification/multimodal_smoke.py \
  --url http://127.0.0.1:8000 --out /tmp/video.json \
  --video docs/sm70-c73-qualification/synthetic-red-1s.mp4
```

Paired request hashes, input token hashes, completed output budgets,
server request counts, output text, prefix reuse and MTP draft work must
be reviewed together. Wall ratios are descriptive endpoint observations;
per-draft decode phase includes CPU, scheduling and collectives. It is not
an isolated kernel measurement. Ordinary cached choices are observed, with
new kernels allowed; there is no replay loader or kernel allowlist.

## Ordinary compilation choice control

Torch 2.10 standalone/AOT artifacts carry both static autotuners and bundled
autotuning records. Loading an old artifact can restore its compiled
candidates and overwrite a new `.best_config`. Copying JSON to an unused
model hash, or retaining old inner artifacts, does not align active choices.

We match backbone and draft separately. The 92 backbone and 80 draft
records have identical helper filenames, source SHA and candidate-set hash
across the two frozen arms. `canonical-kernel-choices.json` publishes the
chosen parameters. The `.best_config` `triton_cache_hash` field is not a
binary constraint: Torch discards it on reading. It is omitted from the
portable choice file.

Use stopped, separate task caches. Obtain each arm's active backbone/draft
AOT directories and inner vLLM namespace from ordinary startup logs.
Preserve the initial evidence and copy the recorded choices into the actual
role cache, for example:

```sh
.venv/bin/python docs/sm70-c73-qualification/align_choices.py \
  --backbone /tmp/comparison/active-backbone/inductor_cache \
  --draft /tmp/comparison/active-draft/inductor_cache
```

The default prints the matched count. Add `--apply` to write the records.
Before restarting, back up and move each active rank's outer `model`, the
active inner vLLM namespace, and `fxgraph`/`aotautograd` within each role's
Inductor directory. Preserve helpers, matching `.best_config` and pure
Triton binary caches. Our recorded run instead backed up the complete active
role trees and rebuilt those trees with only the matched choices, producing
new helpers normally. Production caches are separate.

Start a new ordinary server process with the same configuration; warm all
six timing shapes before the eight timed cohorts. Actual Torch remote and
bundled remote autotuning defaults were unset and unused. After timing,
compare active-role choices, generated helper sources and ordinary
serialized compiled-result references. Cache snapshots before and after
timing detect new compilation without changing the loader or rejecting it.

Recompute the earlier unaligned observations explicitly:

```sh
.venv/bin/python docs/sm70-c73-qualification/summarize.py \
  --labels upstream-timed-unaligned E-timed-unaligned
```

AI assistance was used. Human line-by-line review and hosted CI are pending.
