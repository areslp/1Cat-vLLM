# E: CPU qualification of the retained SM70 residuals

This records the independent refresh of draft PR903 onto frozen upstream
`c73eb2d1d1009e1ab8b980f9c1dea265ba877015`, with implementation
`f64666c6b368483ff30b8c9872d5194c579a9635`. E is this new candidate.
Previously qualified production D remains
`c64c4824e830301b3f0fec2da47a202e49ede20e`. E has not been deployed.

## Implementation changes

- Preserve upstream GDN execution plans and per-engine state resources.
  The upstream block-table row selection remains unchanged. Retain CPU-index
  selection only for additional align/cache-all state fields. Capture the
  legacy grouped-metadata alias once; explicit typed configuration wins.
- Keep target-sampling hooks and completed target logits. Native upstream
  routes precede local fallbacks; a completed projection is reused.
- Remove the duplicate local greedy MTP verifier. The fallback now calls
  the upstream verifier. Its guarded full-LM-head-GEMM top1 projection remains
  distinct from the native model top-token providers.
- Preserve the upstream multistep draft graph manager. Scoped padded-row
  routing covers single-step and multistep capture and restores router state
  after capture failure.
- Keep ordered warmup tasks and engine-owned configuration. Bind retained
  short-convolution metadata descriptors per engine and per backend builder.
  The generic-module layering baseline is reduced without new exemptions.

Broader guarded GDN/PLE paths, shared-gate and expert scratch fusion,
selected/padded draft rows, segmented grouped page4 attention, finite JIT
coverage and opt-in pacing remain in the patch. Their numerical and serving
qualification on this new base is still pending.

## CPU result and reproduction

On 2026-10-09, Python 3.12.13 with Torch 2.10.0+cu128 completed the exact
50-file scope: **1050 passed, 161 skipped, 0 failures, 32 warnings** in 28.44s.
The 64 changed files passed all applicable official pre-commit hooks,
including mypy, environment/configuration metadata and layering checks.
`scope.json` lists every test; `results.json` records counts and skip reasons.

Use a virtual environment with the repository's test dependencies:

```sh
.venv/bin/python docs/sm70-c73-cpu/run_cpu.py --junitxml=/tmp/e-cpu.xml
```

The bootstrap selects the real `CpuPlatform` in the pytest interpreter and
its spawned child interpreters. CUDA is hidden; model inspection uses local
test configurations. The implementation's native operators are unchanged.
Root conftest is excluded for this standalone owner-contract scope.
The portable wrapper above packages the recorded scope and CPU bootstrap;
its CLI has been checked, but it has not been rerun as an additional test suite.

These checks cover policy capture/hash, state/metadata CPU oracles, upstream
dispatch priority, logits reuse, finite warmup signatures and router scope
restoration. CPU capture stand-ins do not execute CUDA graphs. The 161 skips
are explicit CUDA, SM70 or native-ABI cases, not GPU passes.

## Validation boundary

Fresh native/source/ABI validation of `_C`, `_moe_C` and Flash-V100 is
required before GPU execution. The upstream delta from production D's base
includes 27 native/build files. E still needs GPU numerical/state/changed-input
and graph tests, MTP/function/quality checks, representative long-context and
concurrent performance checks, and ordinary serving-path acceptance.
Previous D GPU/performance evidence does not qualify E.

This was a single frozen-base refresh. Upstream subsequently merged #1140
and #1141, reaching `5dc280bb7f073f76c514a4696c5f434a0080c156` during the
final CPU checks. Those later graph/collective and configuration changes are
outside this snapshot. Draft status, hosted CI and human review remain required.
AI assistance was used; human review of every changed line is pending.
