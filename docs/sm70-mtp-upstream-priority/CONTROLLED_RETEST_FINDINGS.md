# Whole-graph configuration control: incomplete retest

The 2026-10-09 bounded retest produced **zero qualified performance pairs**.
The old-route arm completed 81 groups; the corrected-route arm did not reach
API health and completed zero timed groups. It reported an NCCL unhandled
CUDA error on worker TP1 during post-capture warmup. The earlier natural
corrected-source trial completed all 39 groups. This new diagnostic-policy
startup failure does not isolate a PR, native projection, configuration or
hardware cause. The previous 17 slower observations remain unresolved.

Both arms used source `96cedecef355cf1a4865996080ad75021da9d9b7`, the same
source path, version, flags, native artifacts and unchanged weights. The
experimental old route overrides M1 to `(2,64,64,2,4)`; the corrected route
retains upstream `(2,128,64,4,3)`. Their common vLLM compilation namespace is
`118347e9e4`. Independent AOT/Inductor roots were initially empty. Copies of
common Triton/CPP caches and one predeclared 72-entry launch policy did not
freeze the resulting whole compiled graphs.

## What remained different

The observer recorded 42 unique `CachingAutotuner.run` source/config
signatures per rank per arm. These observations can include compilation
benchmarks; they are not direct CUDA-graph replay traces. A separate static
audit follows all 80 actual artifact references, comparing 40 pairs of
generated runtime and compile-time benchmark sources.

| Rank | Equal ordered signatures | Old known-policy coverage | Corrected known-policy coverage |
| --- | --- | --- | --- |
| 0 | 42/42 | 19/19 | 19/19 |
| 1 | 39/42 | 19/19 | 17/19 |
| 2 | 42/42 | 19/19 | 19/19 |
| 3 | 39/42 | 19/19 | 17/19 |

All remaining known-policy kernels match their declared source/config. Four
original known kernels are absent on the corrected side; they must not be
reported as matched or ignored to make alignment pass. Runtime and benchmark
ASTs match in **38/40** pairs. The two different pairs are the first drafter
subgraph on ranks 1 and 3. Its three changed helper/reduction sources are
referenced by real generated runtime code as well as benchmarks.

| Operation on ranks 1 and 3 | Old route | Corrected route |
| --- | --- | --- |
| Embedding helper | Combo wrapper | Standalone generated kernel |
| FP32 RMS reduction, extent 10240 | X=1, R=2048, 16 warps, 1 stage | X=1, R=16384, 32 warps, 1 stage |
| FP32 RMS reduction, extent 2560 | X=1, R=4096, 16 warps, 1 stage | X=1, R=2048, 16 warps, 1 stage |

Operations are paired by mathematical extent, not event ordinal: the two
reductions exchange ordinals. Comparing ordinal 35 alone would incorrectly
describe the 2560-element and 10240-element reductions as the same operation.
Changed floating values have not been measured in this trial.

The compiler also benchmarks **combo versus standalone generation**. Pinning
launch choices alone leaves this additional decision free. Moreover, the
diagnostic's unknown-helper fallback sorted JSON launch kwargs lexically;
that can order 16384 before 2048. It was deterministic but did not ensure
equivalent runtime work or a suitable launch choice. This experimental policy
is excluded from the model PR and production. No source optimization is
justified by this incomplete comparison.

The old-route arm's 78 c1 observations cover 26 distinct inputs with three
rotating repetitions. Its text and draft counters are stable within the arm;
both selected short gold probes passed. These are unpaired observations.
The three planned live-activation/token-chain diagnostics were not executed.
Old-route M1 uses Triton as observed during inspection; corrected-route M1
admission was not inspected successfully in this trial. The shared startup
M1/M5 message cannot substitute for that inspection.

## Corrected control for the next trial

Use one complete compiled seed from the successful old-route arm, bind it to
the unchanged source/path/version/compiler namespace, and make separate
copies for both arms. Freeze **all 40 generated graphs** as well as complete
kernel configurations, including CTA and register settings. Reject any
unqualified kernel, changed source, unavailable choice or unexpected
compilation before collecting performance observations.

Plain-text compile indexes can be relocated after static parsing; opaque
artifacts must not be edited by unpickling them. Retained absolute references
need byte validation and a read-only mount in each experimental child. Copying
an index alone does not relocate embedded artifact references. The original
task cache and common seed must remain immutable during both arms. Direct
all-rank route/coverage inspection, generated-source checks and observer
removal still need live validation before timing. CPU preparation does not
claim successful fixed-route startup or a completed paired test.

Simply passing `benchmark_combo_kernel=False` on the CLI is insufficient in
this frozen source: the SM70 Flash-V100 configuration finalizer unconditionally
sets both `combo_kernels` and `benchmark_combo_kernel` to true. The proposed
complete-seed control avoids treating that flag as effective. An alternative
that overrides the finalizer is another diagnostic policy requiring its own
GPU qualification.

## Outcome, recovery and reproduction

The controller paused the task driver on the alignment failure. Its raw
unfinished JSON has an empty failure list; it is not a final success record.
The separate final outcome records the startup error and failed alignment.
The independent remote deadline restored original C **once**, without waiting
for the Mac. C's frozen 2098-artifact/217-weight-record checks, clean Git tree,
health and three real Mac-entry checks passed. Recovery and Mac acceptance
completed inside the reserved recovery budget. No candidate was permanently
deployed, no weights were changed, and no upstream PR was merged.

The standard-library portable audit reproduces the 162/168 equal signature
count, 38/40 source-pair equality and zero admitted performance pairs:

```sh
.venv/bin/python -m unittest discover -s docs/sm70-mtp-upstream-priority -p test_controlled_retest.py -v
.venv/bin/python docs/sm70-mtp-upstream-priority/audit_controlled_retest.py --observations docs/sm70-mtp-upstream-priority/CONTROLLED_RETEST_OBSERVATIONS.json --out /tmp/controlled-retest-audit.json
```

Eight CPU tests cover incomplete drivers, changed configurations,
missing/repeated ranks or graphs, runtime and benchmark differences, startup
failure, missing gold/groups and independent request validation. They do not
run a model or qualify GPU code.
Raw service logs, generated full sources, weights, host/cache paths, process
identifiers and tensors remain private. Keep draft #903's source correction
and historical results separate from this failed controlled retest.
