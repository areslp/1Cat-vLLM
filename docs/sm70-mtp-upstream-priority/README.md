# Preserve upstream MTP projection admission

An integration defect in measured draft head
`91377976388a60fd69aef87ac9190555f642b30b` changed the Qwen3.8 TP4 drafter's
M1 tile from the upstream `(2, 128, 64, 4, 3)` to `(2, 64, 64, 2, 4)`.
The tuple means block M, block N, block K, warp count and stage count.
The existing upstream exact FP16 projection requires block N 128 and four
warps, so this local tuning silently makes both M1 expert projections use
Triton. M5 remains eligible for the native operator; the startup message
mentioning M1/M5 does not distinguish these calls.

The integration introduced the override in
`960f837963a99cd8b09a5b4dbe715a1870b84f3a`. The native implementation is already
upstream in `0930fd3b6d4c804bfb04abbdabf779108ffcb824`; its arithmetic and CUDA
source are unchanged. Correction
`96cedecef355cf1a4865996080ad75021da9d9b7` retains the upstream M1 tile and
leaves the additional local batch-size tuning intact. This is a correction
to the existing draft PR, not another native-kernel proposal.

## Evidence

The original configuration test checked M1 only if the local table already
contained the expected upstream tile. That condition skipped precisely the
defect. Its assertion is now unconditional.

The new CPU tests exercise the real `get_default_config` and
`dispatch_fused_moe_kernel`, covering M1/M5, both expert projections, and the
native enable/disable switch. Tensor metadata uses the meta device and
kernel calls are spies; no CUDA allocation or model loading occurs.

| Source | CPU result | Qualified M1 native calls |
| --- | --- | --- |
| Original PR model code with strengthened tests | 104 pass, 3 fail | Both expert projections incorrectly use Triton |
| Corrected model code | 107 pass, 0 fail | Both use native when enabled; explicit disable still uses Triton |

All applicable source pre-commit checks passed. Raw host logs stay private.
`INVESTIGATION.json` includes source hashes, admission guards, CPU counts
and the original three fixed-budget 1K single-request regression cohorts.
The earlier throughput loss remains an observation of the original head.

## CPU source-contract reproduction

Use a Python 3.12 environment managed by uv. The following audit uses only
the standard library; it does not import Torch or initialize a device.

```sh
mkdir -p /tmp/mtp-priority-audit
git show c4f6245f841466782752a8c3283e4727565cf17a:vllm/model_executor/layers/fused_moe/fused_moe.py > /tmp/mtp-priority-audit/upstream.py
git show 91377976388a60fd69aef87ac9190555f642b30b:vllm/model_executor/layers/fused_moe/fused_moe.py > /tmp/mtp-priority-audit/original.py
.venv/bin/python docs/sm70-mtp-upstream-priority/audit_dispatch.py /tmp/mtp-priority-audit/upstream.py /tmp/mtp-priority-audit/original.py vllm/model_executor/layers/fused_moe/fused_moe.py
```

At M1, the native tile guard is admitted for upstream, rejected for the
original integration, and admitted for the correction. M5 is admitted in
all three. This audit checks only tile admission: the real dispatcher also
requires the device, tensor shape/dtype/layout, operator and policy guards.

The actual runtime CPU routing tests are in
`tests/kernels/moe/test_sm70_mtp_upstream_dispatch.py`; they require the
project's import dependencies but no GPU. They replace only device metadata
and the final kernel invocation, so the production selector and dispatcher
remain the code under test.

## GPU retest

The old head and correction were tested sequentially in one finite window.
Each completed all 39 groups and 69 requests, with 37 fixed-budget performance
pairs, 67 requests and 9,472 output tokens per arm. Request hashes, tokenized
input hashes/counts, budgets, output counts and throughput recomputation have
zero validation errors. The two prequalified short gold probes passed on both
heads. This is a narrow regression retest, not comprehensive model admission.

| Cohort | Pairs | Median corrected/old throughput ratio | Range |
| --- | --- | --- | --- |
| Original three 1K/c1 regression prompts, 128 output | 3 | 1.11142 | 1.05011–1.14366 |
| All 30 1K/c1 observations (26 distinct inputs), 128 output | 30 | 0.98753 | 0.84964–1.30588 |
| 1K/c4, 128 output/request | 3 | 1.00733 | 0.99518–1.01768 |
| 1K/c8, 128 output/request | 3 | 1.01668 | 1.00994–1.01947 |
| 1K/c1, 1024 output | 1 | 1.10513 | Single sample |

The original three prompts use 174/155 draft rounds and accept 210/228
speculative tokens (old/corrected). Their median prefill is 0.32154/0.32222s;
median decode is 1.84942/1.65121s. The expanded c1 cohort has **13 gains and
17 losses**, total wall 61.12166/61.24466s, and 1597/1603 draft rounds.
Its total time is essentially unchanged; the correction does **not** show a
uniform c1 performance gain. Most generated texts differ. Pure upstream main
was not rerun, so the retest cannot establish general main/correction parity.
The additional 27 c1 requests repeat earlier concurrency inputs; their wire
bodies remain identical between arms. A later full counter audit confirms
zero prefix-cache hits, zero cached prompt tokens and zero preemptions in all
30 c1 observations. Four input pairs are duplicates, so there are 26 distinct
c1 inputs; the 30 observations are not 30 independent prompts.

Six input scales, invalid expert routes, four real checkpoint TP weight slices,
M1/M5 and both expert projections were checked: **16 cases, 96 scale checks**.
Native versus upstream-tile Triton is bitwise equal throughout; the old
64-column Triton tile also matches on these inputs. The corrected real
dispatcher invokes the native operator in all 16 cases. All four weight slices
were run sequentially on one V100, not four simultaneous device trials.
Projection timings alternate three implementations, separately from HTTP
requests; M1 old/native median time ratios across TP slices are 1.0350 for
up-projection and 1.8747 for down-projection. These are individual projection
observations, not complete draft-round speedups. The M5 64-column control is
diagnostic only: the old production head already uses native M5.

## Reproduce the saved paired analysis

Use the project environment and standard-library analysis; no model or GPU is
needed for these checks.

```sh
.venv/bin/python -m unittest discover -s docs/sm70-mtp-upstream-priority -p test_analysis.py -v
.venv/bin/python docs/sm70-mtp-upstream-priority/analyze_retest.py --root docs/sm70-mtp-upstream-priority/paired-observations --out /tmp/mtp-retest-analysis.json
```

Six admission tests reject early EOS, incomplete streams, unequal wire/input,
wrong prompt/output counts and inconsistent throughput. The saved compact
observations reproduce `RETEST_ANALYSIS.json` exactly. Raw SSE chunks and
private service/native-map/control logs are preserved outside this branch.
`synthetic-fixtures.jsonl` contains only public synthetic requests.

For live kernel checks, build the source's normal SM70 extension and use a
compatible checkpoint with the same MTP expert layout. The exact frozen
operational checkpoint artifacts are private; timings are not a self-contained
public model benchmark.

```sh
.venv/bin/python docs/sm70-mtp-upstream-priority/benchmark_routes.py --model /path/to/compatible-checkpoint --out /tmp/mtp-projections.json
```

## Attribution and qualification limits

The native admission defect and corrected routing are confirmed. It is
incorrect to attribute all end-to-end changes to the one tile change.
Although the arms started from isolated copies of the same cache, their
actual vLLM compilation namespaces differ and both performed compilation.
Upstream `_compute_backend_code_hash` includes traced file paths as well as
contents, so an isolated source directory changes this cache factor even
for unchanged forward source. `VllmConfig.compute_hash` also includes the
generated commit-bearing vLLM version; the exact version stamps differ. Cache copying therefore does not prove aligned
runtime choices. Cached and actual launch selections require separate audit;
normal independent autotuning is not pinned into production by this fix.
`CACHE_NAMESPACE_AUDIT.json` records equal environment/compiler factors and
equal comment-free FX ASTs for all eight rank/model pairs, with unequal
code/config hashes. Equal FX graphs do not prove identical runtime arithmetic.

The native/old-tile bitwise observations also do not explain the changed
accepted-token chains by themselves. Selected-row MoE execution and valid
compiled reduction choices remain possible contributors. Historical
controlled diagnostics showed different-M drafter post-MoE values, but used
other commits; they do not isolate the current cohort's cause.

Keep the performance observations and these limits together. No broad quality,
long-context requalification, exhaustive operator parity or universal speedup
is claimed. Production was restored to the previously qualified C after this test;
no correction is permanently deployed. No upstream PR is merged or marked
ready. AI assistance was used; submitting human review remains required.

## Expanded c1 follow-up (earlier CPU/read-only audit)

The 17 slower observations were audited individually. Sixteen use more draft
rounds (57 extra rounds in total); their extra decode wall is 1.78710s. An
exact symmetric work/cost decomposition assigns +1.81495s to extra rounds and
−0.02785s to changed per-round cost. The remaining observation has identical
text, draft and acceptance counts and is 0.705ms slower (0.038%); a single
measurement does not establish a repeatable regression. The per-draft ratios
for all slower observations range from 0.99653 to 1.00042. These are request
phase wall/work diagnostics, not isolated GPU timing.

The four duplicated input pairs reproduce identical text and draft counts
within each arm. After averaging each duplicated case, 26 distinct inputs
have a median corrected/old ratio of 0.98744: 14 cases with extra draft work,
one equal-work timing difference, and 11 gains. Twenty-eight of the original
30 output texts differ between arms, so a fixed token budget did not hold the
generation/acceptance path fixed. `EXPANDED_C1_FINDINGS.md` lists every slower
observation; `analyze_samples.py` reproduces `SAMPLE_ATTRIBUTION.json` using
the existing portable saved observations. Per-position acceptance counters
are in `MTP_ACCEPTED_POSITIONS.json`, with pre-budget-clipping semantics.

A static audit followed the actual saved compile handles for four ranks,
target and drafter. It compared 80 referenced artifacts (40 pairs), including
40 old handles pointing outside their copied active directory; those original
bytes match the copied artifacts exactly. The lowered runtime and compile-time
benchmark ASTs match for all 40 pairs after ignoring comments and line
locations, including comments inside embedded Triton source. Torch digests
and corresponding candidate-configuration hashes match.

Of 188 saved autotune entry pairs, 53 have different launch kwargs or
warp/stage counts. Ninety-two entry pairs map to generated runtime-source
references; 25 of those differ, including eight reduction entries. The
remaining unmapped entries can be compile-time candidates and are not counted
as runtime launches. Target Q/K normalization differs on three ranks; drafter
Q/K normalization differs on all four, and another drafter RMS reduction
differs on rank 0. For example, target rank 0 Q/K reduction changes
`(XBLOCK,R0_BLOCK,warps)=(16,32,4)` to `(64,64,16)`.

This establishes a concrete saved-configuration confound for the current
comparison. It does **not** observe actual launches/CUDA-graph selections or
prove those choices caused a particular output difference. Projection parity
on the previously tested inputs still does not prove parity on every live M1
activation. No speculative code fix, cache policy, model load or new GPU
experiment follows from this static evidence; current-cohort attribution
needs a separate same-input, same-configuration control. Production stays
on the previously qualified C. The source correction head is unchanged.

```sh
.venv/bin/python -m unittest discover -s docs/sm70-mtp-upstream-priority -p "test_*.py" -v
.venv/bin/python docs/sm70-mtp-upstream-priority/analyze_samples.py --root docs/sm70-mtp-upstream-priority/paired-observations --out /tmp/mtp-sample-attribution.json
.venv/bin/python docs/sm70-mtp-upstream-priority/audit_saved_artifacts.py --observations docs/sm70-mtp-upstream-priority/COMPILED_OBSERVATIONS.json --out /tmp/mtp-compiled-selection-audit.json
```

The five new tests verify that normalization ignores comments without erasing
changed arithmetic, audits benchmark code separately, maps binary Torch cache
keys/tags correctly, and never relabels saved choices as observed launches or
causality. Portable metadata reproduces the compiled-selection summary; raw
generated source and absolute cache/host paths remain private.

## Subsequent whole-graph control attempt

The authorized 2026-10-09 common-configuration retest produced **zero valid
performance pairs**. The old-route arm completed 81 groups, including three
repeats of all 26 distinct c1 inputs; the corrected-route arm failed post-capture
startup with an NCCL/CUDA error and completed no timed groups. Both used the
same source/path/version and a common launch policy, but combo-versus-standalone
generation differed on two drafter graphs: 38/40 runtime/benchmark AST pairs
match and 162/168 ordered observed source/config signatures match. Missing
original kernels and additional defaults invalidate whole-graph alignment.
This diagnostic startup failure does not prove a native-projection or PR cause.

[CONTROLLED_RETEST_FINDINGS.md](CONTROLLED_RETEST_FINDINGS.md) records actual
coverage, reduction extents, the control defect and the proposed complete-seed
control. The portable observations and eight CPU admission tests reproduce the
failed gate. The original 17 slower observations remain unresolved; no new
paired speedup or broad quality claim is made. Original C was restored once
inside the reserved budget and its real Mac-entry checks passed. The corrected
source head remains unchanged and is not permanently deployed.
