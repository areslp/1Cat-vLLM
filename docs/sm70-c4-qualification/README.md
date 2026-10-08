# Fixed-main SM70 qualification evidence

The model source is `91377976388a60fd69aef87ac9190555f642b30b`, tree `3a3ff46c9aa3f1456e75b74171f654eb2e216d70`, on upstream `c4f6245f841466782752a8c3283e4727565cf17a`. The evidence branch adds documentation and synthetic records; it does not change that model source. AI assistance was used. Human every-line and end-to-end review remains required; this is draft review material.

## Review without a GPU

Only the Python standard library is required for the Python checks. They inspect internally consistent saved records, not a live tokenizer or GPU execution.

```bash
cd docs/sm70-c4-qualification
shasum -a 256 -c SHA256SUMS
mkdir data
tar -xzf observations.tar.gz -C data
uv venv --python 3.12 .venv-review
.venv-review/bin/python -m unittest discover -s . -p test_review_evidence.py -v
.venv-review/bin/python review_evidence.py data --output REVIEW.json
.venv-review/bin/python analyze_results.py data --output ANALYSIS.json
```

The eight small CPU tests cover matching work, missing arms, corrupted wire identity, early EOS, a forged fixed-budget flag forged throughput, signed/Unicode integer gold and performance-group parsing. Missing arms and early EOS never become timing gains. Full records include frozen messages, input-token hashes/counts, request-body hashes, actual usage, streams, phase clocks, resource counters and output text. The archive contains 184 synthetic fixtures and a 65-group matrix, including unmeasured tails. Private host paths, UUIDs, credentials, weights, service configuration and raw private logs are excluded; archive owner metadata is normalized.

## Scope and results

Main ran in the first window; the candidate ran in the second. Comparisons are **noncontemporaneous**, with independently fresh caches and the same frozen model/input/tokenizer/template, output budgets and serving contract. `PAIRING_POLICY.json`, both window records and native/source bindings preserve this distinction. Do not interpret the data as simultaneous A/B or isolated kernel measurements.

Main completed 51 groups. Candidate completed 44, one additional 128K repetition hit the remaining-budget timeout, and 20 were unmeasured. Intersection: 44 complete groups; main-only: seven; neither complete: fourteen. The partial timeout is excluded from gains. Of the 44 pairs, 22 performance groups completed identical fixed budgets: 74 requests and 33,664 output tokens on each arm. The remaining pairs are functional/quality checks. Wire, input/count, budget and throughput validation reports zero errors.

| Check | Main | Candidate | Interpretation |
| --- | --- | --- | --- |
| Short gold | 10/14 | 10/14 | Same four failures, also present in saved production C |
| Parallel gold | 4/8 | 4/8 | Repeats those same four failures |
| Retrieval through 261888 tokens | 9/9 requests | 9/9 requests | Seven context/concurrency groups |
| Tool-auto and constrained JSON | Both passed | Both passed | Ordinary unconstrained JSON failures remain visible |
| CPU source scope | Real ModelRunner eligibility: seven passed | 278 passed, ten CUDA-required skips | Seven skipped state cases subsequently passed on GPU; three specific JIT GPU tests remain unmeasured |
| GPU components | Not substituted for model comparison | 91 pytest cases, zero failures/skips; five H2 and five TP4 cases passed | Source-specific component evidence |

Four distinct short-gold failures are parentheses arithmetic, remainder, JSON fields and JSON boolean formatting. Request success is not answer correctness. Free generation chains frequently differ; no full token-parity or comprehensive model-quality claim is made.

| Performance workload | Cohorts | Median main-wall/candidate-wall ratio | Range |
| --- | --- | --- | --- |
| 1K, c1, 128 output tokens/request | 3 | 0.95269 | 0.94659–0.95580 |
| 1K, c4, 128 output tokens/request | 3 | 1.15167 | 0.99047–1.18041 |
| 1K, c8, 128 output tokens/request | 3 | 1.34557 | 1.18285–1.34940 |
| 1K, c8, 1024 output tokens/request | 1 | 1.49441 | Single cohort |
| 32K, c8, 1024 output tokens/request | 1 | 1.14414 | Single cohort |
| 64K, c8, 1024 output tokens/request | 1 | 1.15450 | Includes KV-capacity queueing |

`paired_metrics.csv` and `ANALYSIS.json` separate group throughput, request wall/TTFT tails, prefill/decode/queue phase means and MTP counters for every eligible pair. Sample p95 values use nearest rank and are descriptive: eight-request p95 is effectively the maximum, not a production tail estimate. SSE gaps are not per-token ITL. Cold JIT samples are retained; the third 1K/c8 cohort remains approximately 18.3% faster than its matching main sample despite JIT spikes.

The 1K/c1 regression is retained. Median prefill is 0.32276/0.32263 seconds (main/candidate), decode 1.74221/1.84913 seconds, with negligible queueing and no cache/preemption explanation. Each candidate short cohort uses six additional MTP draft rounds and accepts fewer tokens; different generation chains prevent acceptance-equivalent compute attribution. Decode-phase wall divided by draft rounds is about 4.8% lower, consistent with more rounds explaining the longer total, but not proof of kernel causality. Across the 22 fixed-budget pairs, MTP acceptance is 43.27%/44.02% and mean acceptance length 2.731/2.761; this mixed aggregate does not erase the c1 result.

Last per-rank compute-memory snapshots are 31,338 MiB each for main and 31,234/31,236/31,234/31,234 MiB for candidate. These are different final workloads, not peak or same-group measurements; no memory-saving claim is qualified.

## Source stages and coverage

| Commit | Upstream implementation retained first; remaining gap | Evidence | Unverified boundary |
| --- | --- | --- | --- |
| `17688b131` | Ordinary scheduler plus separate opt-in fixed pacing | CPU pacing/configuration tests; production pacing zero | No enabled-pacing model gain claimed |
| `60e504df5` | Adds broader guarded SM70 GDN/fusion primitives | GDN 13, fusion five, state seven GPU cases; CPU dispatch contracts | No per-family model timing attribution |
| `be69ceed3` | Upstream mixed-QKV, strided verification and direct-output GDN retain exact admission priority | Real-metadata route CPU tests plus numeric/state cases | Unsupported shapes keep ordinary fallback |
| `d3dd57dce` | Upstream request indices remain authoritative; broader speculative GDN/shortconv metadata and state contracts retained | GPU state and fusion cases; CPU accepted-token/metadata guards; end-to-end MTP4 | Numerical coverage is scoped, not all layouts |
| `a565c114c` | Upstream admitted small ngram path first; broader output-buffer/stride fallback | PLE 13 GPU ID cases, CPU dispatch, fusion/state tests | No claim of a new universal PLE route |
| `960f83796` | Upstream shared-gate admission first; separate standalone gate and NVFP4 scratch/row-map ABI | Fusion cases and five synthetic H2 scratch/changed-input graph cases | H2 oracle is E192/K8/H2560, not exact model E512/K10; slot limit 1024 retains upstream fallback |
| `9f76e53be` | Upstream compact head/local-top1 first; selected/padded rows, graph declines and full-logits fallback remain guarded | CPU head/prefill scope; five TP4 NCCL packet cases, finite/ties/NaNs/zero/minus-inf and changed-input replay; model MTP counters | Synthetic NCCL packet test does not qualify every vLLM transport; no chain parity |
| `71e34c0ab` | Upstream packed attention fallback retained; grouped page4 request/query segmentation is separate | Fresh Flash-V100 build, page4 40 plus QSA/FP64 13 GPU cases; long contexts and concurrent decode | No arbitrary geometry or isolated model-family speed claim |
| `08157fb29` | Upstream auxiliary warmup retained; finite metadata coverage and bounded JIT manifests/telemetry | CPU scope, actual startup/cache writing and retained JIT events | Three CUDA/Triton integration unit cases unmeasured; finite warmup does not eliminate all cold JIT |
| `913779763` | Real upstream live-prefill eligibility executes before the residual draft-list graph guard | Seven real ModelRunner CPU cases on main and candidate | CPU eligibility is not GPU graph replay qualification |

Experimental E7 is production-off and deferred; its complete implementation remains on the local archival ref. It is not claimed replaced by upstream. Open PR1007 overlaps graph/materialization callers and needs maintainer coordination; this refreshes existing PR903 rather than opening a duplicate. Related PR1039, PR961, PR983 and PR828 do not supply these scoped residuals. Upstream code on the frozen base remains first where equivalent.

## Public component reproduction

After building source `913779763` and its SM70/native dependencies using the repository instructions, the six component commands are recorded in `data/COMPONENT_RESULT.json`. They exercised:

- `tests/models/qwen4_exp/test_sm70_gdn_verify_fused.py`: 13 cases.
- `tests/models/qwen4_exp/test_production_fusion_equivalence.py`: five cases.
- `tests/v1/attention/test_gdn_spec_state_contract.py`, selection `cuda or sync_debug`: seven cases.
- `tests/kernels/test_sm70_ple_ngram_batch.py`: 13 cases.
- `tests/kernels/attention/test_sm70_segmented_page4.py`: 40 cases.
- `tests/kernels/attention/test_sm70_grouped_e4m3_fp32_multi_kv_head.py`, selection `multi_kv_head`: 13 cases; the filename match includes the complete file.

The recorded tests were copied to an isolated test directory with `--confcutdir` pointing there to avoid unrelated root fixtures; operator imports used the exact fresh source. The numerical inputs and oracles are in those public test files. With the built source/dependencies available, additional model-free checks are:

```bash
SM70_REPRO_OUTPUT=./h2-output .venv/bin/python docs/sm70-c4-qualification/component_repro.py h2
CUDA_VISIBLE_DEVICES=0,1,2,3 SM70_REPRO_OUTPUT=./tp4-output MASTER_PORT=18344 \
  .venv/bin/python docs/sm70-c4-qualification/component_repro.py tp4
.venv/bin/python docs/sm70-c4-qualification/run_cpu_scope.py \
  -q tests/v1/worker/test_gpu_cudagraph_prefill_state_priority.py
```

The public component helper adapts output handling, binds loop-local tensors as function defaults, uses the required accelerator synchronization/device API, and retains the corrected NCCL teardown scaffolding; arithmetic assertions match the tested helper. These public scaffolding/formatting adaptations were not a new GPU run. It resets the captured graph before barrier/destroy; the first numerical TP4 attempt passed values but hung in cleanup, and its typed result is retained alongside the clean second attempt. No model weights are needed for these component tests. Exact operational model timings require the frozen private NVFP4 artifacts, so the timings are not a self-contained public model benchmark.

## Runtime and remaining limits

Fresh build/import bindings cover the source trees and ten native artifacts. Worker map snapshots explicitly record the four common loaded DSOs on all four ranks. Flash-V100's imported package binary and build-output alias have the same recorded SHA and ABI, with GPU component and model route evidence; the worker snapshot records canonical build paths and does not independently enumerate the package alias in maps. No blanket claim that all ten DSOs were mapped in every worker is made; several artifacts are optional.

Both retiring-worker identity checks interrupted the test controller. Opening resumed only when the original workers were absent and GPU contexts empty; deadlines were not reset and no unknown process was killed. At final stop the independent recovery service took over. These control-plane events are separate from model source, and their exact transient identity cause was not captured. Original production C was restored and fingerprint/real-entry/small direct acceptance passed; the temporary timer was removed. No permanent candidate deployment or upstream merge occurred.

Not qualified: all 65 paired groups, exact E512/K10 standalone H2 oracle, three specific JIT GPU unit cases, multimodal numerics, peak-memory profiling, per-family timing attribution, exhaustive transport/layout coverage, private-model public reproducibility, human every-line review and new-head CI. The existing historical CI trust failure is distinct from local checks; no trust label or bypass is applied.
