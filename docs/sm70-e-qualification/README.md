# SM70 MTP residual qualification

The complete fixed-compiler results are in [RESULTS.md](RESULTS.md). All 26 historical C1 inputs improve against the control with identical output and MTP work; the old-D differences remain explicit.
The earlier automatic-tuning runs are historical observations: a restarted U
changed output or MTP work in 25 of 26 C1 inputs. They do not establish a clean
patch-only performance benefit and are not pooled with the fixed runs.

The final implementation extends upstream main `16628e2f0a70761ba6525bf46e5e844c04b3e66f`, at commit `69801806572ea094b91d5f18864beb22c6c8b6fb`. Upstream PR1160 landed after the frozen matrix and replaces our equivalent local report serializer. GPU computation/native source is unchanged; [final-source-equivalence.json](final-source-equivalence.json) records the exact differences and CPU equality checks. The frozen matrix uses the exact commits in `runtime-commits.json`.
The control is **frozen upstream plus common compatibility fixes**, not
unmodified main. It includes the reporting-only fix required to initialize
an internal MTP draft after target MoE policy binding, and the SM70 fix that
preserves explicit Inductor combo configuration. Without the latter, platform
defaults overwrite `benchmark_combo_kernel=false`, causing deterministic
compilation to fail during startup. Full common commits and diffs are in
`common-compatibility-fixes.json` and `common-compatibility-fixes.patch`;
their correctness is recorded separately from local patch speedups.
Exact serving commits are in
`runtime-commits.json`. Both arms keep the same model files, global FP16 dtype,
TP4, MTP4, FP8 E4M3 KV cache, and PCIe communication settings.

## Reproduce the serving comparison

Use four PCIe Tesla V100 32 GB GPUs and a compatible Qwen Flash-Next NVFP4
checkpoint. Keep the identical model directory for every arm. Checkpoint
weights are not distributed here. The saved input token hashes detect a
different tokenizer or chat template. Results apply to the tested checkpoint;
they are not model-independent quality or speed guarantees.

Build each source checkout normally, including its SM70 `_C`, `_moe_C`,
Flash-V100, FA2 and bundled FlashQLA extensions. The required native policy ABI
is 67, runtime ABI 1, FA2 policy ABI 1 and FlashQLA policy ABI 1. The reference
environment uses Python 3.12.13, PyTorch 2.10.0+cu128, CUDA 12.8.93 and GCC 13.3.
Use a virtual environment with the repository dependencies and
`prometheus_client`, `regex` and `pybase64` installed.

Start one arm at a time, using ordinary separate compiler caches:

```sh
.venv/bin/python serve.py --arm U --source "$CONTROL_CHECKOUT" \
  --model-dir "$MODEL_DIR" --cache-root "$BENCH_CACHE"
```

From another shell in this evidence directory, after the service is healthy:

```sh
.venv/bin/python run_suite.py U-fixed-warm.json
.venv/bin/python run_suite.py U-fixed-timed.json
.venv/bin/python run_suite.py U-fixed-cached.json
```

Run quality while U is serving, then stop U, start E and use the corresponding
E plans. The fixed comparison has three repetitions of 37 cold groups, with
group order reversed for the second repetition. Each arm has one startup.
Each result label must have a fresh output directory; preserve an earlier
attempt before retrying. No custom kernel loader or allowlist is used.

`serving-configuration.json` passes the ordinary PyTorch 2.10 deterministic
compiler configuration and disables pointwise, coordinate-descent and combo
benchmark selection. It also sets both vLLM automatic-tuning environment
variables to zero, because single-size compilation otherwise overwrites two
of the CLI settings. Combo fusion remains enabled. This controls compiler
choices without changing the model, dtype or runtime kernel loader.

`previous-serving-configuration.json`, the old a/b plans and
`align_kernel_choices.py` document the earlier attempt. Copying saved choices
alone did not control new AOT cache namespaces created on restart. Do not use
that old procedure as evidence of fixed actual kernel selection.

After timing, inspect the saved graph configurations and their referenced
Triton binaries with the same Torch environment:

```sh
CUDA_VISIBLE_DEVICES= .venv/bin/python inspect_compiler_choices.py \
  --cache-root "$BENCH_CACHE" --output compiler-choice-review.json
```

This reads the comparison's own serialized Torch 2.10 artifacts. It reports
common helpers, serialized candidates and binary/metadata digests for each
rank and both target/draft roles. A multi-candidate record does not identify
the candidate actually launched. It does not trace live kernel calls or
alter loading. Use a fresh cache root when reproducing; a directory containing
several compiler namespaces requires selecting the namespace for that run
with `--u-namespace` and `--e-namespace`.

The hardware-specific NUMA mapping and other serving options are explicit in
`serving-configuration.json`. Adjusting them produces a different experiment.
Serving should be otherwise idle. The benchmark uses unique request cache
salts, verifies zero cached tokens and full computed-KV work, completes the
fixed output budget, and checks that every request contributes to every phase
metric. It does not clear the production prefix cache or infer cache misses
from prompt length alone.

The separate `U-fixed-cached.json` and `E-fixed-cached.json` plans prime
32K/64K prefixes once and then repeat them three times. Priming
is excluded from the cached comparison. Each measured reuse must have a
positive cached-token count, and cached plus computed tokens must equal the
input length. These results are separate from the uncached cohorts.

Run `analyze_fixed.py` after placing result directories under `results/`.
It separates prefill, queue, TTFT, decode and completion time and reports all
26 historical C1 cases, output identities, MTP rounds and acceptance.
`analyze_performance.py` and `analyze_slow_cases.py` reproduce the earlier
automatic-tuning results, retained separately. Decode duration per MTP round is
a serving-work proxy, not an isolated GPU kernel measurement. Small-sample
bootstrap intervals are descriptive and do not establish production p99.

Long-context groups use 32K/64K input tokens at concurrency one. Concurrent
groups use 1024 input tokens at concurrency four or eight. This matrix does
not claim an eight-request, 64K-per-request result.

## Component correctness

Run the same synthetic quality, protocol, concurrent-isolation, long-retrieval
and media checks while each arm is serving:

```sh
.venv/bin/python run_quality.py --arm U-fixed --url http://127.0.0.1:18200
# After collecting the corresponding E and reference results:
.venv/bin/python analyze_fixed.py
```

These requests allow natural completion; their times are not throughput
samples. The runner retains every golden-answer failure for per-case
comparison instead of stopping after the first baseline model error. The
synthetic image is generated locally, and the included one-second red video
is synthetic. Install Pillow and the server's ordinary video dependencies.
`quality-observations.json` contains the exact tested request inputs and golden
answers. Input-token and request digests remain available with the results.

The source tree contains synthetic GPU tests for segmented/grouped attention,
native M1/M5 admission, changed-input graph replay, GDN/PLE fusion equivalence,
short-convolution metadata and drafter state isolation. The commands and
per-test outcomes are attached in `test-results.json`.
Component passes alone do not establish a model-level speedup.

`test-results.json` lists the focused test commands and each final case
outcome. The CPU union has 1,476 distinct passes, one inactive-GGUF hash failure also
reproduced on unmodified main, and 177 CUDA-dependent skips. The separate
architecture suite passes 17 checks. The GPU-environment component stage has
244 distinct passes, including some host-side admission/ownership checks.
The common combo-configuration fix has a separate 173-pass CPU run on each
arm, including five settings cases; the four explicit-setting combinations
failed against the original platform defaults. No GPU or model was loaded for
those CPU checks.

AI assistance was used. Public artifacts exclude raw operational logs, private
host paths, credentials and model weights.

## Final deployment status

E accepted and serving. See [final-deployment.json](final-deployment.json).
