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

## Qualification limits

The integration defect is confirmed and its CPU routing fix is tested.
GPU numerical and end-to-end performance qualification of the correction
is pending. Restoring the native path has not yet been shown to eliminate
the whole 4.7% throughput loss or to restore MTP acceptance. The original
three requests were greedy, had identical inputs/output budgets, and all
used six more draft rounds in the integrated arm, but their generated text
also differed. Source inspection and wall time divided by draft count do
not isolate the effect of this one patch.

Other possible contributors still include different valid compiled launch
choices in independent caches and the drafter's selective prefill row
count. A controlled old/corrected comparison with identical weights and
isolated copies of the same cache, plus alternating projection checks on
identical weight slices/inputs, is prepared. Production remains on the
previously qualified C while that qualification is pending. No upstream PR
has been merged or marked ready. AI assistance was used for this audit,
the correction and the tests; submitting human review remains required.
