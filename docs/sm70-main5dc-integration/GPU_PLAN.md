# Proposed follow-up: not executed

Compare pure upstream `5dc280bb7f073f76c514a4696c5f434a0080c156` with the
exact final PR903 CPU integration commit. Record both full SHAs and clean
source/native manifests before any future authorized GPU window. Keep the
same model weights, tokenizer, TP4/FP16/MTP4 configuration and request bytes.
This plan does not authorize a service stop, cache change or GPU reservation.

1. Build the configured collective native ABI from each arm. Check real
   imports, policy field ordering, pointer owner, fallback, TP4 reductions
   and graph capture/replay. Requalify MTP/GDN/PLE/attention components on
   the new source. Preserve old E and D for provenance and rollback.
2. Warm ordinary code and all selected shapes with separate arm caches;
   record active role choices and compiled references. Keep first-use/JIT
   work separate. Match choices only where helper/source/candidate hashes
   agree. Record unmatched choices instead of installing loader overrides.
3. Repeat C4, short C8 and longer-decode C8 with both U→E and E→U orders,
   at least three completed pairs per order when the authorized window
   allows. Repeat the same arm without changing input to measure its own
   variability. Preserve every attempted pair, finish reason and budget.
   Report paired distributions and order effects, not only the best run.
4. Collect raw request phase counters and cache snapshots as described in
   the README. In a separately labeled diagnostic run, collect actual
   scheduler/runner `num_reqs`, `num_tokens`, batch descriptors and ordered
   request-to-slot maps. Use the existing opt-in step profiler for target
   forward, draft and CPU wall measurements; retain its averaging interval
   and overhead. Record accepted/drafted tokens, draft rounds and each
   measured step's cost. Aggregate Prometheus engine iteration counts
   alone cannot recover actual batches or per-round GPU cost.
5. For a small set of diverging outputs, first check same-arm repetition.
   Teacher-force the identical generated prefix on both arms and inspect
   only the first differing target decision: logits/top candidates, margin,
   draft proposals and verifier acceptance. Separate the first divergence
   from later work amplification. Do not attribute upstream's own
   warm/timed variation to E.
6. Run a small marker/copy cohort with distinct markers per request. Reverse
   row order, stagger completions, then reuse released slots with new markers.
   Compare outputs against their own input and inspect request/slot maps
   for cross-talk or stale state, including changed-input graph replay.
7. Measure full long prefill separately: code warm, prefix cache empty,
   32K and 64K C1. Reset only the idle task endpoint before each request,
   verify `prompt_tokens_cached_total` delta is zero, and verify computed
   prefill KV tokens equal the full prompt. Keep the original cached
   32K/64K cohorts separate. 32K C4, 256K and MTP-off remain additional
   unqualified scopes unless explicitly included in that later window.

Use the existing bounded maintenance/recovery procedure if a future window
is approved. Do not create a new watchdog framework for this CPU task.
Accept or reject GPU/production deployment only from the new observations;
the prior 11.4% median endpoint difference is not acceptance of this tree.
