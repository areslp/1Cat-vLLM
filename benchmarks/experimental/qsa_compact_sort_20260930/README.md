# QSA compact-sort experiment and review snapshot

This directory preserves the QSA planner prototype, service integration hooks,
benchmark controls, and bounded review results from `oh-my-gpu`. It is an
**experimental source snapshot for review**, not an enabled vLLM feature or a
self-contained replay package. No production code outside this directory changes.

The recorded experiment used vLLM commit
`2dd437062a971a83d1f905249d288269c8daf501` in
`/home/l/work/1Cat-vLLM-kv-w12`. The publication branch starts at
`fe67339ddf862df0441a4e844fc664d9cafdffd0`. These sources have **not** been ported,
built, or benchmarked against that newer branch base.

## Review entry points

- [Final review pack](snapshot/review/context-qsa-closure-20261002/REVIEW-PACK.md)
  and [all 19 result rows](snapshot/review/context-qsa-closure-20261002/CONTEXT-RESULTS.md).
- [Handoff and value assessment](snapshot/REVIEW-HANDOFF-20261003.md) and
  [independent reviewer prompt](snapshot/REVIEW-PROMPT-20261003.md).
- [Algorithm guide](snapshot/review/context-qsa-independent-20261002/ALGORITHM-REVIEW-GUIDE.md).
- [Per-request findings](snapshot/review/context-qsa-closure-20261002/ROOT-FINAL-REVIEW.json)
  and [independent cross-review](snapshot/review/context-qsa-run4-postprocess-20261003/CROSS-REVIEW.md).
- [Source manifest](SOURCE-MANIFEST.json), [host evidence index](HOST-EVIDENCE-INDEX.json),
  and [publication handoff](HANDOFF.md).

The frozen documents retain their original dates, paths, conclusions, and
instructions. Their historical statement that nothing was pushed describes the
experiment closeout. The user subsequently authorized this review-branch commit
and push; production merge and permanent deployment remain out of scope. Some
links in frozen documents point to evidence retained only on the host.

## Recorded result

`context-qsa-run4` completed 19 context/concurrency rows, two repeats, and three
arms (A0/off, B/on, A2/off): **114 groups, 798 HTTP requests, 112488 output tokens**.
Contexts span 512 to 261632 input tokens. This is distinct from historical
`service run4`; mixed/staggered scheduling results must not substitute for the
same-length context comparison.

At 32K/c4 the two descriptive gains were +0.602% / +0.776%; at 64K/c4 they were
+0.652% / +0.578%. Both rows had identical output tokens across the three arms.
Some c8 rows changed sign between repeats. All six 64K/c8 groups had
`NO_COMMON_WINDOW`, retained as missing. With n=2, no per-group dynamic candidate
hit count, and client SSE timing rather than kernel timing, these results do not
establish general speedup, admission, or a physical limit.

Independent raw SSE reconstruction found 146 request triples: 88 all equal,
48 with A0/A2 self-variation, and **10 with A0=A2 but B different**. The original
group-level classification masks those individual disagreements. They block an
overall output-consistency claim, while the cause remains unresolved. Finite W1
planner/forward out/LSE checks passed on four V100s (280 cases per card), but do
not prove complete service-state/logit equivalence. The original analysis exit 3
and outer/window exit 1 remain preserved.

Resource audits passed the recorded 32384 MiB per-process/per-device limits and
1 GiB client cap with zero memory.events.max/oom/oom_kill and Swap. GPU maxima
are sampled, not continuous peaks. The original production baseline was restored
and passed fresh **53-request / 17-gate** acceptance and subsequent health readback.
This is a dated restoration result, not a live health assertion at checkout time.

The current recommendation is to pause additional QSA experiments pending
independent review of correctness and engineering value. A sub-1% descriptive
gain does not justify deployment while output attribution remains unresolved.

## Source map

| Directory under `snapshot/` | Purpose |
| --- | --- |
| `implementation/` | Compact-sort CUDA planner, bindings, microbenchmark and original/reference implementation; original BSD license retained |
| `w1-implementation/` | Finite planner/forward equivalence, eager/graph checks and guards |
| `service-implementation-retry2/` | Opt-in service policy, graph hook, build and validation code |
| `context-qsa-prep1/` | Frozen matrix, group generation, analyzer and two seed inputs |
| `context-qsa-control-prep4/` | Frozen orchestration, source identity and restoration controls |
| `context-qsa-transport-prep1/` | Bounded streaming transport supporting long input lines |
| `service-w2-control-prep4/`, `service-w2-prep/` | Shared pinned transport, counters and restoration helpers |
| `context-qsa-run4/` | Actual frozen plan, selected control scripts and original analysis result |
| `review/` | Independent raw review implementations and bounded final results |

Other small directories preserve dependencies pinned by the actual run plan.
Reference code is retained for provenance and comparison, not registered as a
second runtime implementation. No import hooks or build integration were added
to the repository.

## Reproduction boundary and provenance

Access the full evidence archive using local `ssh oh-my-gpu` at:

```text
/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930
```

The full archive contains 36398 regular files (3083389890 bytes) and six preserved
historical symlinks. Compiled libraries, raw SSE, tensor fixtures, generated prompt
groups, environments and bulk historical logs are not published in Git. Two
small seed JSON inputs are included. The archive inventory SHA256 is
`e5e6824e1cecc2c938859df35e204dfb98d9332e2b691170d18dc90cc6e1e24e`.
The run plan SHA256 is
`355ef8caeee318fc79abad12cf4dcf70b05dd3a822177eb17b03ec2675b64ba4`.

Every file under `snapshot/` is copied without edits and has its original host
path, byte count and SHA256 in `SOURCE-MANIFEST.json`. The exporter selects source
and small reports from the existing local mirror; it does not run experiments.
`HOST-EVIDENCE-INDEX.json` pins important host-only artifacts, including the full
archive inventory. The full dependency manifests are intentionally unchanged
and include files omitted from this Git package. Absolute paths, runtime guards,
historical missing/dangling fixture links, and external restoration dependencies
remain visible. Do not bypass those guards or treat this snapshot as runnable on
an arbitrary checkout. Any replay needs a separately authorized window and the
original environment/archive, including its external dependencies.

Publication validation consists of source-hash comparison against the host,
inventory/size/credential-signature review, and Git diff/ref checks. No new GPU
loads, tests, linters, builds, or production changes were performed for publication.
