# Handoff

## 2026-10-03 — QSA branch publication

The user authorized committing and pushing the existing QSA experiment on
`codex/qsa-compact-sort-review-20261003` to `areslp/1Cat-vLLM`. Production merge and
permanent deployment remain out of scope. The primary checkout is unchanged;
publication uses an isolated managed worktree.

See the [package README](../benchmarks/experimental/qsa_compact_sort_20260930/README.md)
and [detailed handoff](../benchmarks/experimental/qsa_compact_sort_20260930/HANDOFF.md)
for source provenance, prior experiment outcomes, known output differences,
restoration evidence and remaining review questions.

Publication checks compare 265 source/report files and 11 host evidence index
entries with their original host bytes. The per-file result is preserved in
[PUBLICATION-CHECKS.json](../benchmarks/experimental/qsa_compact_sort_20260930/PUBLICATION-CHECKS.json).
No tests, linters, builds or benchmarks were rerun, and no compatibility with the
current main branch is claimed. The full raw archive remains on `oh-my-gpu`;
Git contains source, small reports and hash/path indexes only.

After pushing, the final commit and independently read remote ref are recorded at
`/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930/review/qsa-git-publication-20261003/`.
The next action is independent review, not further experimentation or deployment.
