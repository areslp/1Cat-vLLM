# QSA publication handoff — 2026-10-03

## Intent and scope

The user requested that the existing work be organized on `oh-my-gpu`, reviewed
independently, and then committed and pushed on a branch. The target is
`areslp/1Cat-vLLM`, branch `codex/qsa-compact-sort-review-20261003`. Merge-back is
intentionally skipped. This package is a source/evidence snapshot; no runtime
integration is implied by placing it in the repository.

The tested vLLM source was `2dd437062a971a83d1f905249d288269c8daf501`; the Git
publication base is `fe67339ddf862df0441a4e844fc664d9cafdffd0`. The preserved
baseline, contracts and paths must be used when interpreting recorded results.

## Validation and evidence

The previous host run completed the 114-group / 798-request / 112488-token
matrix. Independent raw token reconstructions agreed on 88 equal, 48 A/A-varying,
and 10 A0=A2/B-different request triples. Resource gates and the fresh 53-request /
17-gate production restoration passed. See the linked review pack for historical
commands, result paths, failed attempts and unchanged experiment exit codes.

For publication, source files are SHA256-pinned in `SOURCE-MANIFEST.json` and
important host-only evidence in `HOST-EVIDENCE-INDEX.json`. A separate
`PUBLICATION-CHECKS.json` records byte comparison against the original host,
inventory checks and skipped execution checks. The final Git commit/remote-ref
readback receipt is stored outside the frozen archive inventory in
`review/qsa-git-publication-20261003/` on the host. Git whitespace checking reports one original blank line at EOF in the frozen
`input-8192.json` seed; it is preserved to keep its tested SHA unchanged. The
independent package review is `PASS_WITH_LIMITS`. No new tests or linters are
claimed; no benchmark was repeated during publication.

The isolated managed worktree leaves the primary checkout untouched. Only this
experiment package and the repository publication changelog/handoff are staged.
Original source bytes and historical reports must not be reformatted or silently
rewritten to make the snapshot look like a validated integration.

## Remaining review questions

1. Explain the 10 per-request B disagreements before attributing them to QSA or
   dismissing them as baseline variation. Group-level labels are insufficient.
2. Assess whether small, descriptive gains merit further engineering effort.
   32K/c4 and 64K/c4 are candidates for a controlled follow-up, not accepted wins.
3. Keep n=2, A/A drift, missing common windows, synthetic long inputs and missing
   dynamic candidate-hit counts in view. SSE timing does not prove physical limits.
4. Review frozen source and contracts on their tested baseline before any port
   to the current main branch. The package is not self-contained for execution.

Additional GPU runs, production changes and merge remain separate actions.
The default next action is independent review of this branch and the host archive.
