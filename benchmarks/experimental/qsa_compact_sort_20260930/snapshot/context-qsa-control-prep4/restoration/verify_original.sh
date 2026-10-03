#!/usr/bin/env bash
# Run4 fresh maturity/snapshot/Python53 sequence, own outputs/timer only.
set -euo pipefail
D=/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930/context-qsa-run4
P=/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930/context-qsa-control-prep4
PY=/home/l/work/1Cat-vLLM/.venv/bin/python
cd "$D"
"$PY" restoration/wait_original.py > restoration/readiness.json
systemctl cat flash-next-vllm.service > restoration/post-window-unit.txt
cmp restoration/pre-window-unit.txt restoration/post-window-unit.txt
sudo -n systemctl stop step58-contextqsa4-expiry.timer
systemctl show step58-contextqsa4-expiry.timer -p ActiveState -p SubState > restoration/expiry-state-cleared.txt
"$PY" "$P/control/orchestrator_evidence.py" restore > restoration/owned-cpu-orchestrator.log
"$PY" restoration/original-check/snapshot.py before > restoration/original-check/preflight.log
"$PY" -u restoration/original-check/closure_requests.py > restoration/original-check/requests.log 2>&1
"$PY" restoration/original-check/supplement.py > restoration/original-check/supplement.log
printf 'ORIGINAL_RESTORATION_GATE_PASS\n'
