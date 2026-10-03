#!/usr/bin/env bash
# Root supplies the exact reviewed fresh plan SHA and expected production PID.
set -uo pipefail
umask 077
D=/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930/context-qsa-run4
P=/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930/context-qsa-control-prep4
PY=/home/l/work/1Cat-vLLM/.venv/bin/python
U=flash-next-recovery-orchestrator-step58-contextqsa4.service
cd "$D" || exit 1
test "$(systemctl show "$U" -p LoadState --value)" = not-found || exit 125
test ! -e control/outer-launch.exit || exit 125
test ! -e control/attempt1 || exit 125
sudo -n systemd-run --wait --collect --unit="$U" --property=User=l --property=Group=l \
    --property=WorkingDirectory="$D" --property=RuntimeMaxSec=31200s \
    --property=TimeoutStopSec=30s --property=KillMode=control-group \
    --property=MemoryMax=8G --property=MemorySwapMax=0 \
    --property=AllowedCPUs=0-13,15-41,43-55 --property=Environment=CUDA_VISIBLE_DEVICES= \
    --property="Environment=EXPECTED_PID=${EXPECTED_PID:?fresh PID required}" \
    --property="Environment=STEP58_CONTEXTQSA4_APPROVED_PLAN_SHA256=${STEP58_CONTEXTQSA4_APPROVED_PLAN_SHA256:?exact root SHA required}" \
    --property="StandardOutput=append:$D/control/outer.log" --property="StandardError=append:$D/control/outer.log" \
    /bin/bash "$D/control/run_and_restore.sh" > control/systemd-outer-wait.log 2>&1
outer_rc=$?
printf '%s\n' "$outer_rc" > control/outer-launch.exit
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 "$PY" "$P/control/orchestrator_evidence.py" post > control/post-outer.log 2>&1
post_rc=$?
printf '%s\n' "$post_rc" > control/post-outer.exit
printf 'OUTER_EXIT=%s POST_OUTER_CLEANUP_EXIT=%s\n' "$outer_rc" "$post_rc"
test "$outer_rc" = 0 && test "$post_rc" = 0
