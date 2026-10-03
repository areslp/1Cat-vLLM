#!/usr/bin/env bash
# Parent-only FD8 maintenance ownership; every operational child closes FD8.
set -euo pipefail
umask 077
D=/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930/context-qsa-run4
P=/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930/context-qsa-control-prep4
PY=/home/l/work/1Cat-vLLM/.venv/bin/python
A="$D/control/attempt1"
cd "$D"
date --iso-8601=seconds > "$A/controller-started.txt"
exec 9>control/offline-window.lock
flock -n 9
timeout --kill-after=5 100 "$PY" "$P/control/step58_control.py" prepare --plan "$D/control/plan.json" --expected-pid "${EXPECTED_PID:?fresh PID required}"
restore() {
    window_rc=$?; trap - EXIT TERM INT; set +e
    printf '%s\n' "$window_rc" > "$A/controller.exit"
    cleanup_rc=0; start_rc=0
    if test -f restoration/stop-attempted.txt; then
        timeout --kill-after=5 750 "$PY" "$P/control/step58_control.py" cleanup --label exit-final > "$A/exit-final-cleanup.log" 2>&1 8>&-
        cleanup_rc=$?
        printf '%s\n' "$cleanup_rc" > "$A/exit-final-cleanup.exit"
        exec 8>&-
        timeout --kill-after=5 20 sudo -n systemctl start --no-block flash-next-vllm.service > "$A/original-start.log" 2>&1
        start_rc=$?
        printf '%s\n' "$start_rc" > "$A/original-start.exit"
        printf 'ORIGINAL_START_REQUESTED %s\n' "$(date --iso-8601=seconds)" > restoration/start-requested.txt
    else
        timeout --kill-after=5 20 sudo -n systemctl stop step58-contextqsa4-expiry.timer > "$A/unused-timer-stop.log" 2>&1
        cleanup_rc=$?
    fi
    test "$window_rc" = 0 || exit "$window_rc"
    test "$cleanup_rc" = 0 || exit "$cleanup_rc"
    exit "$start_rc"
}
trap restore EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
timeout --kill-after=5 20 sudo -n systemd-run --unit=step58-contextqsa4-expiry --on-active=462m --timer-property=AccuracySec=1s --property=TimeoutStartSec=155s --property=TimeoutStopSec=30s --property=KillMode=control-group /bin/bash "$D/control/expiry_restore.sh" > "$A/expiry-arm.log" 2>&1
systemctl show step58-contextqsa4-expiry.timer -p ActiveState -p SubState > restoration/expiry-state-armed.txt
test "$(systemctl show step58-contextqsa4-expiry.timer -p ActiveState --value)" = active
printf 'STOP_ATTEMPTED %s\n' "$(date --iso-8601=seconds)" > restoration/stop-attempted.txt
timeout --kill-after=5 20 sudo -n systemctl stop flash-next-vllm.service
printf 'STOPPED %s\n' "$(date --iso-8601=seconds)" > restoration/stopped.txt
systemctl show flash-next-vllm.service -p ActiveState -p SubState -p MainPID > restoration/stopped-state.txt
test "$(systemctl show flash-next-vllm.service -p ActiveState --value)" = inactive
exec 8>/home/l/.local/state/host-insight-mcp/gpu-maintenance.lock
flock -w 15 8
printf '{"status":"INNER_LOCK_HELD_CHILD_FD_CLOSED"}\n' > "$A/maintenance-lock-acquired.json"
timeout --kill-after=5 65 "$PY" "$P/control/step58_control.py" drain --label post-original-stop 8>&-
"$PY" "$P/control/step58_control.py" run 8>&-
