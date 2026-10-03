#!/usr/bin/env bash
# Surviving CPU outer attempts full new-PID restoration before aggregation.
set -uo pipefail
umask 077
D=/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930/context-qsa-run4
P=/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930/context-qsa-control-prep4
PY=/home/l/work/1Cat-vLLM/.venv/bin/python
A="$D/control/attempt1"
cd "$D" || exit 1
for path in "$A" restoration/stop-attempted.txt restoration/verify.exit control/window.exit; do
    test ! -e "$path" || exit 125
done
mkdir -m 700 "$A" || exit 125
timeout --signal=TERM --kill-after=20 27600 bash control/offline_window.sh > "$A/window.log" 2>&1
window_rc=$?
printf '%s\n' "$window_rc" > control/window.exit
restore_rc=125; evaluate_rc=125; cleanup_rc=0; start_rc=0
if test -f restoration/stop-attempted.txt; then
    timeout --kill-after=5 700 "$PY" "$P/control/step58_control.py" cleanup --label outer-before-restore > "$A/outer-cleanup.log" 2>&1
    cleanup_rc=$?
    printf '%s\n' "$cleanup_rc" > "$A/outer-cleanup.exit"
    timeout --kill-after=5 20 sudo -n systemctl start --no-block flash-next-vllm.service > "$A/outer-start.log" 2>&1
    start_rc=$?
    printf '%s\n' "$start_rc" > "$A/outer-start.exit"
    timeout --kill-after=20 2100 bash restoration/verify_original.sh > restoration/verify.log 2>&1
    restore_rc=$?
fi
printf '%s\n' "$restore_rc" > restoration/verify.exit
if test "$restore_rc" = 0 && test -f restoration/stopped.txt; then
    timeout --kill-after=5 60 "$PY" restoration/evaluate.py > restoration/evaluate.log 2>&1
    evaluate_rc=$?
fi
printf '%s\n' "$evaluate_rc" > restoration/evaluate.exit
CUDA_VISIBLE_DEVICES= timeout --kill-after=5 30 "$PY" "$P/control/step58_control.py" artifacts > control/artifact-final.log 2>&1
artifact_rc=$?
printf '%s\n' "$artifact_rc" > control/artifact-final.exit
printf 'WINDOW_EXIT=%s CLEANUP_EXIT=%s ORIGINAL_START_EXIT=%s RESTORE_EXIT=%s EVALUATE_EXIT=%s ARTIFACT_EXIT=%s\n' "$window_rc" "$cleanup_rc" "$start_rc" "$restore_rc" "$evaluate_rc" "$artifact_rc"
test "$window_rc" = 0 && test "$cleanup_rc" = 0 && test "$start_rc" = 0 && test "$restore_rc" = 0 && test "$evaluate_rc" = 0 && test "$artifact_rc" = 0
