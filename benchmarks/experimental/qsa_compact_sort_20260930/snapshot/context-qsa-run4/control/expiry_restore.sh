#!/usr/bin/env bash
# Fallback start request only; outer proves complete53/newPID independently.
set -uo pipefail
umask 077
D=/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930/context-qsa-run4
exec >> "$D/control/attempt1/expiry-recovery.log" 2>&1
restore() {
    prior_rc=$?; trap - EXIT TERM INT; set +e
    timeout --kill-after=5 20 systemctl start --no-block flash-next-vllm.service
    start_rc=$?
    printf 'EXPIRY_BODY_EXIT=%s ORIGINAL_START_EXIT=%s\n' "$prior_rc" "$start_rc"
    test "$prior_rc" = 0 && test "$start_rc" = 0
}
trap restore EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf 'EXPIRY_RECOVERY %s\n' "$(date --iso-8601=seconds)"
timeout --kill-after=5 50 systemctl stop step58-contextqsa4-a0.service step58-contextqsa4-b.service step58-contextqsa4-a2.service step58-contextqsa4-http-a0.service step58-contextqsa4-http-b.service step58-contextqsa4-http-a2.service
exit "$?"
