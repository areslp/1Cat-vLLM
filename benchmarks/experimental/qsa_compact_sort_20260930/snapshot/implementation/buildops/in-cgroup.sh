#!/bin/bash
# STEP58 CPU build only; source and command-contract remain frozen.
set -uo pipefail
TASK_D=/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930
TASK_RUN="$TASK_D/buildops/attempt1"
TASK_PY=/home/l/work/1Cat-vLLM/.venv/bin/python
TASK_UNIT=step58-build1.service
TASK_FINISHED=0

finish() {
  local task_status=$?
  if [[ "$TASK_FINISHED" == 1 ]]; then return; fi
  TASK_FINISHED=1
  set +e
  printf '%s\n' "$task_status" > "$TASK_RUN/inner-exit.code"
  date -u +%FT%TZ > "$TASK_RUN/inner-finished.utc"
  systemctl show "$TASK_UNIT" \
    -p Id -p User -p WorkingDirectory -p MemoryMax -p MemorySwapMax \
    -p MemoryPeak -p MemoryCurrent -p OOMPolicy -p RuntimeMaxUSec \
    -p KillMode -p TimeoutStopUSec -p MainPID -p ControlGroup \
    -p ActiveState -p SubState -p CPUAffinity \
    > "$TASK_RUN/cgroup-properties-before-exit.txt" 2>&1
  local task_cgroup
  task_cgroup=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
  printf '%s\n' "$task_cgroup" > "$TASK_RUN/cgroup-path.txt"
  for task_name in memory.peak memory.current memory.events memory.events.local \
                   memory.max memory.swap.max memory.swap.current; do
    cat "/sys/fs/cgroup$task_cgroup/$task_name" \
      > "$TASK_RUN/$task_name" 2>&1
  done
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

[[ "${CUDA_VISIBLE_DEVICES-UNSET}" == "" ]] || exit 90
[[ "$MAX_JOBS" == 2 && "$TORCH_CUDA_ARCH_LIST" == 7.0 ]] || exit 91
[[ "$BUILD_DIR" == "$TASK_D/build/candidate1" ]] || exit 92
[[ ! -e "$BUILD_DIR" ]] || exit 93
date -u +%FT%TZ > "$TASK_RUN/inner-started.utc"
printf '%s\n' "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  "MAX_JOBS=$MAX_JOBS" "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST" \
  "CUDA_HOME=$CUDA_HOME" "BUILD_DIR=$BUILD_DIR" \
  > "$TASK_RUN/build-environment.txt"
timeout --signal=TERM --kill-after=20s 600s \
  taskset -c 14,42 nice -n 15 "$TASK_PY" -u source/build58.py \
  2>&1 | tee "$TASK_D/build/candidate1.log"
TASK_BUILD_EXIT=${PIPESTATUS[0]}
printf '%s\n' "$TASK_BUILD_EXIT" > "$TASK_RUN/build-exit.code"
exit "$TASK_BUILD_EXIT"
