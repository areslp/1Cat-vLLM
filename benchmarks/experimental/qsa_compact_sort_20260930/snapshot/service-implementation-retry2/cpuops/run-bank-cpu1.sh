#!/bin/bash
set -u -o pipefail
D=/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930/service-implementation
E="$D/evidence/cpu-torch-attempt1"
mkdir "$E" || exit 73
unit=step58-service-bank-cpu1.service
sudo systemd-run --unit="$unit" --wait --pipe --remain-after-exit \
  --uid=l --working-directory="$D" \
  --property=MemoryMax=1G --property=MemorySwapMax=0 \
  --property=RuntimeMaxSec=35s --property=KillMode=control-group \
  /usr/bin/timeout 30 /usr/bin/env CUDA_VISIBLE_DEVICES= \
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 /usr/bin/taskset -c 14 /usr/bin/nice -n 15 \
    /home/l/work/1Cat-vLLM/.venv/bin/python "$D/cpu_torch_bank.py" \
    --output "$E/RESULT.json" > "$E/stdout-stderr.log" 2>&1
code=$?
printf '%s\n' "$code" > "$E/observed-exit.code"
sudo systemctl show "$unit" -p ActiveState -p SubState -p ExecMainStatus \
  -p MemoryMax -p MemorySwapMax -p MemoryPeak -p OOMPolicy -p ControlGroup \
  -p RuntimeMaxUSec -p User -p WorkingDirectory > "$E/unit-properties.txt"
sudo journalctl -u "$unit" --no-pager > "$E/journal.complete.log"
cg=$(sudo systemctl show "$unit" -p ControlGroup --value)
if [ -n "$cg" ] && [ -r "/sys/fs/cgroup$cg/memory.events" ]; then
  cat "/sys/fs/cgroup$cg/memory.events" > "$E/memory.events"
fi
sudo systemctl stop "$unit" > "$E/cleanup-stop.log" 2>&1
sudo systemctl reset-failed "$unit" > "$E/cleanup-reset.log" 2>&1 || true
sudo systemctl show "$unit" -p LoadState -p ActiveState -p SubState \
  > "$E/cleanup-unit.txt" 2>&1 || true
exit "$code"
