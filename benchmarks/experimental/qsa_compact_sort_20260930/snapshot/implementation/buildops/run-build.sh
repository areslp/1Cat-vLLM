#!/bin/bash
# Single owned CPU compilation. No package installs, GPU calls or service changes.
set -euo pipefail
TASK_D=/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930
TASK_K=/home/l/work/1Cat-vLLM-kv-w12
TASK_RUN="$TASK_D/buildops/attempt1"
TASK_PY=/home/l/work/1Cat-vLLM/.venv/bin/python
TASK_UNIT=step58-build1.service
cd "$TASK_D/implementation"
[[ ! -e "$TASK_RUN" && ! -e "$TASK_D/build/candidate1" ]] || exit 94
[[ $(git -C "$TASK_K" rev-parse HEAD) == 2dd437062a971a83d1f905249d288269c8daf501 ]] || exit 95
sha256sum -c evidence/candidate-manifest.sha256
mkdir -p "$TASK_D/build" "$TASK_D/buildops"
mkdir "$TASK_RUN"
date -u +%FT%TZ > "$TASK_RUN/started.utc"
git -C "$TASK_K" rev-parse HEAD > "$TASK_RUN/source-head.txt"
sha256sum -c evidence/candidate-manifest.sha256 > "$TASK_RUN/source-readback.txt"
sha256sum buildops/in-cgroup.sh buildops/run-build.sh > "$TASK_RUN/runner.sha256"
"$TASK_PY" --version > "$TASK_RUN/python-version.txt" 2>&1
/home/l/.local/bin/uv --version > "$TASK_RUN/uv-version.txt" 2>&1
/home/l/work/qwen38-bench/cuda-home/bin/nvcc --version \
  > "$TASK_RUN/nvcc-version.txt" 2>&1
c++ --version > "$TASK_RUN/cxx-version.txt" 2>&1
lscpu -p=CPU,ONLINE > "$TASK_RUN/cpu-online.txt"
printf '%s\n' 'Frozen contract uv --version runtime path correction: /home/l/.local/bin/uv (outside PATH); no source/contract mutation.' \
  > "$TASK_RUN/uv-path-correction.txt"
for task_cpu in 14 42; do
  awk -F, -v wanted="$task_cpu" '$1 == wanted && $2 == "Y" {found=1} END {exit !found}' \
    "$TASK_RUN/cpu-online.txt"
done
set +e
sudo -n systemd-run --unit="$TASK_UNIT" --service-type=exec --wait --pipe \
  --property=User=l --property=WorkingDirectory="$TASK_D/implementation" \
  --property=MemoryMax=8589934592 --property=MemorySwapMax=0 \
  --property=MemoryAccounting=yes --property=CPUAccounting=yes \
  --property=RuntimeMaxSec=650s --property=KillMode=control-group \
  --property=TimeoutStopSec=20s --property=OOMPolicy=stop \
  --setenv=CUDA_VISIBLE_DEVICES= --setenv=MAX_JOBS=2 \
  --setenv=TORCH_CUDA_ARCH_LIST=7.0 \
  --setenv=CUDA_HOME=/home/l/work/qwen38-bench/cuda-home \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=BUILD_DIR="$TASK_D/build/candidate1" \
  /bin/bash "$TASK_D/implementation/buildops/in-cgroup.sh" \
  2>&1 | tee "$TASK_RUN/systemd-run.log"
TASK_RUN_EXIT=${PIPESTATUS[0]}
printf '%s\n' "$TASK_RUN_EXIT" > "$TASK_RUN/systemd-run-exit.code"
systemctl show "$TASK_UNIT" \
  -p Id -p LoadState -p ActiveState -p SubState -p Result \
  -p ExecMainCode -p ExecMainStatus -p MemoryPeak -p MemoryMax \
  -p MemorySwapMax -p ControlGroup -p MainPID -p RuntimeMaxUSec \
  -p KillMode -p User -p WorkingDirectory \
  > "$TASK_RUN/unit-final-properties.txt" 2>&1
sudo -n journalctl -u "$TASK_UNIT" --no-pager -o short-iso \
  > "$TASK_RUN/unit-complete-journal.log" 2>&1
if [[ -f "$TASK_D/build/candidate1.log" ]]; then
  rg 'ptxas|stack frame|spill|register|lmem|smem|CANDIDATE_PATH|built ' \
    "$TASK_D/build/candidate1.log" > "$TASK_RUN/ptxas-summary.txt"
fi
if [[ -f "$TASK_D/build/candidate1/qsa_planner58.so" ]]; then
  sha256sum "$TASK_D/build/candidate1/qsa_planner58.so" \
    > "$TASK_RUN/candidate.sha256"
fi
sudo -n systemctl stop "$TASK_UNIT" > "$TASK_RUN/cleanup-stop.txt" 2>&1
sudo -n systemctl reset-failed "$TASK_UNIT" > "$TASK_RUN/cleanup-reset-failed.txt" 2>&1
systemctl show "$TASK_UNIT" -p LoadState -p ActiveState -p SubState -p MainPID \
  > "$TASK_RUN/cleanup-unit.txt" 2>&1
date -u +%FT%TZ > "$TASK_RUN/finished.utc"
printf 'STEP58_BUILD_EXIT=%s\n' "$TASK_RUN_EXIT"
exit "$TASK_RUN_EXIT"
