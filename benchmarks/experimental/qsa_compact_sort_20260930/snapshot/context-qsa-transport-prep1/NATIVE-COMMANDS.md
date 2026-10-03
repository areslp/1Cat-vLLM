# Native CPU512 capacity fixture; no production endpoint

Copy only `check_wire.py`, `transport_worker.py`, `transport_deadline_context.py`, `dependencies.py`, `DEPENDENCIES.json` before running. The original `context-qsa-prep1` matrix/body/dependency closure stays at its existing path. This is a synthetic real-HTTP/socket capacity test, not model timing, output equality, QSA admission, or run3 peak attribution. Server uses a dedicated ephemeral loopback port and a separate unit, outside client memory charging. It cannot contact port8200/8201. Explicit CPU-only URL routing calls the unchanged real requests/urllib3 transport, original frozen SSE parser and copied original Pump.

After root confirms source SHA, create a new root-owned review directory (example below). Start server outside client unit:

```bash
qsa_cpu_base=/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930
qsa_cpu_package=$qsa_cpu_base/context-qsa-transport-prep1
qsa_cpu_review=$qsa_cpu_base/review/context-qsa-transport-native512-20261002
qsa_cpu_python=/home/l/work/1Cat-vLLM/.venv/bin/python
mkdir "$qsa_cpu_review"
sudo -n systemd-run --collect --unit=step58-contextqsa4-cpu-server \
  --property=User=l --property=Group=l --property=RuntimeMaxSec=320s \
  --property=MemoryMax=512M --property=MemorySwapMax=0 \
  --property=AllowedCPUs=0-13,15-41,43-55 \
  --property=Environment=CUDA_VISIBLE_DEVICES= \
  --property=Environment=PYTHONDONTWRITEBYTECODE=1 \
  "$qsa_cpu_python" -I -B "$qsa_cpu_package/check_wire.py" \
  --server-only --seconds 300 --output "$qsa_cpu_review/server"
timeout 30 bash -c 'until test -f "$1"; do sleep 1; done' _ \
  "$qsa_cpu_review/server/SERVER.json"
qsa_cpu_port=$(CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  "$qsa_cpu_python" -I -B -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["port"])' \
  "$qsa_cpu_review/server/SERVER.json")
```

Run the four formal maximum cases, then independent line/received/count negative cases. `--precharge-tree` is optional; the example includes the root-approved closed run3 A0 prefix as a bounded ≤256MiB new owned copy to charge file cache to this client. This is capacity stress, not a retrospective model of run3 peak composition. Longest261632/c2 is tested first; fixed production matrix order/body/salt remains unchanged. CPU-case60s/request and whole suite300s are fixture time limits, not changes to production600/660/7200.

```bash
sudo -n systemd-run --wait --collect --unit=step58-contextqsa4-cpu512 \
  --property=User=l --property=Group=l --property=WorkingDirectory="$qsa_cpu_base" \
  --property=RuntimeMaxSec=360s --property=TimeoutStopSec=10s \
  --property=KillMode=control-group --property=MemoryMax=512M \
  --property=MemorySwapMax=0 --property=AllowedCPUs=14,42 --property=CPUQuota=200% \
  --property=Environment=CUDA_VISIBLE_DEVICES= \
  --property=Environment=PYTHONDONTWRITEBYTECODE=1 \
  "$qsa_cpu_python" -I -B "$qsa_cpu_package/check_wire.py" \
  --server-port "$qsa_cpu_port" --output "$qsa_cpu_review/client" \
  --precharge-tree "$qsa_cpu_base/context-qsa-run3/control/attempt1/A0/http"
qsa_cpu_client_rc=$?
sudo -n systemctl stop step58-contextqsa4-cpu-server.service
printf 'CPU_CLIENT_EXIT=%s\n' "$qsa_cpu_client_rc"
```

Always stop/collect the server after success or failure. Root should wrap that cleanup in its normal trap if invoking the commands as one shell. Do not retry in a larger cgroup automatically. `client/client-cgroup.json` contains actual memory.max/current/peak/events/stat/swap/cpuset after jobs close/reap; `512MiB_peak_pass` remains false if peak exceeds512. A zero transport exit and `PASS_REAL_LOOPBACK_TRANSPORT_NOT_MODEL_OR_512_GUARANTEE` certify only the HTTP/socket/parser cases. They do not turn a capacity failure into budget PASS. Preserve FAILURE and resource readback if either fails, then review any new preregistered client budget independently.

`--full-A0` is available for a separately authorized capacity lifetime test: all38 groups／266 HTTP in original body/salt/order, sequential primes and measured barriers, synthetic outputs. It is not necessary for the first bounded maximum-case CPU512 fixture and must use a fresh output directory. It does not create OriginalGuard success flags or engine/GPU timing evidence.
