"""Bounded wait for actual HTTP ready, then six minutes of process maturity."""
import datetime
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
D=Path(__file__).resolve().parent
started=time.time();deadline=started+1080
ready=None;pid=None
while time.time()<deadline:
    state=subprocess.check_output(['systemctl','show','flash-next-vllm.service','-p','ActiveState','--value'],text=True).strip()
    if state=='failed':raise RuntimeError('original service failed startup')
    try:
        with urllib.request.urlopen('http://127.0.0.1:8200/health',timeout=3) as r:
            if r.status==200:
                ready=time.time();pid=subprocess.check_output(['systemctl','show','flash-next-vllm.service','-p','MainPID','--value'],text=True).strip();break
    except (OSError,urllib.error.URLError):pass
    time.sleep(5)
if ready is None:raise TimeoutError('original health ready1080sdeadline')
(D/'http-ready-observed.json').write_text(json.dumps({'ready_epoch':ready,'pid':pid,'utc':datetime.datetime.fromtimestamp(ready,datetime.timezone.utc).isoformat()})+'\n')
# This wait is intentionally after real HTTP readiness, not systemd active timestamp.
time.sleep(360)
assert subprocess.check_output(['systemctl','show','flash-next-vllm.service','-p','MainPID','--value'],text=True).strip()==pid
with urllib.request.urlopen('http://127.0.0.1:8200/health',timeout=5) as r:assert r.status==200
print(json.dumps({'status':'READY_AND_MATURE','pid':int(pid),'http_ready_epoch':ready,'verified_epoch':time.time(),'ready_maturity_s':time.time()-ready,'total_wait_s':time.time()-started}))
