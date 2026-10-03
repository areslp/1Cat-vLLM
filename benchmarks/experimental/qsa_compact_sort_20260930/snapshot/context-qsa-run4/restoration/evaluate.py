"""Fail-closed restoration verdict; retain raw snapshot flag limitation."""
import datetime
import json
import subprocess
from pathlib import Path

D=Path(__file__).resolve().parent
check=D/'original-check'
before=json.loads((check/'before-snapshot.json').read_text())
after=json.loads((check/'after-snapshot.json').read_text())
requests=json.loads((check/'requests-result.json').read_text())
supp=json.loads((check/'supplement.json').read_text())
ready=json.loads((D/'readiness.json').read_text())
baseline=json.loads((D.parent/'baseline/before-snapshot.json').read_text())
stopped=(D/'stopped.txt').read_text().strip().split(' ',1)[1]
epoch=int(datetime.datetime.fromisoformat(stopped).timestamp())
journal=subprocess.run(['sudo','-n','journalctl','-k','--since',f'@{epoch}',
    '--no-pager','--grep','Out of memory|Killed process|oom-kill'],capture_output=True,text=True)
journal_evidence={'since_epoch':epoch,'returncode':journal.returncode,
                  'stdout':journal.stdout,'stderr':journal.stderr}
(D/'kernel-window.json').write_text(json.dumps(journal_evidence,indent=2)+'\n')
gates={
 'controller_restore_exit':(D/'verify.exit').read_text().strip()=='0',
 'readiness_maturity':ready['status']=='READY_AND_MATURE' and ready['ready_maturity_s']>=360,
 'stable_pid':str(ready['pid'])==before['systemd']['MainPID']==after['systemd']['MainPID'] and str(ready['pid']) != str(baseline['systemd']['MainPID']),
 'no_restarts':after['systemd']['NRestarts']=='0',
 'source_config_kv_swap_cleanup_worker':all(after['checks'][k] and before['checks'][k] for k in
     ('health','source_config','kv_capacity','swap','cleanup','worker_identity')),
 'original_unit':after['unit_sha256']==baseline['unit_sha256'],
 'requests':requests['status']=='PASS' and len(requests['requests'])==24 and requests['request_pass'],
 'six_fast_rounds':requests['fast_state_pass'] and len(requests['rounds'])==6,
 'all_24_fast':len(requests['requests'])==24 and all(x['fast'] for x in requests['requests']),
 'fixed_exact':len(requests['fixed'])==3 and all(all(x['checks'].values()) for x in requests['fixed']),
 'finite_concurrency':len(requests['concurrent'])==4 and all(all(x['checks'].values()) for x in requests['concurrent']),
 'e7_rank_equal_compressed':requests['e7_pass'],
 'no_active_jobs':not supp['active_jobs'],
 'no_guardian':not after['guardian']['active'],
 'no_oom':journal.returncode in (0,1) and not journal.stderr.strip()
     and journal.stdout.strip() in ('','-- No entries --'),
 'timer_cleared':'ActiveState=inactive' in (D/'expiry-state-cleared.txt').read_text(),
}
events=dict(line.split() for line in supp['memory_events'].splitlines())
gates['cgroup_oom_zero']=all(int(events.get(k,-1))==0 for k in ('oom','oom_kill'))
r={'status':'PASS' if all(gates.values()) else 'FAIL','gates':gates,
   'pid':int(after['systemd']['MainPID']), 'head':after['head'],
   'kv_tokens':after['kv_tokens'], 'rounds':requests['rounds'],
   'raw_flags_boolean':after['checks']['flags'],
   'flag_evidence_limitation':'Raw child /proc flag check remains false; retain it. API/unit, mapped binary/source, runtime routes and E7 rank telemetry provide separate evidence.',
   'readiness':ready}
(D/'RESTORATION.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
raise SystemExit(0 if r['status']=='PASS' else 1)
