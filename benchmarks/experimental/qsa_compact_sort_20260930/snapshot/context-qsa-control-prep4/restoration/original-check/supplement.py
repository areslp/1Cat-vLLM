"""Additional read-only deployment evidence; no service control."""
from pathlib import Path
import json
import subprocess
import time
from snapshot import D, S, CG

pids=(CG/'cgroup.procs').read_text().split()
rows=[]
for pid in pids:
    env=Path('/proc',pid,'environ').read_bytes()
    matches=[]
    for raw in env.split(b'\0'):
        if b'ONECAT_FUSE47=' in raw:
            # Only expose the task's non-secret flag fragment, not other env values.
            pos=raw.index(b'ONECAT_FUSE47=')
            matches.append({'byte_offset':env.index(raw)+pos,'value':raw[pos:].decode(errors='replace')})
    rows.append({'pid':int(pid),'fuse47_fragments':matches})
log=subprocess.check_output(['tail','-n','5500','/home/l/work/flash-next/logs/vllm.log'],text=True)
keep=[s for s in log.splitlines() if ('ONECAT_FUSE47' in s and any('pid='+p+')' in s for p in pids)) or ('Profiler stopped' in s)]
(D/'runtime-routes-profiler.log').write_text('\n'.join(keep)+'\n')
state=json.loads(Path('/home/l/.local/state/host-insight-mcp/experiments/experiment-state.json').read_text())
active=[{'id':j['id'],'status':j['status'],'kind':j['kind']} for j in state['jobs'].values() if j['status'] not in ('completed','failed','cancelled','canceled')]
since=(D.parent/'stopped.txt').read_text().strip().split(' ',1)[1]
result={'t':time.time(),'proc_flag_fragments':rows,'active_jobs':active,'memory_events':(CG/'memory.events').read_text(),
'oom_since':since,'oom_kernel_log':subprocess.run(['sudo','-n','journalctl','-k','--since',since,'--no-pager','--grep','Out of memory|Killed process|oom-kill'],capture_output=True,text=True).stdout}
(D/'supplement.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
