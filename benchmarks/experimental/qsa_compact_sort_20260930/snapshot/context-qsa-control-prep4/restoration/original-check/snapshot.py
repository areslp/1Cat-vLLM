"""Read-only STEP-53 deployment/cleanup snapshot, no imports of model code."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

D = Path(__file__).resolve().parent
R = Path('/home/l/work/flash-next/perf-20260923-resume')
K = Path('/home/l/work/1Cat-vLLM-kv-w12')
S = 'flash-next-vllm.service'
CG = Path('/sys/fs/cgroup/system.slice') / S
BASE = json.loads((R / 'step51-e7-service-20260930/baseline-step51.json').read_text())

def out(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def snapshot(tag):
    state = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'beijing': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
    state['runtime'] = {'python': sys.version, 'executable': sys.executable,
                        'uv': out('/home/l/.local/bin/uv', '--version').strip(),
                        'profile': 'existing run-flash-next-w12.sh R/.venv; no environment mutation'}
    unit = out('systemctl', 'cat', S)
    state['unit_sha256'] = hashlib.sha256(unit.encode()).hexdigest()
    (D / (tag + '-unit.txt')).write_text(unit)
    state['systemd'] = dict(line.split('=', 1) for line in out(
        'systemctl', 'show', S, '-p', 'MainPID', '-p', 'NRestarts', '-p', 'ActiveState',
        '-p', 'SubState', '-p', 'MemorySwapCurrent', '-p', 'ActiveEnterTimestamp').splitlines())
    state['head'] = out('git', '-C', str(K), 'rev-parse', 'HEAD').strip()
    state['branch'] = out('git', '-C', str(K), 'branch', '--show-current').strip()
    state['git_status'] = out('git', '-C', str(K), 'status', '--short')
    state['tracked_diff'] = out('git', '-C', str(K), 'diff', 'HEAD', '--name-only')
    state['files'] = {BASE['binary_path']: sha(BASE['binary_path']), BASE['shim_path']: sha(BASE['shim_path'])}
    sources = {**BASE['runtime_sources'], **BASE['D1_sources'], **BASE['E7_sources'],
               'vllm/v1/worker/gpu/model_runner.py': BASE['model_runner_sha256']}
    state['sources'] = {p: sha(K / p) for p in sources}
    state['dropins'] = {p.name: sha(p) for p in (Path('/etc/systemd/system') / (S + '.d')).glob('*.conf')}
    state['temporary_dropins'] = {str(p): sha(p) for p in (Path('/run/systemd/system') / (S + '.d')).glob('*.conf')}
    for key in ('memory.events', 'memory.swap.current', 'cpuset.cpus.effective', 'memory.current'):
        state[key] = (CG / key).read_text().strip()
    state['procs'] = []
    for pid in (CG / 'cgroup.procs').read_text().split():
        p = Path('/proc') / pid
        try:
            env = dict(x.split('=', 1) for x in (p / 'environ').read_text().split('\0') if '=' in x)
            status = (p / 'status').read_text()
            state['procs'].append({'pid': int(pid), 'cmdline': (p / 'cmdline').read_text().replace('\0', ' '),
              'exe': os.readlink(p / 'exe'), 'cwd': os.readlink(p / 'cwd'),
              'flags': {k: v for k, v in env.items() if k in BASE['selected_flags'] or k in ('CODE', 'PYTHONPATH', 'FAV100_SHIM', 'VLLM_SM70_QSA_GROUPED_PAD_FIX')},
              'status': [x for x in status.splitlines() if x.startswith(('Name:', 'VmSwap:', 'Cpus_allowed_list:', 'Mems_allowed_list:'))],
              'binary_maps': sorted(set(x.split()[-1] for x in (p / 'maps').read_text().splitlines() if 'flash_attn_v100' in x))})
        except FileNotFoundError:
            pass
    state['workers'] = []
    pids = {x['pid'] for x in state['procs']}
    for p in Path(BASE['selected_flags']['ONECAT_E7_STATS_DIR']).glob('e7-*.json'):
        row = json.loads(p.read_text())
        if row['pid'] in pids:
            state['workers'].append(row)
    state['workers'].sort(key=lambda x: x.get('rank', -1))
    cache = Path('/home/l/.cache/vllm/torch_compile_cache/torch_aot_compile')
    ids = {'target': '2c0d487b48e8afba30bb1b95d7cdf9cd1b755de012280249d1ae0e01812aa442',
           'drafter': '901dfaa6bc892726f5d4a4ebe44fa778bf1f648a1363d261e7c2776895faccab'}
    state['best_configs'] = {role: {str(p.relative_to(cache / key / 'inductor_cache')): sha(p)
         for p in (cache / key / 'inductor_cache').rglob('*.best_config')} for role, key in ids.items()}
    ref = json.loads((R / 'step48-service-20260929/reference-state.json').read_text())['best_configs']
    with urllib.request.urlopen('http://127.0.0.1:8200/health', timeout=10) as response:
        state['health'] = response.status
    with urllib.request.urlopen('http://127.0.0.1:8200/metrics', timeout=10) as response:
        metrics = response.read().decode()
    (D / (tag + '-metrics.txt')).write_text(metrics)
    state['queue'] = {x.split('{')[0]: float(x.rsplit(' ', 1)[1]) for x in metrics.splitlines()
                      if x.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))}
    log = Path('/home/l/work/flash-next/logs/vllm.log')
    # Tail is bounded; startup/profiler evidence extracted from it only.
    tail = out('tail', '-n', '4500', str(log))
    marker = f"(APIServer pid={state['systemd']['MainPID']})"
    relevant = [x for x in tail.splitlines() if ('GPU KV cache size:' in x or 'Application startup complete' in x
        or 'Starting vLLM server' in x or 'profiler' in x.lower() or '[fav100_shim]' in x)]
    (D / (tag + '-startup-profiler.log')).write_text('\n'.join(relevant) + '\n')
    # Last capacity entry must belong to currently live EngineCore.
    engines = [x['pid'] for x in state['procs'] if 'EngineCor' in '\n'.join(x['status'])]
    kv = [int(m.group(1).replace(',', '')) for x in tail.splitlines() for m in
          [re.search(r'GPU KV cache size: ([\d,]+) tokens', x)] if m and any(f'pid={pid})' in x for pid in engines)]
    state['kv_tokens'] = kv[-1] if kv else None
    state['ready_log'] = [x for x in relevant if marker in x and 'Starting vLLM server' in x]
    state['gpu'] = out('nvidia-smi', '--query-gpu=index,memory.used,memory.total,clocks.sm,clocks.mem,power.draw,temperature.gpu', '--format=csv')
    state['active_experiment_units'] = out('systemctl', 'list-units', '--state=active', '--no-legend', '--no-pager', 'step*', 'perf-profile*', '*guardian*')
    state['guardian'] = json.loads(Path('/home/l/.local/state/host-insight-mcp/guardian/profile-guardian-state.json').read_text())
    state['experiment_manager'] = json.loads(Path('/home/l/.local/state/host-insight-mcp/experiments/experiment-state.json').read_text())
    state['experiment_processes'] = [x for x in out('ps', '-eo', 'pid,ppid,args').splitlines()
       if any(t in x for t in ('restore51', 'restore48', 'profile_guardian', 'validation3.py', 'capture.py'))]
    checks = {
      'health': state['health'] == 200 and not any(state['queue'].values()),
      'source_config': state['head'] == BASE['head'] and not state['tracked_diff'] and state['sources'] == sources
          and state['dropins'] == BASE['dropins'] and state['unit_sha256'] == BASE['unit_sha256']
          and state['files'][BASE['binary_path']] == BASE['binary_sha256']
          and state['files'][BASE['shim_path']] == BASE['shim_sha256'] and state['best_configs'] == ref,
      'kv_capacity': state['kv_tokens'] == 663816,
      'swap': state['memory.swap.current'] == '0' and all('VmSwap:\t       0 kB' in '\n'.join(p['status']) or not any(x.startswith('VmSwap:') for x in p['status']) for p in state['procs']),
      'cleanup': not state['temporary_dropins'] and not state['active_experiment_units'],
      'worker_identity': [x['rank'] for x in state['workers']] == [0, 1, 2, 3]
          and all(x['mode'] == 'on' for x in state['workers']),
      'flags': all(all(p['flags'].get(k) == v for k, v in BASE['selected_flags'].items()) for p in state['procs'])
    }
    state['checks'] = checks
    (D / (tag + '-snapshot.json')).write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps({'tag': tag, 'checks': checks, 'pid': state['systemd']['MainPID'], 'kv_tokens': state['kv_tokens'],
                      'guardian': state['guardian'], 'active_units': state['active_experiment_units'], 'ready': state['ready_log']}), flush=True)
    return state

if __name__ == '__main__':
    snapshot(sys.argv[1])
