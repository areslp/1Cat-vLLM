"""One bounded off/shadow service diagnostic; stdlib orchestration only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

D = Path('/home/l/work/flash-next/perf-20260923-resume/'
         'step58-qsa-compact-sort-20260930/service-run3')
ROOT = D.parent
IMPL = ROOT / 'service-implementation-retry2'
K = Path('/home/l/work/1Cat-vLLM-kv-w12')
PY = '/home/l/work/1Cat-vLLM/.venv/bin/python'
S = 'flash-next-vllm.service'
HEAD = '2dd437062a971a83d1f905249d288269c8daf501'
MODEL = '/home/l/models/Qwen3.8-Flash-Next-NVFP4-QSA-E4M3-MTP'
A = D / 'control/attempt1'
OUTER = 'flash-next-recovery-orchestrator-step58-service3.service'
TIMER = 'step58-service3-expiry'
UNITS = ['step58-service-off3.service', 'step58-service-shadow3.service',
         'step58-service-smoke3.service']
UUIDS = ['GPU-54a5dec0-85a9-837b-c54d-4a3752c1b620',
         'GPU-94e7fdea-085e-eea4-86a5-83bbd6ad0883',
         'GPU-7aaea8dd-2241-2730-1234-96a9cd033e11',
         'GPU-72ca49d9-70ae-f881-70b3-a16295209e47']
TARGETS = sorted(f'language_model.model.layers.{i}.self_attn.attn'
                 for i in range(3, 48, 4))
DESCRIPTORS = [(20, 4, 5), (30, 6, 5), (40, 8, 5)]
BINARY_PINS = {
    'VLLM_FAV100_SO': (
        '/home/l/work/flash-next/prod-w13/'
        'flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so',
        'c707b819b00924fcfe0fc85721a8dc45c10a0e92ef10ec73aeb85248d7f059e7'),
    'ONECAT_FAV100_SO': (
        '/home/l/work/flash-next/prod-w48/'
        'flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so',
        'a2c3c845572e36a6eb788e53c1861734285e2512ffb12bc5b3c19909871e433b')}

PREFIXES = ('ONECAT_', 'VLLM_', 'NCCL_', 'CUDA_', 'TORCH_', 'TRITON_',
            'OMP_', 'MKL_', 'HF_', 'HOST_INSIGHT_', 'PYTORCH_', 'NUMA_',
            'TRANSFORMERS_', 'TORCHINDUCTOR_', 'KMP_', 'GOMP_', 'CUBLAS_', 'CUDNN_')

KEYS = {'R', 'D', 'MODEL', 'CODE', 'FAV100_SHIM', 'EXTRA_ARGS', 'PATH', 'HOME',
        'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'TZ', 'TMPDIR', 'PYTHONPATH',
        'LD_LIBRARY_PATH', 'VIRTUAL_ENV', 'PYTHONDONTWRITEBYTECODE',
        'PYTHONHASHSEED', 'CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS',
        'LD_PRELOAD', 'CMAKE_PREFIX_PATH', 'HOST', 'PORT', 'CTX', 'SEQS', 'MTP',
        'MTP_FP8', 'KVDTYPE', 'PLE', 'RETENTION', 'PACE', 'BATCHED'}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(16 * 1024**2):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def save(path, data):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as out:
        out.write(json.dumps(data, indent=2) + '\n')

def output(*argv):
    return subprocess.check_output(argv, text=True, stderr=subprocess.PIPE,
                                   timeout=20).strip()

def prepare(plan_path, expected_pid):
    plan = read(plan_path)
    if not plan['reviewed']:
        if os.environ.get('STEP58_SERVICE_APPROVED_PLAN_SHA256') != sha(plan_path):
            raise ValueError('parent approval of exact draft SHA256 required')
        plan = dict(plan, reviewed=True)
    validate_plan(plan)
    assert int(expected_pid) == plan['expected_pid']
    volume = os.statvfs(D)
    assert volume.f_bavail * volume.f_frsize >= 12 * 1024**3
    identities = output('nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader')
    assert identities.splitlines() == UUIDS
    before = read(D / 'baseline/before-snapshot.json')
    gates = ('health', 'source_config', 'kv_capacity', 'swap', 'cleanup',
             'worker_identity')
    if not all(before['checks'][key] for key in gates):
        raise ValueError('six fresh baseline gates must pass')
    if str(before['systemd']['MainPID']) != str(expected_pid):
        raise ValueError('expectedPID differs from fresh baseline')
    if output('systemctl', 'show', S, '-p', 'MainPID', '--value') != str(expected_pid):
        raise ValueError('live original PID changed')
    if output('systemctl', 'show', S, '-p', 'ActiveState', '--value') != 'active':
        raise ValueError('original service inactive')
    if output('git', '-C', str(K), 'rev-parse', 'HEAD') != HEAD:
        raise ValueError('serving source HEAD changed')
    if output('git', '-C', str(K), 'diff', 'HEAD', '--name-only'):
        raise ValueError('tracked serving tree dirty')
    unit = subprocess.check_output(['systemctl', 'cat', S], timeout=20)
    if hashlib.sha256(unit).hexdigest() != before['unit_sha256']:
        raise ValueError('original unit fingerprint changed')
    if unit != (D / 'baseline/before-unit.txt').read_bytes():
        raise ValueError('original unit byte identity changed')
    for relative, expected in before['sources'].items():
        if sha(K / relative) != expected:
            raise ValueError('serving runtime source changed')
    for path, expected in before['files'].items():
        if sha(path) != expected:
            raise ValueError('serving binary/shim changed')
    if list((Path('/run/systemd/system') / (S + '.d')).glob('*.conf')):
        raise ValueError('temporary original drop-ins present')
    active = output('systemctl', 'list-units', '--state=active', '--no-legend',
                    '--no-pager', 'step*', 'perf-profile*', '*guardian*')
    if active:
        raise ValueError('concurrent experiment unit present')
    orchestrators = output('systemctl', 'list-units', '--state=active',
        '--no-legend', '--no-pager', 'flash-next-recovery-orchestrator-*')
    if [row.split()[0] for row in orchestrators.splitlines()] != [OUTER]:
        raise ValueError('unexpected CPU recovery orchestrator present')
    state = Path('/home/l/.local/state/host-insight-mcp')
    if read(state / 'guardian/profile-guardian-state.json')['active']:
        raise ValueError('profile guardian active')
    terminal = {'completed', 'failed', 'cancelled', 'canceled', 'rejected'}
    jobs = read(state / 'experiments/experiment-state.json')['jobs'].values()
    if any(job['status'] not in terminal for job in jobs):
        raise ValueError('concurrent manager job active')
    for name in UNITS + [TIMER + '.timer']:
        if output('systemctl', 'show', name, '-p', 'LoadState',
                  '--value') != 'not-found':
            raise ValueError('owned unit identity already exists')
    with urllib.request.urlopen('http://127.0.0.1:8200/metrics', timeout=5) as response:
        lines = response.read().decode().splitlines()
    queue = [float(row.rsplit(' ', 1)[1]) for row in lines if row.startswith(
        ('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
    if len(queue) != 2 or any(queue):
        raise ValueError('original queue nonempty')
    gpu_procs = output('nvidia-smi', '--query-compute-apps=pid,process_name',
                       '--format=csv,noheader').splitlines()
    if len(gpu_procs) != 4 or any('/system.slice/' + S not in
            (Path('/proc') / row.split(',')[0].strip() / 'cgroup').read_text()
            for row in gpu_procs):
        raise ValueError('live GPU ownership differs from original cgroup')
    with socket.socket() as port:
        port.bind(('127.0.0.1', 8201))
    raw = (Path('/proc') / str(expected_pid) / 'environ').read_bytes()
    env = dict(row.decode().split('=', 1) for row in raw.split(b'\0') if b'=' in row)
    env = {key: value for key, value in env.items()
           if key in KEYS or key.startswith(PREFIXES)}
    if env.get('CODE') != str(K) or env.get('MODEL') != MODEL:
        raise ValueError('effective API code identity changed')
    for name, (path, expected) in BINARY_PINS.items():
        if env.get(name) != path or sha(path) != expected:
            raise ValueError('active/legacy binary override changed')
    if '--numa-bind-nodes 0 0 1 1' not in env.get('EXTRA_ARGS', ''):
        raise ValueError('original NUMA binding differs')
    sites = [Path(p) / 'sitecustomize.py' for p in env['FAV100_SHIM'].split(':')
             if (Path(p) / 'sitecustomize.py').is_file()]
    if sites != [Path('/home/l/work/flash-next/prod-w12/sitecustomize.py')]:
        raise ValueError('original shim chain differs')
    save(A / 'original-api-task-config.private.json', env)
    save(A / 'reviewed-plan.json', plan)
    save(A / 'preflight.json', {'status': 'PASS', 'pid': expected_pid,
        'head': HEAD, 'queue': queue, 'raw_flags_boolean': before['checks']['flags'],
        'unit_sha256': before['unit_sha256'], 'plan_sha256': sha(plan_path),
        'private_config_sha256': sha(A / 'original-api-task-config.private.json')})
    fd = os.open(D / 'restoration/pre-window-unit.txt',
                 os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'wb') as out:
        out.write(unit)
    print('PREFLIGHT_PASS private_config_values_retained')

def drain(label):
    started = time.monotonic()
    evidence = {'status': 'TIMEOUT', 'budget_s': 45}
    while time.monotonic() - started < 45:
        proc = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
            '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        evidence.update(exit=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        if proc.returncode == 0 and not proc.stdout.strip():
            evidence['status'] = 'DRAINED'
            break
        time.sleep(2)
    save(A / (label + '-drain.json'), evidence)
    if evidence['status'] != 'DRAINED':
        raise TimeoutError('GPU drain timeout')

def gpu_snapshot():
    return output('nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,name,'
                  'clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,memory.used',
                  '--format=csv')

def artifact_budget(label):
    files = [p for p in D.rglob('*') if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    smoke = sum(p.stat().st_size for p in files if p.suffix == '.pt'
                and p.is_relative_to(D / 'smoke1'))
    failure = sum(p.stat().st_size for p in files if p.suffix == '.pt'
                  and not p.is_relative_to(D / 'smoke1'))
    # Reserve space for this receipt, final notifications and cleanup receipt.
    allowance = 64 * 1024
    passed = (total + allowance <= 8 * 1024**3
              and failure <= 256 * 1024**2 and smoke <= 64 * 1024**2)
    save(A / f'artifact-budget-{label}.json', {
        'status': 'PASS' if passed else 'FAIL', 'scope': 'whole service-run3',
        'measured_bytes': total, 'receipt_cleanup_allowance_bytes': allowance,
        'upper_bound_bytes': total + allowance, 'limit_bytes': 8 * 1024**3,
        'failure_tensor_bytes': failure, 'failure_limit_bytes': 256 * 1024**2,
        'synthetic_smoke_tensor_bytes': smoke,
        'synthetic_smoke_limit_bytes': 64 * 1024**2,
        'threshold_only_not_journal_hard_limit': True,
        'files': len(files), 'no_tensor_read': True})
    return 0 if passed else 1


def inside(path):
    path = Path(path)
    assert path.is_absolute() and path.resolve().is_relative_to(D.resolve())
    return path


def phase(message):
    print(message, flush=True)
    try:
        os.write(3, (message + '\n').encode())
    except OSError:
        pass


def validate_plan(plan):
    assert plan['schema'] == 'step58-service-control-v1'
    assert plan['reviewed'] is True
    assert plan['budgets'] == {
        'boot_including_maturity_s': 960, 'maturity_s': 360, 'http_arm_s': 600,
        'inner_s': 3600, 'kill_grace_s': 20, 'expiry_s': 3660, 'outer_s': 6300,
        'restoration_s': 2100, 'smoke_s': 60, 'requests': 296, 'request_cap': 320}
    assert plan['units'] == UNITS and plan['outer'] == OUTER and plan['timer'] == TIMER
    assert plan['device_uuids'] == UUIDS
    previous = D.with_name('service-run1')
    assert plan['smoke']['required'] is False
    assert plan['smoke']['reuse'] == {
        'result': str(previous / 'smoke1/RESULT.json'),
        'samples': str(previous / 'control/attempt1/smoke-NVML-samples.json'),
        'resource_gate': str(previous / 'control/attempt1/smoke-resource-gate.json'),
        'proposal': str(ROOT / 'service-implementation/evidence/'
                        'GPU-SMOKE-PROPOSAL.final.json'),
        'unit': 'step58-service-smoke1.service'}
    for path, expected in plan['files'].items():
        assert re.fullmatch('[0-9a-f]{64}', expected) and sha(path) == expected
    contract = read(IMPL / 'contract.json')
    for pin in contract['supplemental_source_evidence'].values():
        assert plan['files'][pin['path']] == pin['sha256']
    prerequisites = plan['prerequisites']
    prior = read(prerequisites['supplemented_recovery'])
    assert prior['status'] == 'PASS' and len(prior['gates']) == 17
    assert all(prior['gates'].values())
    assert plan['recovery_sampling_boundary'] == {
        'before_concurrent': 'fresh/stable full-counter exporter fence',
        'after_concurrent': 'fresh/stable full-counter exporter fence',
        'passive_budget_each_s': 15, 'export_period_s': 1,
        'additional_requests': 0, 'existing_requests': 53,
        'fast_requests': 24, 'concurrent_groups': 4, 'fixed_requests': 3,
        'thresholds_changed': False}
    packets = read(inside(plan['packets']))
    assert packets['total_http_requests'] == 296
    assert packets['admission_request_cap'] == 320
    assert packets['cold_performance_claim'] is False
    assert packets['long_source']['sha256'] == (
        '49f84812ea46e3ba32171f6f57ed7ca6a5634c4b0324f2dcb41cfe9c5b32125e')
    identifiers = []
    for mode, count in (('off', 166), ('shadow', 130)):
        cfg = read(inside(plan['configs'][mode]))
        assert cfg['mode'] == mode and cfg['required_kv_tokens'] == 663816
        assert len(cfg['epochs']) == cfg['max_drains_per_rank'] == 20
        assert cfg['baseline_off_receipts'] == []
        assert cfg['diagnostic_host_audit'] is (mode == 'off')
        for key in ('output_dir', 'arm_file', 'drain_file', 'cohort_dir',
                    'baseline_off_capture_dir'):
            inside(cfg[key])
        for pin in cfg['source_pins'].values():
            assert plan['files'][pin['path']] == pin['sha256']
        arm = packets['arms'][mode]
        assert arm['http_requests'] == count and arm['timeout_s'] == 600
        assert len(arm['epochs']) == 20
        for row in arm['epochs']:
            epoch = cfg['epochs'][row['epoch']]
            assert [p['request_id'] for p in row['requests']] == (
                epoch['external_request_ids'])
            assert row['sentinel']['request_id'] == (
                epoch['sentinel_external_request_id'])
            for group in ('requests', 'outside_before', 'outside_after', 'sentinel'):
                rows = row[group] if group != 'sentinel' else [row[group]]
                for packet in rows:
                    identifiers.append(packet['header_id'])
                    body = packet['body']
                    assert body['request_id'] == packet['header_id']
                    assert body['model'] == 'flash-next' and body['stream'] is False
                    assert body['n'] == body['best_of'] == 1
                    assert body['temperature'] == 0 and body['logprobs'] == 1
                    assert 'prompt_logprobs' not in body
                    assert 1 <= body['max_tokens'] <= 16
                    assert all(type(t) is int and t >= 0 for t in body['prompt'])
    assert len(identifiers) == len(set(identifiers)) == 296
    gate = packets['gate']
    assert gate['script'] == str(IMPL / 'admission.py')
    assert gate['sha256'] == plan['files'][gate['script']]
    assert plan['generator'] == str(IMPL / 'baseline.py')
    return packets


def command(argv, log_path, seconds, cwd=D, env=None):
    started = time.time()
    result = {'argv': argv, 'started_epoch': started, 'budget_s': seconds}
    error = None
    with Path(log_path).open('xb') as log:
        try:
            proc = subprocess.run(argv, cwd=cwd, env=env, timeout=seconds,
                                  stdout=log, stderr=subprocess.STDOUT,
                                  close_fds=True, start_new_session=True)
            result['exit'] = proc.returncode
        except BaseException as caught:
            result.update(exit=124 if isinstance(caught, subprocess.TimeoutExpired)
                          else 1, error=repr(caught))
            error = caught
    result['ended_epoch'] = time.time()
    save(str(log_path) + '.exit.json', result)
    if error is not None:
        raise error
    assert result['exit'] == 0, f'bounded command failed: {Path(log_path).name}'
    return result


def candidate_environment(mode, config):
    study = A / mode
    study.mkdir(mode=0o700)
    for name in ('diagnostics', 'telemetry', 'traces'):
        (study / name).mkdir(mode=0o700)
    original = read(A / 'original-api-task-config.private.json')
    candidate = dict(original)
    candidate.update(HOST='127.0.0.1', PORT='8201', D=str(study),
        FAV100_SHIM=str(IMPL) + ':' + original['FAV100_SHIM'],
        STEP58_ORIGINAL_SHIM='/home/l/work/flash-next/prod-w12/sitecustomize.py',
        STEP58_SERVICE_CONFIG=str(config),
        STEP58_RAW_GENERATION_PATH=str(study / 'diagnostics/raw-generations.jsonl'),
        ONECAT_E7_STATS_DIR=str(study / 'telemetry/e7'),
        VLLM_TRITON_JIT_MANIFEST=str(study / 'triton-jit-manifest.jsonl'),
        PYTHONDONTWRITEBYTECODE='1')
    args, count = re.subn(r'(?<=--profiler-config.torch_profiler_dir=)\S+',
                         str(study / 'traces'), original['EXTRA_ARGS'])
    assert count == 1
    candidate['EXTRA_ARGS'] = args
    logging = read('/home/l/work/flash-next/diagnostics/logging.json')
    for name, handler in logging['handlers'].items():
        if 'filename' in handler:
            handler['filename'] = str(study / 'diagnostics' / (name + '.log'))
    logging['formatters']['step58_raw'] = {'format': '%(message)s'}
    logging['handlers']['step58_raw'] = {
        'class': 'logging.handlers.RotatingFileHandler',
        'filename': str(study / 'diagnostics/raw-generations.jsonl'),
        'maxBytes': 64 * 1024**2, 'backupCount': 7, 'encoding': 'utf-8',
        'formatter': 'step58_raw', 'level': 'INFO'}
    logging['loggers']['localai.raw_generation'] = {
        'handlers': ['step58_raw'], 'level': 'INFO', 'propagate': False}
    save(study / 'diagnostics/logging.json', logging)
    candidate['VLLM_LOGGING_CONFIG_PATH'] = str(study / 'diagnostics/logging.json')
    save(study / 'launch-task-config.private.json', candidate)
    effective = dict(candidate)
    effective['PYTHONPATH'] = candidate['FAV100_SHIM'] + ':' + candidate['CODE']
    effective['PATH'] = ('/home/l/work/qwen38-bench/cuda-home/bin:'
        '/home/l/.local/bin:/home/l/.cargo/bin:' + candidate['PATH'])
    effective['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    save(study / 'api-task-config.private.json', effective)
    path = study / 'api-task-config.private.env'
    with path.open('x') as out:
        for key, value in sorted(candidate.items()):
            assert re.fullmatch('[A-Z_][A-Z_0-9]*', key)
            assert not any(c in value for c in ('\n', '\r', '\0'))
            value = value.replace('\\', '\\\\').replace('"', '\\"')
            out.write(f'{key}="{value}"\n')
    path.chmod(0o600)
    return path


def start_candidate(mode, config):
    unit = UNITS[0 if mode == 'off' else 1]
    env_file = candidate_environment(mode, config)
    argv = ['sudo', '-n', 'systemd-run', '--collect', '--unit=' + unit,
        '--property=User=l', '--property=Group=l',
        '--property=WorkingDirectory=/home/l/work/flash-next',
        '--property=RuntimeMaxSec=1650s', '--property=TimeoutStopSec=30s',
        '--property=KillMode=control-group', '--property=MemorySwapMax=0',
        '--property=AllowedCPUs=0-13,15-41,43-55',
        '--property=EnvironmentFile=' + str(env_file),
        '/home/l/work/flash-next/run-flash-next-w12.sh']
    command(argv, A / mode / 'unit-start.log', 30)
    invocation = output('systemctl', 'show', unit, '-p', 'InvocationID', '--value')
    assert re.fullmatch('[0-9a-f]{32}', invocation)
    save(A / mode / 'startup-invocation.json', {
        'unit': unit, 'invocation_id': invocation, 'config_path': str(config),
        'config_sha256': sha(config)})
    for label, args in (
            ('unit', ['systemctl', 'show', unit]),
            ('gpu-before', ['nvidia-smi', '--query-gpu=index,uuid,memory.used,'
                'clocks.sm,clocks.mem,pstate', '--format=csv'])):
        (A / mode / (label + '.private.txt')).write_text(output(*args) + '\n')
        (A / mode / (label + '.private.txt')).chmod(0o600)


FATAL_MARKERS = ('EngineCore failed to start.', 'WorkerProc hit an exception.',
                 'Engine core initialization failed.')


def terminal_failure(mode, unit, invocation):
    paths = [D / 'control', A, A / mode / 'capture']
    errors = [str(p) for directory in paths
              for p in directory.glob('*error-pid*.json')]
    if errors:
        return {'kind': 'OWNED_HOOK_ERROR_RECEIPT', 'files': errors}
    argv = ['journalctl', '-u', unit, '_SYSTEMD_INVOCATION_ID=' + invocation,
            '-n', '200', '--no-pager', '--output=cat']
    result = subprocess.run(argv, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, 'cannot verify current engine startup journal'
    assert len(result.stdout.encode()) <= 1024**2, 'bounded journal byte ceiling'
    match = next((p for p in FATAL_MARKERS if p in result.stdout), None)
    if match:
        path = A / mode / 'terminal-engine-journal-tail.log'
        with path.open('x') as stream:
            stream.write(result.stdout)
        path.chmod(0o600)
        return {'kind': 'CURRENT_INVOCATION_TERMINAL_ENGINE', 'marker': match,
                'unit': unit, 'invocation_id': invocation,
                'journal_tail': str(path), 'journal_tail_sha256': sha(path)}
    return None


def ready(mode):
    started, deadline, health_at = time.monotonic(), time.monotonic() + 960, None
    unit = UNITS[0 if mode == 'off' else 1]
    invocation = read(A / mode / 'startup-invocation.json')
    assert invocation['unit'] == unit
    row = {'status': 'TIMEOUT', 'budget_s': 960, 'maturity_required_s': 360}
    while time.monotonic() < deadline:
        state = output('systemctl', 'show', unit, '-p', 'ActiveState', '--value')
        if state != 'active':
            row.update(status='FAILED', unit_state=state)
            break
        current = output('systemctl', 'show', unit, '-p', 'InvocationID', '--value')
        assert current == invocation['invocation_id'], 'startup invocation changed'
        failure = terminal_failure(mode, unit, current)
        if failure is not None:
            row.update(status='FAILED_TERMINAL_ENGINE', failure=failure)
            phase(f'STEP58_SERVICE_PHASE={mode.upper()}_TERMINAL_ENGINE_FAILURE')
            break
        try:
            with urllib.request.urlopen('http://127.0.0.1:8201/health', timeout=5) as r:
                assert r.status == 200
            if health_at is None:
                health_at = time.monotonic()
                row['actual_health_elapsed_s'] = health_at - started
            if time.monotonic() - health_at >= 360:
                row.update(status='READY_MATURE', mature_seconds=(
                    time.monotonic() - health_at))
                break
        except (OSError, urllib.error.URLError):
            health_at = None
        time.sleep(2)
    row['elapsed_s'] = time.monotonic() - started
    save(A / mode / 'readiness.json', row)
    assert row['status'] == 'READY_MATURE', 'candidate bounded boot/maturity failed'
    phase(f'STEP58_SERVICE_PHASE={mode.upper()}_READY_MATURE')


def candidate_gate(mode, config_path, template_path):
    cfg = read(config_path)
    directory = Path(cfg['output_dir'])
    assert not Path(cfg['arm_file']).exists() and not Path(cfg['drain_file']).exists()
    assert not list(directory.glob('*error-pid*.json'))
    assert not list(Path(config_path).parent.glob('*error-pid*.json'))
    unit = UNITS[0 if mode == 'off' else 1]
    pid = int(output('systemctl', 'show', unit, '-p', 'MainPID', '--value'))
    group = output('systemctl', 'show', unit, '-p', 'ControlGroup', '--value')
    assert group == '/system.slice/' + unit
    pids = sorted(int(x) for x in (Path('/sys/fs/cgroup') /
        group.lstrip('/') / 'cgroup.procs').read_text().split())
    assert pid in pids
    listen = [row.split()[9] for row in Path('/proc/net/tcp').read_text().splitlines()
              if '0100007F:2009' in row and row.split()[3] == '0A']
    assert len(listen) == 1, 'exact localhost8201 listener missing/ambiguous'
    listener_pids = []
    for candidate_pid in pids:
        for fd in (Path('/proc') / str(candidate_pid) / 'fd').iterdir():
            try:
                if os.readlink(fd) == 'socket:[' + listen[0] + ']':
                    listener_pids.append(candidate_pid)
            except FileNotFoundError:
                pass
    assert listener_pids, '8201 listener is outside owned cgroup'
    api_env = dict(x.decode().split('=', 1) for x in (
        Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0') if b'=' in x)
    expected_env = read(A / mode / 'api-task-config.private.json')
    for key, expected in expected_env.items():
        assert api_env.get(key) == expected, 'candidate effective API config differs'
    with urllib.request.urlopen('http://127.0.0.1:8201/v1/models', timeout=5) as r:
        assert [v['id'] for v in json.load(r)['data']] == ['flash-next']
    with urllib.request.urlopen('http://127.0.0.1:8201/metrics', timeout=5) as r:
        metrics = r.read(2 * 1024**2).decode()
    queue = [float(row.rsplit(' ', 1)[1]) for row in metrics.splitlines()
             if row.startswith(('vllm:num_requests_running{',
                                'vllm:num_requests_waiting{'))]
    assert len(queue) == 2 and not any(queue)
    text = subprocess.check_output(['journalctl', '-u', unit, '--no-pager'],
                                   text=True, timeout=20)
    capacities = re.findall(r'GPU KV cache size:\s*([\d,]+) tokens', text)
    assert capacities and int(capacities[-1].replace(',', '')) == 663816
    rows = [read(directory / f'capture-ready-rank{rank}.json') for rank in range(4)]
    assert len({r['pid'] for r in rows}) == 4
    for rank, row in enumerate(rows):
        assert row['status'] == 'CAPTURE_READY_SERVICE_UNVERIFIED'
        assert row['rank'] == rank and row['pid'] in pids and row['mode'] == mode
        assert row['config_sha256'] == sha(config_path)
        assert row['config_path'] == str(config_path)
        assert row['source_pins'] == cfg['source_pins']
        assert set(row['loaded_source_sha256']) == {
            'model', 'mtp', 'owner', 'ops', 'cg', 'marker', 'input_batch',
            'states', 'block_table', 'runner'}
        for name, pin in cfg['source_pins'].items():
            assert sha(pin['path']) == pin['sha256']
            if name in row['loaded_source_sha256']:
                assert row['loaded_source_sha256'][name] == pin['sha256']
        for field in ('candidate_binary_sha256', 'original_binary_sha256'):
            assert row[field] == cfg[field]
        assert sha(cfg['candidate_binary']) == cfg['candidate_binary_sha256']
        assert sha(cfg['original_binary']) == cfg['original_binary_sha256']
        stage = row['stage']
        assert stage['stage_installed_before_step58'] is True
        assert stage['class_execute_model_unmodified_by_step58'] is True
        assert stage['stage_source_sha256'] == cfg['source_pins']['marker']['sha256']
        assert stage['stage_wrapper_file'] == cfg['source_pins']['marker']['path']
        assert row['owners'] == TARGETS
        assert row['draft_owners_excluded'] == ['mtp.layers.48.self_attn.attn']
        nodes = {(tuple(v['descriptor']), v['owner']) for v in row['nodes']}
        assert nodes == {(key, owner) for key in DESCRIPTORS for owner in TARGETS}
        assert all(n['planner'] == ('original' if mode == 'off' else 'candidate')
                   for n in row['nodes'])
        assert row['epoch_contract'] == cfg['epochs']
        assert row['class_stage_chain_preserved'] is True
        assert row['runtime']['device_uuid'].removeprefix('GPU-') == (
            UUIDS[rank].removeprefix('GPU-'))
        assert row['runtime']['cuda_visible_devices'] == api_env['CUDA_VISIBLE_DEVICES']
        assert [v['actual_requests'] for v in row['consumer_table']] == (
            list(range(1, 9)))
        assert len(row['owner_bindings']) == 12
        assert len({v['object_id'] for v in row['owner_bindings']}) == 12
        assert all(v['object_id'] == v['static_context_object_id']
                   and v['module_path'] + '.attn' == v['layer_name']
                   for v in row['owner_bindings'])
        if mode == 'off':
            assert row['counter_shape'] is None
            assert row['private_bytes'] == row['failure_bank_bytes'] == 0
            assert row['outer_off_host_observer'] is True
        else:
            assert row['counter_shape'] == [36, 6]
            assert row['private_bytes'] <= cfg['max_private_bytes_per_rank']
            assert row['failure_bank_bytes'] <= cfg['max_failure_bank_bytes_per_rank']
            assert row['outer_shadow_execute_observer'] is True
    gpu_processes = output('nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                           '--format=csv,noheader')
    assert {int(r.split(',')[0]) for r in gpu_processes.splitlines()} == (
        {r['pid'] for r in rows})
    (A / mode / 'gpu-after-complete-capture.csv').write_text(gpu_snapshot() + '\n')
    receipt = {'status': 'PASS', 'mode': mode, 'api_pid': pid,
        'cgroup_path': group, 'cgroup_pids': pids, 'listener_pids': listener_pids,
        'queue': queue, 'kv_tokens': 663816,
        'config_sha256': sha(config_path), 'template_sha256': sha(template_path),
        'worker_pids': {str(r['rank']): r['pid'] for r in rows},
        'torch_device_uuids': {str(r['rank']): r['runtime']['device_uuid']
                             for r in rows},
        'device_uuids': dict(zip(map(str, range(4)), UUIDS)),
        'capture_receipts': [{'path': str(directory / (
                                 f"capture-ready-rank{r['rank']}.json")),
                             'sha256': sha(directory / (
                                 f"capture-ready-rank{r['rank']}.json"))}
                            for r in rows],
        'full_model_GPU_process_memory': gpu_processes,
        'raw_child_flag_proof': False}
    save(A / mode / 'startup-gate.json', receipt)
    return receipt


def stop_candidate(mode):
    unit = UNITS[0 if mode == 'off' else 1]
    result = subprocess.run(['sudo', '-n', 'systemctl', 'stop', unit],
                            timeout=50, capture_output=True, text=True, close_fds=True)
    save(A / mode / 'unit-stop.json', {'exit': result.returncode,
                                     'stdout': result.stdout, 'stderr': result.stderr})
    since = (A / 'controller-started.txt').read_text().strip()
    command(['sudo', '-n', 'journalctl', '-u', unit, '--since=' + since, '--no-pager'],
            A / mode / 'complete-journal.log', 30)
    drain(mode + '-after')
    assert result.returncode == 0
    phase(f'STEP58_SERVICE_PHASE={mode.upper()}_STOPPED_DRAINED')


def canonical_off(cfg, gate):
    from config_derivation import derive, BINDING_FIELDS
    root = Path(cfg['baseline_off_capture_dir'])
    root.mkdir(mode=0o700)
    bindings = []
    for rank in range(4):
        source = Path(cfg['output_dir']) / f'capture-ready-rank{rank}.json'
        dest = root / source.name
        with dest.open('xb') as stream:
            stream.write(source.read_bytes())
        dest.chmod(0o600)
        row = read(dest)
        binding = {key: row[key] for key in BINDING_FIELDS
                   if key not in ('path', 'sha256', 'device_uuid')}
        binding.update(path=str(dest), sha256=sha(dest),
                       device_uuid=row['runtime']['device_uuid'])
        bindings.append(binding)
    template = read(read(A / 'reviewed-plan.json')['configs']['shadow'])
    derive(template, bindings, gate)
    save(A / 'off-bindings.json', bindings)
    save(A / 'off-canonicalization.json', {'status': 'PASS_EXACT_BYTE_COPIES',
        'off_gate_sha256': sha(A / 'off/startup-gate.json'), 'bindings': bindings})


def smoke_gate(result, samples, proposal, unit=None):
    unit = UNITS[2] if unit is None else unit
    assert result['status'] == proposal['result']
    assert result['device'] == 'cuda:0'
    assert result['device_uuid'].removeprefix('GPU-') == (
        proposal['CUDA_VISIBLE_DEVICES'].removeprefix('GPU-'))
    assert 0 < result['allocated_peak'] <= proposal['allocated_gpu_cap']
    assert 0 < result['reserved_peak'] <= proposal['reserved_gpu_cap']
    assert 0 < result['cpu_peak_rss_bytes'] <= proposal['MemoryMax']
    valid = [row for sample in samples for row in sample.get('processes', [])
             if row.get('cgroup') == '/system.slice/' + unit]
    assert valid, 'no live owned smoke NVML context sample'
    for row in valid:
        assert row['CUDA_VISIBLE_DEVICES'] == proposal['CUDA_VISIBLE_DEVICES']
        assert row['memory_max'] == proposal['MemoryMax']
        assert row['swap_max'] == 0 and row['pid'] > 0
    maximum = max(row['used_memory_bytes'] for row in valid)
    assert 0 < maximum <= proposal['nvml_process_total_proposed_cap']
    return {'status': 'PASS_SAMPLED_SMOKE_RESOURCES_NOT_PEAK',
        'sampled_max_bytes': maximum, 'valid_live_samples': len(valid),
        'proposed_NVML_cap_bytes': proposal['nvml_process_total_proposed_cap'],
        'result': result, 'sampled_observation_not_peak': True}


def run():
    plan = read(A / 'reviewed-plan.json')
    validate_plan(plan)
    if plan['smoke'].get('reuse'):
        previous = plan['smoke']['reuse']
        assert plan['smoke']['required'] is False
        proposal = read(previous['proposal'])
        for row in proposal['source_files']:
            assert sha(IMPL / row['path']) == row['sha256']
        reused = smoke_gate(read(previous['result']),
                            read(previous['samples'])['samples'], proposal,
                            previous['unit'])
        old_gate = read(previous['resource_gate'])
        assert reused == old_gate
        save(A / 'smoke-reused-proof.json', {
            'status': 'REUSED_PRIOR_SYNTHETIC_PROOF_NO_GPU_RERUN',
            'prior_result_sha256': sha(previous['result']),
            'prior_resource_gate_sha256': sha(previous['resource_gate']),
            'unchanged_source_files': proposal['source_files'],
            'sampled_observation_not_peak': True,
            'new_service_graph_proven': False})
        phase('STEP58_SERVICE_PHASE=PRIOR_SYNTHETIC_SMOKE_REUSED')
    if plan['smoke']['required']:
        smoke = plan['smoke']
        assert smoke['argv'] == read(plan['smoke_proposal'])['argv']
        argv = ['sudo', '-n', 'systemd-run', '--wait', '--collect',
            '--unit=' + UNITS[2], '--property=User=l', '--property=Group=l',
            '--property=WorkingDirectory=' + str(IMPL),
            '--property=RuntimeMaxSec=60s', '--property=TimeoutStopSec=5s',
            '--property=KillMode=control-group', '--property=MemoryMax=2G',
            '--property=MemorySwapMax=0', '--property=AllowedCPUs=14',
            '--property=Environment=CUDA_VISIBLE_DEVICES=' + UUIDS[0],
            '--property=Environment=OMP_NUM_THREADS=1',
            '--property=Environment=MKL_NUM_THREADS=1',
            '--property=Environment=OPENBLAS_NUM_THREADS=1',
            '--property=Environment=PYTHONDONTWRITEBYTECODE=1',
            '--property=Environment=STEP58_ORIGINAL_SHIM='
                '/home/l/work/flash-next/prod-w12/sitecustomize.py'] + smoke['argv']
        finished = threading.Event()
        samples = []
        def sample_memory():
            while not finished.is_set():
                try:
                    raw = output('nvidia-smi',
                                 '--query-compute-apps=pid,used_gpu_memory',
                                 '--format=csv,noheader,nounits')
                    processes = []
                    for line in raw.splitlines():
                        pid, used = line.split(',')
                        pid = int(pid)
                        proc = Path('/proc') / str(pid)
                        group = proc.joinpath('cgroup').read_text().split('::', 1)[1]
                        group = group.strip()
                        cg = Path('/sys/fs/cgroup') / group.lstrip('/')
                        env = dict(v.decode().split('=', 1) for v in
                            proc.joinpath('environ').read_bytes().split(b'\0')
                            if b'=' in v)
                        processes.append({'pid': pid, 'cgroup': group,
                            'used_memory_bytes': int(used.strip()) * 1024**2,
                            'CUDA_VISIBLE_DEVICES': env.get('CUDA_VISIBLE_DEVICES'),
                            'memory_max': int((cg / 'memory.max').read_text()),
                            'swap_max': int((cg / 'memory.swap.max').read_text())})
                    samples.append({'epoch': time.time(), 'processes_csv': raw,
                                    'processes': processes})
                except BaseException as error:
                    samples.append({'epoch': time.time(), 'error': repr(error)})
                finished.wait(0.25)
        observer = threading.Thread(target=sample_memory)
        observer.start()
        try:
            command(argv, A / 'smoke.log', 80)
        finally:
            finished.set()
            observer.join(timeout=25)
            save(A / 'smoke-NVML-samples.json', {'samples': samples,
                'sampled_observation_not_peak': True, 'owned_unit': UNITS[2]})
        assert read(smoke['output'])['status'] == (
            'PASS_SYNTHETIC_SHADOW_GRAPH_NOT_SERVICE')
        try:
            resource_gate = smoke_gate(read(smoke['output']), samples,
                                       read(plan['smoke_proposal']))
        except BaseException as error:
            save(A / 'smoke-resource-gate.json', {'status': 'FAIL',
                 'error': repr(error), 'result_sha256': sha(smoke['output'])})
            raise
        save(A / 'smoke-resource-gate.json', resource_gate)
        drain('smoke-after')
        phase('STEP58_SERVICE_PHASE=SYNTHETIC_SMOKE_PASS')
    for mode in ('off', 'shadow'):
        template = Path(plan['configs'][mode])
        config = template
        if mode == 'shadow':
            from config_derivation import write_derived
            config = A / 'shadow.runtime.json'
            write_derived(template, A / 'off-bindings.json',
                          A / 'off/startup-gate.json', config,
                          A / 'shadow-derivation.json')
        start_candidate(mode, config)
        ready(mode)
        gate = candidate_gate(mode, config, template)
        cfg = read(config)
        if mode == 'off':
            canonical_off(cfg, gate)
        env = {k: v for k, v in os.environ.items() if not k.startswith('STEP58_')}
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        argv = ['timeout', '--signal=TERM', '--kill-after=20s', '600s', PY,
            str(D / 'control/http_cohort.py'), '--config', str(config),
            '--manifest', plan['packets'], '--startup-gate',
            str(A / mode / 'startup-gate.json'), '--output', cfg['cohort_dir']]
        command(argv, A / mode / 'http-driver.log', 625, env=env)
        assert read(Path(cfg['cohort_dir']) / 'complete.json')['http_requests'] == (
            {'off': 166, 'shadow': 130}[mode])
        assert artifact_budget(mode) == 0
        phase(f'STEP58_SERVICE_PHASE={mode.upper()}_COHORTS_COMPLETE')
        stop_candidate(mode)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=(
        'prepare', 'run', 'drain-before', 'drain-exit', 'artifact-final', 'validate'))
    parser.add_argument('--plan')
    parser.add_argument('--expected-pid')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.plan, args.expected_pid)
    elif args.action == 'run':
        run()
    elif args.action.startswith('drain-'):
        drain(args.action)
    elif args.action == 'artifact-final':
        raise SystemExit(artifact_budget('outer-final'))
    else:
        plan = read(args.plan)
        validate_plan(dict(plan, reviewed=True))
        print('PASS_ACTUAL_SERVICE_PLAN_BYTE_VALIDATION')


if __name__ == '__main__':
    main()
