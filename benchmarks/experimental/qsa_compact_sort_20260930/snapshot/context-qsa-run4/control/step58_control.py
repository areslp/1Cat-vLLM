"""Finite pure off/on/off cross-context window and exact owned recovery."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

PACKAGE = Path(__file__).resolve().parents[1]
LOCAL_BASE = PACKAGE.parent
sys.path.insert(0, str(LOCAL_BASE / 'service-w2-control-prep4'))
sys.path.insert(0, str(PACKAGE))
from guard_activation import activate, proof
activate()
from common import BASE, WINDOW, SERVICE, MODELS, HTTP, UNITS, TIMER, OUTER, PY, read
from execution_budget import BUDGETS, validate as budget_validate
from service_binding import derive, pure_config
from flow import Flow

D, A = WINDOW, WINDOW / 'control/attempt1'
MATRIX = BASE / 'context-qsa-prep1'
CFG_DIR = D / 'control/configs'
S = 'flash-next-vllm.service'
NATIVE_SOURCE = PACKAGE / 'source/native_ops.production.py'
NATIVE_SHA = '089f362e32253c2d2e3bcd136e51235f8dd95a4c69dac548493fe18854ee520e'
if hashlib.sha256(NATIVE_SOURCE.read_bytes()).hexdigest() != NATIVE_SHA:
    raise ValueError('exact frozen native operational helper source differs')
spec = importlib.util.spec_from_file_location('contextqsa_native_ops', NATIVE_SOURCE)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
native.D, native.A, native.ROOT, native.IMPL = D, A, BASE, SERVICE
native.UNITS, native.TIMER, native.OUTER = UNITS, TIMER, OUTER
from nvml_capture import capture_output
native.output = capture_output(native.output, A / 'nvml-native', 'native')
sha, save, output = native.sha, native.save, native.output


def require(value, message):
    if value is not True:
        raise ValueError(message)


def phase(message):
    print('CONTEXTQSA_PHASE=' + message, flush=True)


def command(argv, log, seconds, env=None):
    return native.command(argv, log, seconds, cwd=D, env=env)


def state(unit):
    keys = ('LoadState', 'ActiveState', 'SubState', 'MainPID', 'InvocationID',
            'ControlGroup', 'MemoryMax', 'MemorySwapMax', 'NRestarts')
    return dict(row.split('=', 1) for row in output('systemctl', 'show', unit,
                *['-p' + key for key in keys]).splitlines())


def validate_plan(plan):
    budget_validate()
    require(plan['schema'] == 'step58-contextqsa4-pure-ABA-control'
            and plan['window'] == str(D) and plan['units'] == UNITS
            and plan['timer'] == TIMER and plan['outer'] == OUTER
            and plan['budgets'] == BUDGETS and plan['arms'] == ['A0', 'B', 'A2']
            and plan['modes'] == ['off', 'on', 'off']
            and plan['inventory'] == read(MATRIX / 'CONTRACT.json')['inventory']
            and plan['resource_contract'] == read(PACKAGE / 'CONTRACT.json')['resources']
            and plan['identity_contract_sha256'] == proof()['identity_module_sha256']
            and plan['candidate_arms'] == 1 and plan['scheduler_observer'] is False
            and type(plan['expected_pid']) is int and plan['expected_pid'] > 0
            and re.fullmatch('[0-9a-f]{32}', plan['expected_invocation_id']) is not None,
            'exact finite contexts/pure ABA inventory and budgets required')
    require(not plan['pending_facts'], 'pending deployment facts fail closed')
    for path, expected in plan['files'].items():
        require(re.fullmatch('[0-9a-f]{64}', expected) is not None and sha(path) == expected,
                'plan dependency changed: ' + path)
    manifest = read(PACKAGE / 'MANIFEST.json')
    require(plan['files'].get(str(PACKAGE / 'MANIFEST.json')) == sha(PACKAGE / 'MANIFEST.json'),
            'reviewed control source manifest required')
    for row in manifest['files']:
        require(plan['files'].get(str(PACKAGE / row['path'])) == row['sha256'],
                'all control source pinned')
        if row['path'].split('/')[0] in ('control', 'restoration'):
            require(plan['files'].get(str(D / row['path'])) == row['sha256'],
                    'actual staged control/recovery input differs')
    for row in read(PACKAGE / 'DEPENDENCIES.json')['files']:
        require(plan['files'].get(str(BASE / row['relative_path'])) == row['sha256'],
                'all immutable matrix/guard/source dependencies pinned')
    import restoration_dependencies
    pins = restoration_dependencies.validate(D)
    require(pins == read(PACKAGE / 'source/restoration-external-reference-pins.json'),
            'unchanged complete53 restoration references')
    for path, expected in pins.items():
        require(plan['files'].get(path) == expected, 'external restoration reference unpinned')
    for name in ('before-snapshot.json', 'before-unit.txt'):
        path = str(D / 'baseline' / name)
        require(plan['files'].get(path) == sha(path), 'fresh full baseline unpinned')
    for arm in ('A0', 'A2'):
        path = CFG_DIR / (arm + '.service.json')
        require(read(path) == derive(arm, D) and plan['files'].get(str(path)) == sha(path),
                'exact owned pure original config required')
    require(not (CFG_DIR / 'B.service.json').exists(),
            'B config is create-only and must bind this window fresh A0')
    require(plan['matrix_path'] == str(MATRIX / 'matrix.frozen.json'),
            'exact reviewed context matrix path required')


def prepare(plan_path, expected_pid):
    approved = os.environ.get('STEP58_CONTEXTQSA4_APPROVED_PLAN_SHA256')
    require(approved == sha(plan_path), 'root review of exact plan SHA required')
    plan = read(plan_path)
    require(plan['expected_pid'] == expected_pid, 'fresh expected production PID differs')
    validate_plan(plan)
    # Preserve this check before native's known-key filtering.
    raw = dict(row.decode().split('=', 1) for row in
        (Path('/proc') / str(expected_pid) / 'environ').read_bytes().split(b'\0') if b'=' in row)
    require(not any(k.startswith(('STEP58_', 'PHYSICAL_', 'STEP56_', 'STEP57_')) for k in raw),
            'original raw API environment already experimental')
    native.validate_plan = validate_plan
    os.environ['STEP58_SERVICE_APPROVED_PLAN_SHA256'] = approved
    native.prepare(plan_path, expected_pid)
    props = state(S)
    require(props['MainPID'] == str(expected_pid) and props['ActiveState'] == 'active'
            and props['InvocationID'] == plan['expected_invocation_id'],
            'fresh production invocation/PID changed during preflight')
    save(A / 'preflight-production-invocation.json', {'status': 'FRESH_BOUND',
        'pid': expected_pid, 'invocation_id': props['InvocationID'],
        'reviewed_plan_sha256': sha(plan_path)})
    phase('PREFLIGHT_PASS')


def ready(arm):
    started, health_at, first_pid = time.monotonic(), None, None
    unit = MODELS[arm]
    invocation = read(A / arm / 'startup-invocation.json')['invocation_id']
    row = {'status': 'TIMEOUT', 'budget_s': BUDGETS['ready_s'],
           'maturity_required_s': BUDGETS['maturity_s']}
    while time.monotonic() - started < BUDGETS['ready_s']:
        props = state(unit)
        require(props['ActiveState'] == 'active' and props['InvocationID'] == invocation
                and props['NRestarts'] == '0', 'owned startup died/replaced/restarted')
        failure = native.terminal_failure(arm, unit, invocation)
        require(failure is None, 'current invocation terminal engine failure: ' + repr(failure))
        require(not list((A / arm / 'service-capture').glob('*error-pid*.json')),
                'owned pure capture hook failure')
        try:
            with urllib.request.urlopen('http://127.0.0.1:8201/health', timeout=3) as response:
                require(response.status == 200, 'owned clone health status')
            pid = int(props['MainPID'])
            ticks = native_pid_start(pid)
            if health_at is None:
                health_at, first_pid = time.monotonic(), (pid, ticks)
            require((pid, ticks) == first_pid, 'owned API changed during maturity')
            if time.monotonic() - health_at >= BUDGETS['maturity_s']:
                row.update(status='READY_MATURE', mature_seconds=time.monotonic() - health_at)
                break
        except (OSError, urllib.error.URLError):
            require(health_at is None, 'clone health lost after first ready')
        time.sleep(2)
    row['elapsed_s'] = time.monotonic() - started
    save(A / arm / 'readiness.json', row)
    require(row['status'] == 'READY_MATURE', 'bounded1080s readiness/maturity failed')


def native_pid_start(pid):
    from original_guard import Observation
    return Observation.pid_start(pid)


def start(arm, cfg_path):
    cfg = pure_config(read(cfg_path))
    require(cfg['mode'] == ('on' if arm == 'B' else 'off'), 'pure arm mode differs')
    env = native.candidate_environment(arm, cfg_path)
    require(Path(cfg['output_dir']) == A / arm / 'service-capture', 'exact owned capture path')
    Path(cfg['output_dir']).mkdir(mode=0o700)
    command(['sudo', '-n', 'systemd-run', '--collect', '--unit=' + MODELS[arm],
        '--property=User=l', '--property=Group=l',
        '--property=WorkingDirectory=/home/l/work/flash-next',
        '--property=RuntimeMaxSec=' + str(BUDGETS['model_runtime_s']) + 's',
        '--property=TimeoutStopSec=30s', '--property=KillMode=control-group',
        '--property=MemoryMax=120G', '--property=MemorySwapMax=0',
        '--property=AllowedCPUs=0-13,15-41,43-55', '--property=EnvironmentFile=' + str(env),
        '/home/l/work/flash-next/run-flash-next-w12.sh'], A / arm / 'unit-start.log', 30)
    props = state(MODELS[arm])
    invocation = props['InvocationID']
    require(re.fullmatch('[0-9a-f]{32}', invocation) is not None,
            'new actual clone invocation required')
    prior = [read(A / old / 'startup-invocation.json')['invocation_id']
             for old in ('A0', 'B', 'A2') if old != arm
             and (A / old / 'startup-invocation.json').exists()]
    original = read(A / 'preflight-production-invocation.json')
    require(invocation not in prior and invocation != original['invocation_id'],
            'all arm invocations must be fresh and independent')
    save(A / arm / 'startup-invocation.json', {'unit': MODELS[arm],
        'invocation_id': invocation, 'config_path': str(cfg_path), 'config_sha256': sha(cfg_path)})
    ready(arm)
    command([PY, '-I', '-B', str(PACKAGE / 'collect_startup.py'), '--arm', arm,
        '--unit', MODELS[arm], '--config', str(cfg_path),
        '--expected-api-environment', str(A / arm / 'api-task-config.private.json'),
        '--study', str(A / arm)], A / arm / 'startup-collector.log', 60)
    model = A / arm / 'model'
    model.mkdir(mode=0o700)
    shutil.copyfile(A / arm / 'guard-binding.json', model / 'binding.json')
    require(sha(model / 'binding.json') == sha(A / arm / 'guard-binding.json'),
            'worker-facing full binding must preserve exact startup bytes')
    phase(arm + '_MATURE_CAPTURE_BOUND')
    return read(A / arm / 'startup-gate.json')


def derive_b(startup):
    bindings = []
    for rank, pin in enumerate(startup['capture_receipts']):
        row = read(pin['path'])
        bindings.append({'rank': rank, 'path': pin['path'], 'sha256': pin['sha256'],
            'pid': row['pid'], 'config_sha256': row['config_sha256'],
            'candidate_binary_sha256': row['candidate_binary_sha256'],
            'original_binary_sha256': row['original_binary_sha256'],
            'device_uuid': row['runtime']['device_uuid'], 'kv_num_blocks': row['kv_num_blocks']})
    path = A / 'A0/startup-gate.json'
    cfg = derive('B', D, off_bindings=bindings, off_dir=startup['service_capture_dir'],
                 off_startup={'path': str(path), 'sha256': sha(path)})
    target = CFG_DIR / 'B.service.json'
    save(target, cfg)
    save(A / 'B-derived-fresh-A0.json', {'status': 'FRESH_A0_SOURCE_BOUND_B_DERIVED',
        'B_config_path': str(target), 'B_config_sha256': sha(target),
        'A0_startup_path': str(path), 'A0_startup_sha256': sha(path),
        'baseline_off_receipts': bindings, 'reviewed_plan_sha256': sha(D / 'control/plan.json')})
    return target


def timed(arm):
    argv = ['sudo', '-n', 'systemd-run', '--wait', '--collect', '--unit=' + HTTP[arm],
        '--property=User=l', '--property=Group=l', '--property=WorkingDirectory=' + str(D),
        '--property=RuntimeMaxSec=' + str(BUDGETS['client_runtime_s']) + 's',
        '--property=TimeoutStopSec=5s', '--property=KillMode=control-group',
        '--property=MemoryMax=1G', '--property=MemorySwapMax=0',
        '--property=AllowedCPUs=14,42', '--property=CPUQuota=200%', '--property=Nice=15',
        '--property=Environment=CUDA_VISIBLE_DEVICES=',
        '--property=Environment=PYTHONDONTWRITEBYTECODE=1',
        '--property=Environment=OMP_NUM_THREADS=1', '--property=Environment=MKL_NUM_THREADS=1',
        '--property=Environment=OPENBLAS_NUM_THREADS=1', PY, '-I', '-B',
        str(PACKAGE / 'http_runner.py'), '--binding', str(A / arm / 'model/binding.json'),
        '--output', str(A / arm / 'http'), '--arm', arm,
        '--matrix', str(MATRIX / 'matrix.frozen.json')]
    command(argv, A / arm / 'http-unit.log', BUDGETS['client_wrapper_s'])
    ended = state(HTTP[arm])
    require(ended['ActiveState'] in ('inactive', 'failed') and ended['MainPID'] == '0',
            'all owned client processes must end before next arm')
    save(A / arm / 'http-unit-ended.json', ended)
    command([PY, '-I', '-B', str(PACKAGE / 'final_guard.py'), '--arm', arm],
            A / arm / 'final-guard.log', BUDGETS['final_guard_s'])


def stop(arm, startup):
    binding = read(A / arm / 'model/binding.json')
    require(startup['arm'] == arm and startup['unit'] == MODELS[arm],
            'startup binding names exact stopping arm')
    command(['sudo', '-n', 'systemctl', 'stop', MODELS[arm]],
            A / arm / 'model-stop.log', BUDGETS['stop_s'])
    native.drain('arm-' + arm.lower() + '-stop')
    props = state(MODELS[arm])
    require(props['MainPID'] == '0' and props['ActiveState'] in ('inactive', 'failed'),
            'same owned model still alive')
    identities = []
    for row in [{'pid': binding['api_pid'], 'starttime_ticks': binding['api_starttime']},
                *binding['workers']]:
        present = (Path('/proc') / str(row['pid'])).exists()
        ticks = native_pid_start(row['pid']) if present else None
        require(ticks != row['starttime_ticks'], 'same owned API/worker survived stop')
        identities.append(dict(row, present_after_stop=present, new_starttime=ticks))
    listener = output('ss', '-H', '-ltnp', 'sport = :8201')
    require(not listener, '8201 listener survived owned arm stop')
    proof = {'status': 'PASS_OWNED_ARM_STOPPED_GPU_EMPTY', 'arm': arm,
        'startup_sha256': sha(A / arm / 'startup-gate.json'), 'identities': identities,
        'unit': props, 'post_stop_8201_listener': listener,
        'GPU_empty_receipt_sha256': sha(A / ('arm-' + arm.lower() + '-stop-drain.json'))}
    save(A / arm / 'worker-stop.json', proof)
    return proof


def run():
    flow, off = Flow(), None
    try:
        for arm in ('A0', 'B', 'A2'):
            path = derive_b(off) if arm == 'B' else CFG_DIR / (arm + '.service.json')
            flow.begin(arm)
            startup = start(arm, path)
            if arm == 'A0':
                off = startup
            timed(arm)
            flow.stop(arm, stop(arm, startup))
        flow.analyzed()
        phase('ALL_THREE_ARMS_STOPPED_COMPLETE_HTTP_ONLY')
        analyze()
    except BaseException as error:
        flow.fail(error)
        raise
    finally:
        save(A / 'flow.json', dict(vars(flow), restore_required=True,
             no_performance_or_new_numeric_admission=True))


def analyze():
    argv = [PY, '-I', '-B', str(MATRIX / 'analyze.py'), '--window', str(D)]
    log = A / 'analysis.log'
    began = time.time()
    error = None
    code = 125
    with log.open('xb') as stream:
        try:
            result = subprocess.run(argv, cwd=D, timeout=BUDGETS['analysis_s'],
                stdout=stream, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
            code = result.returncode
        except BaseException as caught:
            code = 124 if isinstance(caught, subprocess.TimeoutExpired) else 1
            error = caught
    save(str(log) + '.exit.json', {'argv': argv, 'exit': code,
        'started_epoch': began, 'ended_epoch': time.time(), 'budget_s': BUDGETS['analysis_s'],
        'error': repr(error) if error else None})
    if error:
        raise error
    receipt = read(D / 'analysis/RESULT.json')
    require(receipt['diagnostic_exit'] == code and code in (0, 2, 3),
            'analysis result and native decision exit differ')
    require(code == 0, 'native output comparison STOP retained: exit' + str(code))


def cleanup(label, drain_gpu):
    require(label in ('exit-final', 'outer-before-restore'), 'finite cleanup label')
    require(drain_gpu is (label == 'exit-final'), 'outer cannot drain restored production GPU')
    failures, rows = [], []
    for unit in UNITS:
        row = {'unit': unit}
        try:
            row['before'] = state(unit)
            if row['before']['LoadState'] == 'not-found':
                row['stop'] = 'NOT_LOADED_TYPED_SKIP'
            else:
                try:
                    command(['sudo', '-n', 'systemctl', 'stop', unit],
                            A / (label + '-' + unit + '.stop.log'), 50)
                except BaseException as error:
                    failures.append({'unit': unit, 'phase': 'stop', 'error': repr(error)})
                row['after'] = state(unit)
                require(row['after']['MainPID'] == '0' and row['after']['ActiveState'] in
                        ('inactive', 'failed'), 'same owned model/client still alive')
            command(['sudo', '-n', 'journalctl', '-u', unit,
                '--since=' + (A / 'controller-started.txt').read_text().strip(),
                '-n', '5000', '--no-pager'], A / (label + '-' + unit + '.journal.log'), 20)
        except BaseException as error:
            failures.append({'unit': unit, 'phase': 'identity/journal', 'error': repr(error)})
        rows.append(row)
    if drain_gpu:
        try:
            native.drain(label)
        except BaseException as error:
            failures.append({'phase': 'GPU_drain', 'error': repr(error)})
    save(A / (label + '-cleanup.json'), {'status': 'PASS' if not failures else 'FAIL',
        'rows': rows, 'errors': failures, 'original_restore_must_continue': True})
    return 0 if not failures else 1


def artifact_budget(label):
    total = 0
    for path in D.rglob('*'):
        require(not path.is_symlink(), 'unapproved owned artifact symlink')
        if path.is_file():
            total += path.stat().st_size
    limit = 8 * 1024**3
    save(A / ('artifact-' + label + '.json'), {'status': 'PASS' if total + 65536 <= limit else 'FAIL',
        'measured_bytes': total, 'threshold_bytes': limit, 'receipt_allowance_bytes': 65536,
        'hard_FS_quota': False, 'restore_writes_must_continue': True})
    return 0 if total + 65536 <= limit else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'drain', 'run', 'cleanup', 'artifacts'])
    parser.add_argument('--plan')
    parser.add_argument('--expected-pid', type=int)
    parser.add_argument('--label')
    args = parser.parse_args()
    try:
        if args.mode == 'prepare':
            prepare(args.plan, args.expected_pid)
        elif args.mode == 'drain':
            native.drain(args.label)
        elif args.mode == 'run':
            run()
        elif args.mode == 'cleanup':
            return cleanup(args.label, args.label == 'exit-final')
        else:
            return artifact_budget('final')
    except BaseException as error:
        if A.is_dir():
            save(A / (args.mode + '-FAILURE.json'), {'error': repr(error),
                'traceback': traceback.format_exc()[-16384:], 'restore_required': True})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
