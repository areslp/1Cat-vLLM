"""Pure service orchestration plus actual startup; no diagnostic model observer."""
import contextlib
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import time

from execution_budget import BUDGETS, model_runtime_seconds
from common import BASE, PACKAGE, WINDOW, SERVICE, MODELS, HTTP, UNITS, TIMER, OUTER, PY, read, save, sha, cpu_env
from frozen import BASE as LOCAL_BASE
from artifact_budget import free_space_preflight
from service_binding import pure_config
from startup import collect

NATIVE = LOCAL_BASE / 'service-numeric-candidate-B-control-prep1/source/run3_control.production.py'
NATIVE_SHA = '089f362e32253c2d2e3bcd136e51235f8dd95a4c69dac548493fe18854ee520e'
if sha(NATIVE) != NATIVE_SHA:
    raise ValueError('exact frozen operational helper source differs')
spec = importlib.util.spec_from_file_location('w2_original_ops_source', NATIVE)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
native.D, native.A, native.ROOT, native.IMPL = WINDOW, WINDOW / 'control/attempt1', BASE, SERVICE
native.OUTER, native.TIMER, native.UNITS = OUTER, TIMER, list(UNITS.values())
_command = native.command
def command(argv, path, seconds, cwd=WINDOW, env=None):
    return _command(argv, path, seconds, cwd=cwd, env=env)
native.command = command


def analyze():
    """Retain native analysis exits2/3 as decisions, never fabricated exit0."""
    argv = [PY, '-B', str(PACKAGE / 'analysis.py'), '--window', str(WINDOW)]
    log = native.A / 'analysis.log'
    started = time.time()
    with log.open('xb') as stream:
        result = subprocess.run(argv, cwd=WINDOW, env=cpu_env(), timeout=60,
                                stdout=stream, stderr=subprocess.STDOUT,
                                close_fds=True, start_new_session=True)
    save(str(log) + '.exit.json', {'argv': argv, 'started_epoch': started,
         'ended_epoch': time.time(), 'budget_s': 60, 'exit': result.returncode})
    value = read(WINDOW / 'analysis/RESULT.json', 4 * 1024**2)
    expected = {'NO_GO': 2, 'INCONCLUSIVE': 3,
                'PASS_INITIAL_REQUIRES_SEPARATE_APPROVED_CONFIRMATION': 0}
    if expected.get(value['status']) != result.returncode:
        raise ValueError('analysis decision/result/native exit inconsistent')
    return value, result.returncode


def artifact_budget(label):
    # In recovery this only reports/exits; it never prevents best-effort
    # restoration writes. A measured cap violation remains a final failure.
    from io_tools import tree_bytes, WINDOW_CAP
    total = tree_bytes(WINDOW)
    passed = total + 65536 <= WINDOW_CAP
    save(native.A / f'artifact-budget-{label}.json', {
        'status': 'PASS' if passed else 'FAIL', 'measured_bytes': total,
        'window_measured_cap_bytes': WINDOW_CAP, 'receipt_allowance_bytes': 65536,
        'scope': 'Whole owned window: configs/startup/capture/service logs/request raw+JSON/process logs/analysis/restore/control; external immutable source packages are separately pinned.',
        'filesystem_quota': False, 'restoration_writes_must_continue_on_cap_failure': True})
    return 0 if passed else 1
native.artifact_budget = artifact_budget


@contextlib.contextmanager
def current_unit(arm):
    # Pinned native.ready chooses slot0/1. Bind both to this exact arm and
    # always restore all seven owned units, including exceptions.
    native.UNITS = [MODELS[arm], MODELS[arm]]
    try:
        yield
    finally:
        native.UNITS = list(UNITS.values())


def environment(arm, path):
    cfg = pure_config(read(path))
    if cfg['mode'] != ('on' if arm == 'B' else 'off'):
        raise ValueError('pure config arm differs')
    # Preserve actual mkdir ordering: native creates study/log/config files
    # before the sole extra service-capture directory is created.
    result = native.candidate_environment(arm, path)
    study = native.A / arm
    if Path(cfg['output_dir']) != study / 'service-capture':
        raise ValueError('pure capture directory outside current owned arm')
    Path(cfg['output_dir']).mkdir(mode=0o700)
    effective = read(study / 'api-task-config.private.json')
    if any(key.startswith(('STEP58_NUMERIC_', 'STEP58_ADMISSION_', 'PHYSICAL_FIXTURE_'))
           for key in effective):
        raise ValueError('diagnostic injection into timed environment')
    return result


def boot_resource(arm):
    from identity_contract import MODEL_CGROUP_BYTES, DEVICE_NVML_MIB, PROCESS_NVML_MIB
    unit = MODELS[arm]
    keys = ('ControlGroup', 'MemoryCurrent', 'MemoryPeak', 'MemoryMax',
            'MemorySwapCurrent', 'MemorySwapMax')
    props = dict(line.split('=', 1) for line in native.output('systemctl', 'show', unit,
                         *['-p' + key for key in keys]).splitlines())
    if (props['ControlGroup'] != '/system.slice/' + unit
            or int(props['MemoryMax']) != MODEL_CGROUP_BYTES
            or any(int(props[key]) > MODEL_CGROUP_BYTES for key in ('MemoryCurrent', 'MemoryPeak'))
            or any(int(props[key]) != 0 for key in ('MemorySwapCurrent', 'MemorySwapMax'))):
        raise ValueError('boot actual model cgroup memory/swap differs')
    cg = Path('/sys/fs/cgroup') / props['ControlGroup'].lstrip('/')
    events = dict(line.split() for line in (cg / 'memory.events').read_text().splitlines())
    if any(int(events[key]) for key in ('oom', 'oom_kill')):
        raise ValueError('boot model OOM')
    devices = native.output('nvidia-smi', '--query-gpu=uuid,memory.used',
                            '--format=csv,noheader,nounits')
    seen = {}
    for line in devices.splitlines():
        uuid, memory = [value.strip() for value in line.split(',')]
        if not re.fullmatch('[0-9]+', memory) or uuid in seen:
            raise ValueError('boot unknown/duplicate device sample')
        seen[uuid] = int(memory)
    if set(seen) != set(native.UUIDS) or any(not 0 <= v <= DEVICE_NVML_MIB for v in seen.values()):
        raise ValueError('boot four UUID/device memory cap differs')
    processes = native.output('nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                             '--format=csv,noheader')
    for line in processes.splitlines():
        pid, uuid, memory = [value.strip() for value in line.split(',')]
        if (not re.fullmatch('[0-9]+', pid) or uuid not in seen
                or not re.fullmatch('[0-9]+ MiB', memory)
                or not 0 <= int(memory.split()[0]) <= PROCESS_NVML_MIB
                or (Path('/proc') / pid / 'cgroup').read_text().strip() != '0::' + props['ControlGroup']):
            raise ValueError('boot foreign PID/UUID/cgroup or memory')
    return {'model_cgroup': props, 'events': events, 'devices_MiB': seen,
            'processes_raw': processes, 'boot_zero_context_memory_allowed': True}


def start(arm, path):
    file = environment(arm, path)
    # B includes separately retained pool warmup/import preparation as well
    # as its900s sustained section. This bound is inside the15300s inner cap.
    runtime = model_runtime_seconds(arm)
    command(['sudo', '-n', 'systemd-run', '--collect', '--unit=' + MODELS[arm],
        '--property=User=l', '--property=Group=l',
        '--property=WorkingDirectory=/home/l/work/flash-next',
        '--property=RuntimeMaxSec=' + str(runtime) + 's', '--property=TimeoutStopSec=30s',
        '--property=KillMode=control-group', '--property=MemoryMax=120G',
        '--property=MemorySwapMax=0', '--property=AllowedCPUs=0-13,15-41,43-55',
        '--property=EnvironmentFile=' + str(file),
        '/home/l/work/flash-next/run-flash-next-w12.sh'], native.A / arm / 'unit-start.log', 30)
    invocation = native.output('systemctl', 'show', MODELS[arm], '-pInvocationID', '--value')
    if not re.fullmatch('[0-9a-f]{32}', invocation):
        raise ValueError('owned invocation missing')
    save(native.A / arm / 'startup-invocation.json', {'unit': MODELS[arm],
        'invocation_id': invocation, 'config_path': str(path), 'config_sha256': sha(path)})
    prior = native.terminal_failure
    samples = []
    def terminal(mode, unit, invocation):
        result = prior(mode, unit, invocation)
        samples.append(boot_resource(arm))
        if len(samples) > 480:
            raise ValueError('boot resource sample count cap')
        cfg = read(path)
        if list(Path(cfg['output_dir']).glob('*error-pid*.json')):
            raise ValueError('owned service hook failure before maturity')
        return result
    native.terminal_failure = terminal
    try:
        with current_unit(arm):
            native.ready(arm)
    finally:
        native.terminal_failure = prior
        save(native.A / arm / 'boot-resource-samples.json', samples, 4 * 1024**2)
    return collect(arm, MODELS[arm], path,
                   native.A / arm / 'api-task-config.private.json', native.A / arm)


def timed(arm, phase='matrix'):
    if phase == 'matrix':
        script, duration, unit = 'matrix.py', BUDGETS['matrix_per_arm'], HTTP[arm]
    elif arm == 'B' and phase == 'stability':
        script, duration, unit = 'stability.py', 1110, HTTP['stability']
    else:
        raise ValueError('only frozen three matrices and one B stability')
    study = native.A / arm
    args = [PY, '-B', str(PACKAGE / script), '--binding', str(study / 'guard-binding.json'),
            '--output', str(study / phase), '--window', str(WINDOW)]
    if phase == 'matrix':
        args += ['--arm', arm]
    # All request children inherit this same 2GiB/1CPU cgroup. READY/imports
    # precede measured release; no resource/health probes occur in timed kernels.
    argv = ['sudo', '-n', 'systemd-run', '--wait', '--collect', '--unit=' + unit,
        '--property=User=l', '--property=Group=l', '--property=WorkingDirectory=' + str(WINDOW),
        '--property=RuntimeMaxSec=' + str(duration + 10) + 's',
        '--property=TimeoutStopSec=5s', '--property=KillMode=control-group',
        '--property=MemoryMax=2G', '--property=MemorySwapMax=0',
        '--property=AllowedCPUs=14,42', '--property=Nice=15', '--property=CPUQuota=100%',
        '--property=Environment=CUDA_VISIBLE_DEVICES=',
        '--property=Environment=OMP_NUM_THREADS=1', '--property=Environment=MKL_NUM_THREADS=1',
        '--property=Environment=OPENBLAS_NUM_THREADS=1',
        '--property=Environment=PYTHONDONTWRITEBYTECODE=1', *args]
    command(argv, study / (phase + '-unit.log'), duration + 30, env=cpu_env())
    state = native.output('systemctl', 'show', unit, '-pActiveState', '-pMainPID', '-pControlGroup')
    fields = dict(line.split('=', 1) for line in state.splitlines())
    if fields['ActiveState'] not in ('inactive', 'failed') or fields['MainPID'] != '0':
        raise ValueError('HTTP unit/process tree did not terminate')
    save(study / (phase + '-unit-ended.json'), {'unit': unit, 'raw_properties': fields,
        'status': 'ENDED', 'memory_peak_after_GC': 'UNKNOWN_UNLESS_RETAINED_BY_SYSTEMD'})


def stop(arm, startup):
    if startup['arm'] != arm or startup['unit'] != MODELS[arm]:
        raise ValueError('actual startup binding does not name stopping arm')
    command(['sudo', '-n', 'systemctl', 'stop', MODELS[arm]],
            native.A / arm / 'model-stop.log', 50)
    native.drain('arm-' + arm.lower() + '-stop')  # Unique once per arm.
    fields = dict(line.split('=', 1) for line in native.output('systemctl', 'show', MODELS[arm],
        '-pActiveState', '-pMainPID', '-pControlGroup').splitlines())
    if fields['ActiveState'] not in ('inactive', 'failed') or fields['MainPID'] != '0':
        raise ValueError('stopped model unit is still alive')
    identities = []
    for rank in range(4):
        pid = startup['worker_pids'][str(rank)]
        # The explicit binding has exact process starttimes, not guessed PIDs.
        binding = read(native.A / arm / 'guard-binding.json')
        ticks = binding['workers'][rank]['starttime_ticks']
        proc = Path('/proc') / str(pid)
        present = proc.exists()
        new_ticks = None
        if present:
            from original_guard import Observation
            new_ticks = Observation.pid_start(pid)
            if ticks == new_ticks:
                raise ValueError('same owned rank worker survived stop')
        identities.append({'rank': rank, 'pid': pid, 'original_starttime': ticks,
                           'PID_present_after_stop': present, 'observed_new_starttime': new_ticks})
    proof = {'status': 'PASS_OWNED_ARM_STOPPED_GPU_EMPTY', 'arm': arm,
        'startup_sha256': sha(native.A / arm / 'startup-gate.json'),
        'worker_identities': identities, 'unit': fields,
        'GPU_empty_receipt_sha256': sha(native.A / ('arm-' + arm.lower() + '-stop-drain.json'))}
    from original_guard import Observation
    api = binding['api_pid']
    if (Path('/proc') / str(api)).exists() and Observation.pid_start(api) == binding['api_starttime']:
        raise ValueError('same API PID/starttime survived owned arm stop')
    listener = native.output('ss', '-H', '-ltnp', 'sport = :8201')
    if listener:
        raise ValueError('8201 listener survived arm stop')
    proof.update(api_pid=api, api_starttime=binding['api_starttime'],
                 post_stop_8201_listener=listener)
    save(native.A / arm / 'worker-stop.json', proof)
    return proof


def cleanup(label, *, drain_gpu):
    """Best-effort all-unit stop; failures retained, never block restoration."""
    if label not in ('exit-final', 'outer-before-restore'):
        raise ValueError('finite unique cleanup label required')
    if drain_gpu is not (label == 'exit-final'):
        raise ValueError('outer recovery cannot drain after original was restarted')
    failures, rows = [], []
    for unit in UNITS.values():
        path = native.A / (label + '-' + unit)
        row = {'unit': unit}
        try:
            row['before'] = dict(line.split('=', 1) for line in native.output(
                'systemctl', 'show', unit, '-pLoadState', '-pActiveState',
                '-pMainPID', '-pControlGroup').splitlines())
            if row['before']['LoadState'] == 'not-found':
                row['stop'] = 'NOT_LOADED_TYPED_SKIP_NO_SYSTEMCTL_STOP'
            else:
                try:
                    command(['sudo', '-n', 'systemctl', 'stop', unit],
                            str(path) + '.stop.log', 50)
                except BaseException as error:
                    failures.append({'unit': unit, 'phase': 'stop', 'error': repr(error)[:2048]})
                row['after'] = dict(line.split('=', 1) for line in native.output(
                    'systemctl', 'show', unit, '-pLoadState', '-pActiveState',
                    '-pMainPID', '-pControlGroup').splitlines())
                if row['after']['MainPID'] != '0' or row['after']['ActiveState'] not in ('inactive', 'failed'):
                    failures.append({'unit': unit, 'phase': 'ended', 'raw': row['after']})
            # Preserve service journal outside timed groups even after unitGC.
            command(['sudo', '-n', 'journalctl', '-u', unit,
                     '--since=' + (native.A / 'controller-started.txt').read_text().strip(),
                     '--no-pager'], str(path) + '.journal.log', 20)
        except BaseException as error:
            failures.append({'unit': unit, 'phase': 'identity/journal', 'error': repr(error)[:2048]})
        rows.append(row)
    if drain_gpu:
        try:
            native.drain(label)
        except BaseException as error:
            failures.append({'phase': 'GPU_drain', 'error': repr(error)[:2048]})
    save(native.A / (label + '-cleanup.json'), {
        'status': 'PASS_OWNED_CLEANUP' if not failures else 'FAIL_OWNED_CLEANUP',
        'rows': rows, 'errors': failures, 'original_restore_must_continue': True,
        'GPU_drain_requested': drain_gpu})
    return 0 if not failures else 1
