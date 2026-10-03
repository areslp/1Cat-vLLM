"""Actual pure-service startup collector; invoked only by a future owned window."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import urllib.request

from io_tools import save, sha
from original_guard import Observation, output, metric_rows
from service_binding import capture_contract, pure_config, pinned_inputs
from common import MODELS
from guard_activation import proof, require_binding

PY = '/home/l/work/1Cat-vLLM/.venv/bin/python'
UUIDS = ['GPU-54a5dec0-85a9-837b-c54d-4a3752c1b620',
         'GPU-94e7fdea-085e-eea4-86a5-83bbd6ad0883',
         'GPU-7aaea8dd-2241-2730-1234-96a9cd033e11',
         'GPU-72ca49d9-70ae-f881-70b3-a16295209e47']
SELECTED = ('MODEL', 'CODE', 'FAV100_SHIM', 'ONECAT_FAV100_SO',
    'STEP58_SERVICE_CONFIG', 'STEP58_ORIGINAL_SHIM', 'STEP58_RAW_GENERATION_PATH',
    'VLLM_LOGGING_CONFIG_PATH', 'ONECAT_E7_STATS_DIR', 'EXTRA_ARGS',
    'CUDA_VISIBLE_DEVICES', 'PYTHONPATH')
RAW_FLAGS = ('ONECAT_QSA48', 'VLLM_SM70_QSA_GROUPED_PAD_FIX',
    'ONECAT_FUSE47', 'ONECAT_DRAFT47', 'ONECAT_GDN_FUSE',
    'ONECAT_GDN_BV48', 'ONECAT_MTP_GDN_GROUP_META',
    'ONECAT_MTP_SHORTCONV_GROUP_META', 'ONECAT_E7_MODE')
E7_FIELDS = ('pid', 'rank', 'ranks', 'mode', 'module_path', 'package_hashes')


def read(path):
    path = Path(path)
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError('bounded startup input')
    return json.loads(path.read_text())


def environment(pid):
    return dict(row.decode().split('=', 1) for row in
        (Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0') if b'=' in row)


def validate_environment(actual, expected, service_path):
    if (not expected or any(key not in expected for key in SELECTED + RAW_FLAGS)
            or any(actual.get(key) != value for key, value in expected.items())
            or actual.get('STEP58_SERVICE_CONFIG') != str(service_path)
            or actual.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3'
            or any(key.startswith(('STEP58_NUMERIC_', 'STEP58_ADMISSION_',
                                  'PHYSICAL_FIXTURE_')) for key in actual)):
        raise ValueError('actual pure API environment differs or diagnostic activation leaked')
    if any(key not in actual for key in SELECTED):
        raise ValueError('required original API/config/shim/route identity absent')
    return {key: actual[key] for key in SELECTED}


def publish_binding(study, preliminary, receipt_path):
    """Close the actual file-only consumer contract before publishing its input."""
    binding = dict(preliminary)
    binding.update(startup_path=str(receipt_path), startup_sha256=sha(receipt_path))
    observation = Observation(binding)
    resource_guard = require_binding(binding)
    save(Path(study) / 'binding-constructor-gate.json', {
        'status': 'PASS_ORIGINAL_OBSERVATION_FILE_CONSTRUCTOR_NOT_LIVE',
        'core_binding_keys': sorted(binding),
        'startup_path': binding['startup_path'],
        'startup_sha256': binding['startup_sha256'],
        'source_files': len(observation.descriptors),
        'capture_receipts': len(binding['capture_receipts']),
        'resource_guard_identity': resource_guard,
        'identity_HTTP_GPU_called': False}, 2 * 1024**2)
    save(Path(study) / 'guard-binding.json', binding, 2 * 1024**2)
    return binding


def collect(arm, unit, config_path, expected_environment, study):
    if arm not in MODELS or unit != MODELS[arm]:
        raise ValueError('exact current W2 arm/unit identity')
    resource_guard = proof()
    pinned = pinned_inputs()
    config_path, study = Path(config_path), Path(study)
    cfg = pure_config(read(config_path))
    if cfg['mode'] != ('on' if arm == 'B' else 'off'):
        raise ValueError('pure arm mode differs')
    ready = read(study / 'readiness.json')
    invocation = read(study / 'startup-invocation.json')
    if (ready['status'] != 'READY_MATURE' or ready['mature_seconds'] < 360
            or ready['elapsed_s'] > 1080 or invocation['unit'] != unit
            or invocation['config_path'] != str(config_path)
            or invocation['config_sha256'] != sha(config_path)):
        raise ValueError('real bounded mature/config/invocation evidence missing')
    pid = int(output(['systemctl', 'show', unit, '-pMainPID', '--value']))
    if pid <= 0:
        raise ValueError('actual API process absent')
    group = '/system.slice/' + unit
    cpu_set = (Path('/sys/fs/cgroup') / group.lstrip('/') /
               'cpuset.cpus.effective').read_text().strip()
    if cpu_set != '0-13,15-41,43-55':
        raise ValueError('actual model cpuset must exclude client CPUs14/42')
    actual_env = environment(pid)
    selected = validate_environment(actual_env, read(expected_environment), config_path)
    directory = Path(cfg['output_dir'])
    if (Path(cfg['arm_file']).exists() or Path(cfg['drain_file']).exists()
            or list(directory.glob('*error-pid*.json'))
            or list(config_path.parent.glob('*error-pid*.json'))):
        raise ValueError('diagnostic arm/drain or hook failure artifact exists')
    cfg_sha = sha(config_path)
    captures, capture_pins, workers, e7_identity, raw_worker_flags = [], [], [], [], []
    for rank in range(4):
        path = directory / f'capture-ready-rank{rank}.json'
        row = read(path)
        capture_contract(row, cfg, rank, row['pid'], config_path, cfg_sha)
        if row['runtime']['device_uuid'].removeprefix('GPU-') != UUIDS[rank].removeprefix('GPU-'):
            raise ValueError('rank→Torch/raw physical UUID differs')
        wp = row['pid']
        proc = Path('/proc') / str(wp)
        if (proc.joinpath('cgroup').read_text().strip() != '0::' + group
                or os.path.realpath(proc / 'exe') != os.path.realpath(PY)):
            raise ValueError('worker exe/cgroup outside owned runtime')
        mapped = proc.joinpath('maps').read_text()
        for name in ('candidate_binary', 'original_binary'):
            path_binary = Path(cfg[name]).resolve()
            if str(path_binary) not in mapped:
                raise ValueError('exact source-pinned planner/forward library is not mapped')
        worker = {'rank': rank, 'pid': wp,
            'starttime_ticks': Observation.pid_start(wp), 'physical_uuid': UUIDS[rank]}
        workers.append(worker)
        worker_env = environment(wp)
        raw_flags = {key: worker_env.get(key) for key in RAW_FLAGS}
        raw_worker_flags.append({'rank': rank, 'pid': wp, 'observed': raw_flags,
            'matches_API_environment': all(raw_flags[key] == actual_env[key]
                                          for key in RAW_FLAGS)})
        capture_pins.append({'rank': rank, 'path': str(path), 'sha256': sha(path)})
        captures.append(row)
        e7 = read(Path(actual_env['ONECAT_E7_STATS_DIR']) / f'e7-{wp}.json')
        if e7['pid'] != wp or e7['rank'] != rank or e7['mode'] != 'on':
            raise ValueError('existing original E7 identity differs')
        e7_identity.append({key: e7[key] for key in E7_FIELDS})
    if len({w['pid'] for w in workers}) != 4:
        raise ValueError('actual four worker PID bijection absent')
    # Includes complete service source package and config-declared native code,
    # plus exact original and candidate binary. Hash outside timed groups.
    source_files = dict(pinned)
    source_files.update(resource_guard['source_pins'])
    for value in cfg['source_pins'].values():
        if sha(value['path']) != value['sha256']:
            raise ValueError('actual native source pin changed')
        source_files[value['path']] = value['sha256']
    for name in ('candidate_binary', 'original_binary', 'original_shim'):
        expected = cfg[name + '_sha256']
        if sha(cfg[name]) != expected:
            raise ValueError('actual library/original shim changed')
        source_files[cfg[name]] = expected
    descriptors = {path: Observation.file_identity(path) for path in source_files}
    preliminary = {'unit': unit, 'invocation_id': invocation['invocation_id'],
        'api_pid': pid, 'api_starttime': Observation.pid_start(pid), 'workers': workers,
        'selected_API_environment': selected, 'telemetry_dir': actual_env['ONECAT_E7_STATS_DIR'],
        'source_files': source_files, 'capture_receipts': capture_pins,
        'E7_identity': e7_identity}
    actual = Observation.actual_identity(preliminary, descriptors)
    for endpoint in ('health', 'v1/models', 'metrics'):
        with urllib.request.urlopen('http://127.0.0.1:8201/' + endpoint, timeout=5) as response:
            if response.status != 200:
                raise ValueError('startup actual health/model/metrics failed')
            raw = response.read(2 * 1024**2 + 1)
        if len(raw) > 2 * 1024**2:
            raise ValueError('bounded startup endpoint bytes')
        if endpoint == 'v1/models' and [row['id'] for row in json.loads(raw)['data']] != ['flash-next']:
            raise ValueError('actual served model identity differs')
        if endpoint == 'metrics':
            rows = metric_rows(raw.decode())
            if any(rows[(name, None)] for name in ('vllm:num_requests_running', 'vllm:num_requests_waiting')):
                raise ValueError('startup queue not exactly drained')
            with (study / 'startup-metrics.txt').open('xb') as stream:
                stream.write(raw)
    # Current Invocation only; old service boot KV lines cannot establish this.
    journal = output(['journalctl', '_SYSTEMD_INVOCATION_ID=' + invocation['invocation_id'],
                       '-u', unit, '--no-pager', '-o', 'cat'])
    if len(journal.encode()) > 8 * 1024**2:
        raise ValueError('startup journal bound')
    capacities = re.findall(r'GPU KV cache size:\s*([\d,]+) tokens', journal)
    if not capacities or any(int(v.replace(',', '')) != 663816 for v in capacities):
        raise ValueError('current owned startup capacity changed')
    with (study / 'startup-journal.txt').open('x') as stream:
        stream.write(journal)
    receipt_path = study / 'startup-gate.json'
    value = {'status': 'PASS_W2_PURE_SERVICE_STARTUP_NOT_PERFORMANCE',
        'arm': arm, 'mode': cfg['mode'], 'unit': unit,
        'service_config_path': str(config_path), 'service_config_sha256': cfg_sha,
        'service_capture_dir': str(directory), 'capture_receipts': capture_pins,
        'worker_pids': {str(w['rank']): w['pid'] for w in workers},
        'identity': actual, 'api_pid': pid, 'invocation_id': invocation['invocation_id'],
        'resource_guard_identity': resource_guard,
        'maturity': ready, 'kv_tokens': 663816,
        'original_child_raw_flags_boolean': all(row['matches_API_environment']
                                               for row in raw_worker_flags),
        'raw_worker_flag_observations': raw_worker_flags,
        'flag_limitation': 'Raw /proc boolean claims not used as runtime flag proof; config/loaded source/native bindings are explicit.',
        'private_GPU_bytes': [row['private_bytes'] for row in captures],
        'device_counter_shape': [row['counter_shape'] for row in captures],
        'performance_no_numeric_admission_management_host_audit_shadow': True,
        'startup_checks_are_outside_timed_request_groups': True,
        'cpuset_cpus_effective': cpu_set,
        'runtime': {'executable': sys.executable, 'python': sys.version}}
    save(receipt_path, value, 2 * 1024**2)
    publish_binding(study, preliminary, receipt_path)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('A0', 'B', 'A2'), required=True)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--expected-api-environment', required=True)
    parser.add_argument('--study', required=True)
    args = parser.parse_args()
    collect(args.arm, args.unit, args.config, args.expected_api_environment, args.study)


if __name__ == '__main__':
    main()
