"""Thin full-arm envelope; frozen matrix/group/transport/metrics stay unchanged."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

PACKAGE = Path(__file__).resolve().parent
MATRIX = PACKAGE.parent / 'context-qsa-prep1'
sys.path.insert(0, str(PACKAGE))
from guard_activation import activate, proof, require_binding
activate()
from transport_activation import owned, bind, require_runner
owned()  # Select exact relative WORKER before generic frozen imports.
from nvml_capture import install_guard
spec = importlib.util.spec_from_file_location('context2_frozen_runner', MATRIX / 'runner.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
activate()  # Frozen dependency activation may reorder sys.path; module identity is checked again.
bind(original)  # run_group source/arguments unchanged; only its global Pump selector is explicit.
from io_tools import save, sha
CLIENT_CGROUP_BYTES = 1024**3  # New run4 contract, not a retrospective old limit.


def assigned(matrix_path, arm):
    if Path(matrix_path).resolve() != MATRIX / 'matrix.frozen.json':
        raise ValueError('only original frozen matrix path allowed')
    for row in json.loads((MATRIX / 'MANIFEST.json').read_text())['files']:
        path = MATRIX / row['path']
        if path.is_symlink() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('frozen source/input manifest differs')
    return original.inventory(json.loads(Path(matrix_path).read_text()), arm)


def client_checks(files):
    events = dict(line.split() for line in files['memory.events'].splitlines())
    # memory.stat is preserved as the actual kernel file, not inferred RSS.
    stat = dict(line.split() for line in files['memory.stat'].splitlines())
    if not stat or any(int(value) < 0 for value in stat.values()):
        raise ValueError('actual client memory.stat missing/invalid')
    return {
        'memory_max_1GiB': files['memory.max'] == str(CLIENT_CGROUP_BYTES),
        'current_peak_within_1GiB': all(0 <= int(files[name]) <= CLIENT_CGROUP_BYTES
            for name in ('memory.current', 'memory.peak')),
        'events_max_zero': events.get('max') == '0',
        'swap_zero': files['memory.swap.max'] == files['memory.swap.current'] == '0',
        'oom_zero': all(events.get(name) == '0' for name in ('oom', 'oom_kill')),
        'client_cpu14_42': files['cpuset.cpus.effective'] == '14,42',
    }


def client_resource(root, arm):
    raw = Path('/proc/self/cgroup').read_text()
    lines = raw.strip().splitlines()
    if arm not in ('A0', 'B', 'A2'):
        raise ValueError('exact assigned HTTP arm required')
    expected = '/system.slice/step58-contextqsa4-http-' + arm.lower() + '.service'
    if lines != ['0::' + expected]:
        raise ValueError('diagnostic client actual cgroup differs')
    cg = Path('/sys/fs/cgroup') / expected.lstrip('/')
    names = ('memory.max', 'memory.current', 'memory.peak', 'memory.events', 'memory.stat',
             'memory.swap.max', 'memory.swap.current', 'cpuset.cpus.effective')
    files = {name: (cg / name).read_text().strip() for name in names}
    checks = client_checks(files)
    value = {'status': 'PASS_ACTUAL_CLIENT_CGROUP' if all(checks.values()) else 'FAIL',
             'epoch': time.time(), 'proc_self_cgroup_raw': raw, 'cgroup_path': str(cg),
             'files': files, 'checks': checks, 'scope': 'actual client unit after child/scanner cleanup attempts',
             'systemd_run_CLI_peak_used_as_actual_peak': False}
    save(root / 'client-resource.json', value)
    if value['status'] != 'PASS_ACTUAL_CLIENT_CGROUP':
        raise ValueError('actual client cgroup resource checks failed')
    return value


def validate_client_resource(root, arm):
    value = json.loads((Path(root) / 'client-resource.json').read_text())
    expected = '/system.slice/step58-contextqsa4-http-' + arm.lower() + '.service'
    required = {'memory_max_1GiB', 'current_peak_within_1GiB', 'events_max_zero',
                'swap_zero', 'oom_zero', 'client_cpu14_42'}
    if (arm not in ('A0', 'B', 'A2')
            or value['status'] != 'PASS_ACTUAL_CLIENT_CGROUP'
            or value['proc_self_cgroup_raw'].strip() != '0::' + expected
            or value['cgroup_path'] != '/sys/fs/cgroup' + expected
            or set(value['checks']) != required
            or any(v is not True for v in value['checks'].values())
            or value['checks'] != client_checks(value['files'])
            or value['systemd_run_CLI_peak_used_as_actual_peak'] is not False):
        raise ValueError('actual final client cgroup readback absent/failed')
    return value


def run(arm, binding, root, matrix_path):
    root.mkdir(mode=0o700)
    began, records, primary, monitor = time.perf_counter(), [], None, None
    try:
        groups = assigned(matrix_path, arm)
        resource_guard = require_binding(binding)
        transport = require_runner(original)
        if resource_guard['transport_identity'] != transport:
            raise ValueError('startup and HTTP actual transport selection differ')
        save(root / 'INVENTORY.json', {'arm': arm, 'groups': groups,
            'matrix_sha256': sha(matrix_path), 'resource_guard_identity': resource_guard})
        window = root.parents[3]
        save(root / 'disk-preflight.json', original.free_space_preflight(window))
        install_guard(root / 'nvml-raw')
        guard = original.OriginalGuard(original.Observation(binding), root / 'guard')
        monitor = original.AsyncBudgetMonitor(window, root / 'budget-trace.jsonl',
                                              cpu=42, stale_s=5)
        began = time.perf_counter()
        monitor.tick(force=True)
        for index, pin in enumerate(groups):
            group = original.load_group(pin)
            destination = root / f'{index:03d}-{pin["row_id"]}-repeat{pin["ordinal"]}'
            row = original.run_group(group, destination, guard, monitor, began + 7200)
            record = {'row_id': row['row_id'], 'ordinal': row['ordinal'],
                      'status': row['status'], 'path': str(destination / 'GROUP.json'),
                      'sha256': sha(destination / 'GROUP.json')}
            records.append(record)
            save(root / f'group-complete-{index:03d}.json', record)
        value = {'status': 'COMPLETE_HTTP_MATRIX_NOT_PERFORMANCE_ADMISSION',
                 'arm': arm, 'completed_groups': 38, 'measured_requests': 146,
                 'prime_requests': 120, 'HTTP_requests': 266, 'outputs': 37496,
                 'group_records': records, 'elapsed_s': time.perf_counter() - began,
                 'old_run4_status': 'UNCHANGED_NO_GO', 'n_per_row': 2,
                 'resource_guard_identity': resource_guard}
        save(root / 'RESULT.json', value)
        return value
    except BaseException as error:
        primary = error
        save(root / 'FAILURE.json', {'status': 'INCOMPLETE_ARM_STOP_NO_RETRY',
             'arm': arm, 'completed_groups': len(records), 'assigned_groups': 38,
             'error': repr(error), 'traceback': traceback.format_exc()[-16384:],
             'elapsed_s': time.perf_counter() - began})
        raise
    finally:
        cleanup = []
        try:
            if monitor is not None:
                monitor.close()
        except BaseException as error:
            cleanup.append(error)
        # All Pump children are reaped by the reused run_group finally path;
        # scanner close above precedes the real client cgroup readback.
        try:
            client_resource(root, arm)
        except BaseException as error:
            cleanup.append(error)
        if cleanup:
            try:
                save(root / 'client-resource-FAILURE.json', {
                    'primary_error': repr(primary),
                    'cleanup_errors': [repr(error) for error in cleanup],
                    'original_failure_must_remain_visible': True})
            except BaseException as error:
                cleanup.append(error)
            if primary is not None:
                raise BaseExceptionGroup('original HTTP arm failure and cleanup/readback',
                                         [primary, *cleanup])
            raise BaseExceptionGroup('HTTP arm cleanup/resource readback failed', cleanup)


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('A0', 'B', 'A2'), required=True)
    parser.add_argument('--binding', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--matrix', required=True)
    args = parser.parse_args()
    run(args.arm, json.loads(Path(args.binding).read_text()), Path(args.output), Path(args.matrix))
