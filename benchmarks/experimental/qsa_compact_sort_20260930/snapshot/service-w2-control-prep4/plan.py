"""One exact initial W2 plan: no confirmation, no diagnostic injection."""
from pathlib import Path

from common import BASE, PACKAGE, WINDOW, UNITS, OUTER, TIMER, read, sha
from execution_budget import BUDGETS, binding as budget_binding
from stage_bindings import validate_staged
from restoration_dependencies import validate as recovery_pins
import source_auth

RESOURCES = {'model_cgroup_bytes': 120 * 1024**3, 'outer_cgroup_bytes': 8 * 1024**3,
             'HTTP_cgroup_bytes': 2 * 1024**3, 'memory_swap_bytes': 0,
             'HTTP_CPUs': '14,42', 'HTTP_Nice': 15, 'HTTP_CPUQuota': '100%',
             'PID_NVML_MiB': 32000, 'device_NVML_MiB': 32384,
             'whole_window_measured_bytes': 8 * 1024**3,
             'inflight_request_tree_allowance_bytes': 8 * 34 * 1024**2,
             'free_disk_retention_reserve_bytes': 16 * 1024**3,
             'per_request_returned_line_bytes': 16 * 1024**2,
             'per_request_deadline_s': 120, 'term_to_kill_s': 1,
             'returned_line_cap_is_wire_cap': False,
             'aggregate_cap_is_filesystem_quota': False}


def mandatory():
    files = {**source_auth.authenticate(), **source_auth.controller_pins(),
             **source_auth.wire_proof(), **recovery_pins(PACKAGE)}
    files.update(validate_staged(PACKAGE, WINDOW))
    amendment = budget_binding()
    files[amendment['path']] = amendment['sha256']
    for name in ('before-snapshot.json', 'before-unit.txt', 'fresh-identity.json'):
        path = WINDOW / 'baseline' / name
        files[str(path)] = sha(path)
    return files


def validate(plan):
    from identity_contract import UUIDS
    if (plan['schema'] != 'step58-pure-w2-initial-v1'
            or plan['window'] != str(WINDOW) or plan['package'] != str(PACKAGE)
            or plan['arms'] != ['A0', 'B', 'A2'] or plan['units'] != UNITS
            or plan['outer'] != OUTER or plan['timer'] != TIMER
            or plan['device_uuids'] != list(UUIDS)
            or plan['budgets'] != BUDGETS or plan['resources'] != RESOURCES
            or plan['execution_budget_amendment'] != budget_binding()
            or plan['matrix_requests'] != 2010 or plan['maximum_HTTP_requests'] != 4066
            or plan['confirmation_authorized'] is not False
            or plan['new_observer_counter_profiler'] is not False
            or plan['engine_cancel_ack'] != 'UNVERIFIED'
            or plan['engine_cancelled_count'] is not None
            or type(plan['expected_pid']) is not int or plan['expected_pid'] <= 0):
        raise ValueError('initial frozen W2 plan/limits differ')
    expected = mandatory()
    if set(plan['files']) != set(expected) or any(plan['files'][p] != h for p, h in expected.items()):
        raise ValueError('complete source/prior/wire/staged/reference plan pins differ')
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError('actual approved plan bytes changed: ' + path)
    validate_staged(PACKAGE, WINDOW, plan['files'])
    before = read(WINDOW / 'baseline/before-snapshot.json', 4 * 1024**2)
    identity = read(WINDOW / 'baseline/fresh-identity.json')
    if (int(before['systemd']['MainPID']) != plan['expected_pid']
            or identity['api_pid'] != plan['expected_pid']
            or not all(before['checks'][k] for k in
                       ('health', 'source_config', 'kv_capacity', 'swap', 'cleanup', 'worker_identity'))):
        raise ValueError('same fresh original PID/six baseline gates required')
    return plan
