"""One finite homogeneous-context HTTP arm; service ownership stays outside."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dependencies import activate, sha
activate()
from async_budget import AsyncBudgetMonitor
from artifact_budget import free_space_preflight
from io_tools import save
from original_guard import Observation, OriginalGuard
from request_groups import result
from transport_deadline_context import Pump
from context_metrics import measure
from prime_guard import after_prime, PRIME_CHECKS

HERE = Path(__file__).resolve().parent


def inventory(matrix, arm):
    if (matrix['schema'] != 'HOMOGENEOUS_CONTEXT_QSA_MATRIX_V1'
            or matrix['arms'] != ['A0', 'B', 'A2'] or arm not in matrix['arms']
            or matrix['HTTP_requests'] != 798 or matrix['max_output_tokens'] != 112488
            or matrix['repeats'] != 2 or matrix['request_s'] != 600
            or matrix['group_s'] != 660 or matrix['arm_s'] != 7200):
        raise ValueError('finite context inventory/budgets differ')
    groups = [g for g in matrix['groups'] if g['arm'] == arm]
    if (len(groups) != 38 or sum(g['requests'] for g in groups) != 146
            or sum(g['primes'] for g in groups) != 120):
        raise ValueError('complete arm assignment required')
    return groups


def load_group(pin):
    path = HERE / pin['path']
    if (path.is_symlink() or not path.is_relative_to(HERE / 'groups')
            or path.stat().st_size != pin['bytes'] or sha(path) != pin['sha256']):
        raise ValueError('frozen group path/content differs')
    group = json.loads(path.read_text())
    if (group['arm'] != pin['arm'] or group['row_id'] != pin['row_id']
            or group['ordinal'] != pin['ordinal'] or group['concurrency'] != pin['requests']
            or len(group['primes']) != pin['primes']):
        raise ValueError('group/arm/ordinal assignment differs')
    return group


def run_request(packet, root, pump, deadline):
    save(root.with_suffix('.packet.json'), packet, 4 * 1024**2)
    job = pump.add(root.with_suffix('.packet.json'), root)
    pump.wait_ready([job], min(deadline, time.perf_counter() + 30))
    job.release(deadline)
    pump.wait_finished([job], min(deadline + 4, job.deadline + 4))
    return result(job, packet)


def strict_checks(checks, required):
    if set(checks) != set(required) or any(value is not True for value in checks.values()):
        raise ValueError('missing/false group check; no empty-dict success')


def run_group(group, root, guard, monitor, arm_deadline):
    root.mkdir(mode=0o700)
    save(root / 'frozen-group.json', group, 8 * 1024**2)
    pump = Pump(budget_check=monitor.tick)
    primary_error = None
    primes = []
    try:
        monitor.tick(force=True)
        prime_block_deadline = min(arm_deadline, time.perf_counter() +
                                   len(group['primes']) * 630)
        for index, packet in enumerate(group['primes']):
            # Priming is untimed relative to measured decode, but its cold TTFT
            # and native counters/terminal/queue evidence are all retained.
            cold = {**group, 'requests': [packet], 'terminal_states': ['COMPLETE'],
                    'row_id': f'{group["row_id"]}-repeat{group["ordinal"]}-prime{index}',
                    'required_checks': PRIME_CHECKS}
            before = guard.before(cold, prime_block_deadline)
            row = run_request(packet, root / f'prime-{index:02d}', pump,
                              prime_block_deadline)
            after, checks = after_prime(guard, cold, before, row, prime_block_deadline)
            strict_checks(checks, cold['required_checks'])
            primes.append({'index': index, 'request_result': row, 'checks': checks,
                           'before': before, 'after': after,
                           'interpretation': 'unique-salt cold sequential c1 TTFT; not concurrent TTFT'})
            save(root / f'prime-{index:02d}.json', primes[-1], 4 * 1024**2)
            monitor.tick(force=True)
        measured_started = time.perf_counter()
        deadline = min(arm_deadline, measured_started + 660)
        jobs = []
        for index, packet in enumerate(group['requests']):
            path = root / f'request-{index:02d}.packet.json'
            save(path, packet, 4 * 1024**2)
            jobs.append(pump.add(path, root / f'request-{index:02d}'))
        pump.wait_ready(jobs, min(deadline, time.perf_counter() + 30))
        before = guard.before(group, deadline)
        monitor.tick(force=True)
        for job in jobs:
            job.release(deadline)
        pump.wait_finished(jobs, min(arm_deadline + 4, deadline + 4))
        rows = [result(job, packet) for job, packet in zip(jobs, group['requests'])]
        after, checks = guard.after(group, before, rows, deadline)
        strict_checks(checks, group['required_checks'])
        observation = measure(rows)
        value = {'status': 'COMPLETE_HTTP_GROUP_NOT_PERFORMANCE_ADMISSION',
            'arm': group['arm'], 'row_id': group['row_id'], 'ordinal': group['ordinal'],
            'input_tokens': group['input_tokens'], 'concurrency': group['concurrency'],
            'context_state': group['context_state'], 'checks': checks,
            'candidate_route_expected': group['candidate_route_expected'],
            'candidate_activation_evidence': 'static startup graph binding; actual full consumer dispatch not instrumented',
            'dynamic_candidate_activation': group['dynamic_candidate_activation'],
            'request_results': rows, 'priming_results': primes,
            'metrics': observation, 'before': before, 'after': after,
            'release_clocks': [{'index': i, 'pid': j.pid, 'READY_parent': j.ready_observed,
                               'release_parent': j.started,
                               'HTTP_started_child': rows[i]['started_monotonic'],
                               'finished_child': rows[i]['finished_monotonic'],
                               'reaped_parent': j.finished} for i, j in enumerate(jobs)],
            'measured_group_elapsed_s': time.perf_counter() - measured_started,
            'warm_scope': 'aggregate native cache hits do not prove every request warm; actual common client window remains separate',
            'actual_prefix_deltas': {r['name']: r['value'] for r in after['metric_deltas']
                if r['name'] in ('vllm:prefix_cache_queries_total', 'vllm:prefix_cache_hits_total')},
            'server_extra_counters': 'raw guard metrics retained; missing first HTTP/ITL family unavailable, no fabricated zero'}
        if value['measured_group_elapsed_s'] > 660:
            raise TimeoutError('measured group finite bound exceeded')
        save(root / 'GROUP.json', value, 8 * 1024**2)
        monitor.tick(force=True)
        return value
    except BaseException as error:
        primary_error = error
        save(root / 'FAILURE.json', {'status': 'FAILED_ASSIGNED_GROUP_NO_REPLACEMENT',
             'arm': group['arm'], 'row_id': group['row_id'], 'ordinal': group['ordinal'],
             'completed_primes': len(primes), 'error': repr(error),
             'traceback': traceback.format_exc()[-16384:]})
        raise
    finally:
        try:
            pump.close()
        except BaseException as cleanup:
            save(root / 'CLEANUP-FAILURE.json', {'primary_error': repr(primary_error),
                 'cleanup_error': repr(cleanup), 'traceback': traceback.format_exc()[-16384:]})
            raise


def run(arm, binding, root, matrix_path):
    manifest = json.loads((HERE / 'MANIFEST.json').read_text())
    for row in manifest['files']:
        path = HERE / row['path']
        if path.is_symlink() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('frozen source/input manifest differs')
    if matrix_path.resolve() != (HERE / 'matrix.frozen.json').resolve():
        raise ValueError('only frozen assigned matrix accepted')
    matrix = json.loads(matrix_path.read_text())
    groups = inventory(matrix, arm)
    root.mkdir(mode=0o700)
    save(root / 'INVENTORY.json', {'arm': arm, 'groups': groups, 'matrix_sha256': sha(matrix_path)})
    # Window includes sibling model/restoration outputs; scanner is outside
    # request loop and ordinary tick only drains its bounded small receipts.
    window = root.parents[3]
    save(root / 'disk-preflight.json', free_space_preflight(window))
    guard = OriginalGuard(Observation(binding), root / 'guard')
    monitor = AsyncBudgetMonitor(window, root / 'budget-trace.jsonl', cpu=42, stale_s=5)
    began = time.perf_counter()
    records = []
    try:
        monitor.tick(force=True)
        for index, pin in enumerate(groups):
            group = load_group(pin)
            destination = root / f'{index:03d}-{pin["row_id"]}-repeat{pin["ordinal"]}'
            row = run_group(group, destination, guard, monitor, began + 7200)
            record = {'row_id': row['row_id'], 'ordinal': row['ordinal'],
                      'status': row['status'], 'path': str(destination / 'GROUP.json'),
                      'sha256': sha(destination / 'GROUP.json')}
            records.append(record)
            save(root / f'group-complete-{index:03d}.json', record)
        value = {'status': 'COMPLETE_HTTP_MATRIX_NOT_PERFORMANCE_ADMISSION',
                 'arm': arm, 'completed_groups': 38, 'measured_requests': 146,
                 'prime_requests': 120, 'HTTP_requests': 266, 'outputs': 37496,
                 'group_records': records, 'elapsed_s': time.perf_counter() - began,
                 'old_run4_status': 'UNCHANGED_NO_GO', 'n_per_row': 2}
        save(root / 'RESULT.json', value)
        return value
    except BaseException as error:
        save(root / 'FAILURE.json', {'status': 'INCOMPLETE_ARM_STOP_NO_RETRY',
             'arm': arm, 'completed_groups': len(records), 'assigned_groups': 38,
             'error': repr(error), 'traceback': traceback.format_exc()[-16384:],
             'elapsed_s': time.perf_counter() - began})
        raise
    finally:
        monitor.close()


def main():
    os.umask(0o077)
    args = argparse.ArgumentParser()
    args.add_argument('--arm', choices=('A0', 'B', 'A2'), required=True)
    args.add_argument('--binding', required=True)
    args.add_argument('--output', required=True)
    args.add_argument('--matrix', required=True)
    opt = args.parse_args()
    run(opt.arm, json.loads(Path(opt.binding).read_text()), Path(opt.output), Path(opt.matrix))


if __name__ == '__main__':
    main()
