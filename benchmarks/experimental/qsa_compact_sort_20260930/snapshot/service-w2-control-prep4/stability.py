"""Frozen pool closed-loop load. Process preparation counts in real duration."""
import argparse
import copy
import json
from pathlib import Path
import time

from frozen import require
from io_tools import save, aggregate_gate
from original_guard import Observation, OriginalGuard
from request_groups import result
from transport_deadline import Pump
from artifact_budget import BudgetMonitor, free_space_preflight

REQUIRED = ['requests_ok', 'input_tokens', 'output_tokens', 'terminal_states',
            'queue_drained', 'no_restart', 'original_route_counters_valid']


def should_stop(first_submission, now, completed):
    return (first_submission is not None and now - first_submission >= 600
            and completed >= 320)


def packets():
    value = json.loads(require('stability-pool.frozen.json').read_text())
    if len(value['warmup']) != 8 or len(value['pool']) != 2048:
        raise ValueError('exact existing pool and warmup required')
    def adapt(row):
        # Transport metadata only. Every frozen body/header/identity remains.
        row = copy.deepcopy(row)
        row.update(cancel_contract=None, expected_terminal='COMPLETE')
        return row
    return [adapt(row) for row in value['warmup']], [adapt(row) for row in value['pool']]


def group_contract(used):
    return {'requests': used, 'terminal_states': ['COMPLETE'] * len(used),
            'required_checks': REQUIRED}


def prepare(pump, packet, root, number, deadline):
    path = root / f'{number:04d}.packet.json'
    save(path, packet)
    job = pump.add(path, root / f'request-{number:04d}')
    pump.wait_ready([job], min(deadline, time.perf_counter() + 30))
    return job


def run(output, window, binding):
    warmups, pool = packets()
    root = Path(output)
    root.mkdir(mode=0o700)
    save(root / 'disk-preflight.json', free_space_preflight(window))
    monitor = BudgetMonitor(window, root / 'artifact-budget-samples.jsonl')
    guard = OriginalGuard(Observation(binding), root / 'guard')
    warm = root / 'warmup'
    warm.mkdir(mode=0o700)
    pump = Pump(budget_check=monitor.tick)
    rows = []
    phase = 'warmup'
    try:
        # Warmups are retained/count separately and outside measured 900s.
        warm_deadline = time.perf_counter() + 180
        jobs = [prepare(pump, packet, warm, n, warm_deadline)
                for n, packet in enumerate(warmups)]
        before = guard.before(group_contract(warmups), warm_deadline)
        for job in jobs:
            job.release(warm_deadline)
        pump.wait_finished(jobs, warm_deadline + 4)
        warm_rows = [result(job, packet) for job, packet in zip(jobs, warmups)]
        guard.after(group_contract(warmups), before, warm_rows, warm_deadline)
        save(warm / 'RESULT.json', {'status': 'COMPLETE_FROZEN_STABILITY_WARMUP',
                                   'requests': warm_rows}, 2 * 1024**2)
        pump.close()
        pump = Pump(budget_check=monitor.tick)
        phase = 'measured'
        measured = root / 'measured'
        measured.mkdir(mode=0o700)
        prep_deadline = time.perf_counter() + 30
        initial = [prepare(pump, pool[n], measured, n, prep_deadline) for n in range(8)]
        before = guard.before(group_contract([]), prep_deadline + 30)
        measured_begin = time.perf_counter()
        deadline = measured_begin + 900
        for job in initial:
            job.release(deadline)
        active = {job: n for n, job in enumerate(initial)}
        used, completed, next_number = pool[:8], {}, 8
        first_submission = None
        stop_submitting = False
        while active:
            if time.perf_counter() >= deadline:
                raise TimeoutError('frozen 900s measured preparation/load/drain budget')
            pump.tick()
            finished = [(job, n) for job, n in active.items() if job.reaped]
            for job, n in finished:
                row = result(job, pool[n])
                completed[n] = row
                rows.append(row)
                del active[job]
                save(measured / f'completed-{n:04d}.json', {
                    'ordinal': n, 'result_path': str(job.output / 'result.json'),
                    'actual_child_exit': job.process.returncode,
                    'ready_parent_observed_monotonic': job.ready_observed,
                    'release_monotonic': job.started,
                    'reaped_monotonic': job.finished})
            if all(n in completed for n in range(8)) and first_submission is None:
                first_submission = min(completed[n]['started_monotonic'] for n in range(8))
            if first_submission is not None:
                stop_submitting = should_stop(first_submission,
                                               time.perf_counter(), len(completed))
            aggregate_gate(window)
            # Fill every vacancy immediately. Each replacement's native import
            # and READY delay is part of the live load duration, not hidden.
            while not stop_submitting and len(active) < 8 and next_number < len(pool):
                job = prepare(pump, pool[next_number], measured, next_number, deadline)
                if should_stop(first_submission, time.perf_counter(), len(completed)):
                    stop_submitting = True
                    save(measured / f'not-submitted-{next_number:04d}.json', {
                        'status': 'PREPARED_READY_STOP_CONDITION_BEFORE_START_NO_HTTP',
                        'ordinal': next_number, 'pid': job.pid,
                        'ready': job.ready, 'release_monotonic': job.started})
                    job.terminate('prepared at frozen stop boundary; START never sent')
                    break
                job.release(deadline)
                active[job] = next_number
                used.append(pool[next_number])
                next_number += 1
                # Do not miss the frozen stop condition while preparing a child.
                if first_submission is not None:
                    stop_submitting = should_stop(first_submission,
                                                   time.perf_counter(), len(completed))
            if not active and not stop_submitting:
                raise RuntimeError('INSUFFICIENT_STABILITY_POOL_EXHAUSTED_NO_APPEND_NO_IDLE_PADDING')
        # This also reaps a final prepared-but-never-submitted child, if any,
        # before native model queue/metrics evidence is read.
        pump.close()
        last_finish = max(row['finished_monotonic'] for row in rows)
        if (first_submission is None or len(rows) < 320
                or last_finish - first_submission < 600 or not stop_submitting):
            raise ValueError('actual first-to-last duration or completion minimum absent')
        ordered = [completed[n] for n in range(next_number)]
        after, checks = guard.after(group_contract(used), before, ordered, deadline)
        value = {'status': 'COMPLETE_FROZEN_STABILITY_NOT_PERFORMANCE_ADMISSION',
            'warmup_requests': 8, 'submitted': next_number, 'completed': len(rows),
            'pool_cap': 2048, 'concurrency_cap': 8,
            'first_submission_monotonic': first_submission,
            'last_request_finished_monotonic': last_finish,
            'actual_first_to_last_s': last_finish - first_submission,
            'elapsed_including_replacement_process_preparation_s': time.perf_counter() - measured_begin,
            'checks': checks, 'after_original_counters': after,
            'aggregate_measured_bytes': aggregate_gate(window)}
        save(root / 'RESULT.json', value, 2 * 1024**2)
        return value
    except BaseException as error:
        save(root / 'FAILURE.json', {'status': 'STABILITY_STOP_NO_RETRY_NO_APPEND',
            'phase': phase, 'error': repr(error)[:4096], 'completed_measured': len(rows)})
        raise
    finally:
        try:
            pump.close()
        except BaseException as error:
            save(root / 'CLEANUP-FAILURE.json', {'status': 'OWNED_HTTP_PROCESS_CLEANUP_FAILED',
                                               'error': repr(error)[:4096]})
            raise
        finally:
            monitor.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--window', required=True)
    args = parser.parse_args()
    path = Path(args.binding)
    if path.stat().st_size > 1024**2:
        raise ValueError('startup binding size')
    run(args.output, args.window, json.loads(path.read_text()))


if __name__ == '__main__':
    main()
