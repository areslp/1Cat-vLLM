"""Frozen group arrival rules with actual process termination and raw bindings."""
import json
from pathlib import Path
import statistics
import time

from frozen import module
from io_tools import (aggregate_gate, save, sha, REQUEST_TREE_CAP, tree_bytes)
from transport_deadline import Pump


def result(job, packet):
    if not job.reaped or job.process.returncode != 0 or job.abort_reason:
        raise ValueError('request has no successful reaped native process')
    path = job.output / 'result.json'
    if path.stat().st_size > 256 * 1024 or tree_bytes(job.output) > REQUEST_TREE_CAP:
        raise ValueError('request retained artifact cap')
    row = json.loads(path.read_text())
    identities = module('stream_ids')
    if (row['identity'] != identities.ids(packet['body'], packet['header']['X-Request-Id'])
            or row['status'] != packet['expected_terminal']
            or row['body_sha256'] != packet['body_sha256']
            or row['raw_response']['path'] != str(job.output / 'raw-lines.jsonl')
            or row['raw_response']['sha256'] != sha(job.output / 'raw-lines.jsonl')
            or not job.started <= row['started_monotonic'] <= row['finished_monotonic'] <= job.deadline):
        raise ValueError('raw/parser/process/request binding differs')
    close = json.loads((job.output / 'native-close.json').read_text())
    if close['torch_imported'] is not False or close['CVD'] != '':
        raise ValueError('HTTP child imported model/Torch or exposed GPU')
    # This is the actual parent's reaped boundary, not reconstructed service time.
    row['finished_epoch'] = time.time()
    row['process_exit_path'] = str(job.output / 'process-exit.json')
    row['process_exit_sha256'] = sha(job.output / 'process-exit.json')
    return row


def run_one(packet, root, deadline, pump, name):
    packet_path = root / (name + '.packet.json')
    save(packet_path, packet)
    job = pump.add(packet_path, root / name)
    pump.wait_ready([job], min(deadline, time.perf_counter() + 30))
    job.release(deadline)
    pump.wait_finished([job], min(deadline + 4, time.perf_counter() + 124))
    return result(job, packet)


def run_group(group, output, deadline, *, guard, window, budget_check=None):
    if guard is None:
        raise ValueError('missing real original guard')
    root = Path(output)
    root.mkdir(mode=0o700)
    save(root / 'frozen-group.json', group, 4 * 1024**2)
    pump, prime = Pump(budget_check=budget_check), None
    failure = None
    try:
        aggregate_gate(window)
        if group['prime']:
            guard.obs.identity()
            guard.obs.idle()
            prime = run_one(group['prime'], root, deadline, pump, 'prime')
        packets, jobs = group['requests'], []
        for index, packet in enumerate(packets):
            path = root / f'request-{index:02d}.packet.json'
            save(path, packet)
            jobs.append(pump.add(path, root / f'request-{index:02d}'))
        pump.wait_ready(jobs, min(deadline, time.perf_counter() + 30))
        # Every child's import/READY is outside the timed release boundary.
        before = guard.before(group, deadline)
        mixed = group['arrival_rule']['type'].startswith('four_short')
        first_count = 4 if mixed else len(jobs)
        for job in jobs[:first_count]:
            job.release(deadline)
        if mixed:
            trigger_deadline = min(deadline, time.perf_counter() + 120)
            while not any(job.first_positive for job in jobs[:4]):
                if time.perf_counter() >= trigger_deadline:
                    raise TimeoutError('frozen mixed first-positive trigger absent')
                pump.tick()
                if any(job.abort_reason for job in jobs[:4]):
                    raise RuntimeError('mixed trigger source request failed')
            for job in jobs[4:]:
                job.release(deadline)
        pump.wait_finished(jobs, deadline + 4)
        rows = [result(job, packet) for job, packet in zip(jobs, packets)]
        aggregate_gate(window)
        after, checks = guard.after(group, before, rows, deadline, prime=prime)
        metric = statistics.median(rows[index][group['metric']]
                                   for index in group['metric_member_indices'])
        module('group_analyzer').finite_positive(metric)
        value = {**{key: group[key] for key in ('row_id', 'arm', 'ordinal',
            'request_ids', 'body_sha256s', 'arrival_rule_sha256')},
            'status': 'COMPLETE', 'phase': group['phase'],
            'request_terminal_states': [row['status'] for row in rows],
            'checks': checks, 'group_metric': metric, 'request_results': rows,
            'prime_result': prime, 'before_original_counters': before,
            'after_original_counters': after,
            'timing': 'unchanged frozen SSE/perf38 definition; all child READY before release'}
        save(root / 'group-result.json', value, 4 * 1024**2)
        aggregate_gate(window)
        return value
    except BaseException as error:
        failure = error
        save(root / 'FAILURE.json', {'status': 'ASSIGNED_GROUP_FAILED_NOT_REPLACED',
            'error': repr(error)[:4096], 'row_id': group['row_id'],
            'arm': group['arm'], 'ordinal': group['ordinal'], 'phase': group['phase']})
        raise
    finally:
        try:
            pump.close()
        except BaseException as cleanup:
            save(root / 'CLEANUP-FAILURE.json', {'status': 'OWNED_REQUEST_CLEANUP_FAILED',
                'primary_error': repr(failure), 'cleanup_error': repr(cleanup)[:4096]})
            raise
