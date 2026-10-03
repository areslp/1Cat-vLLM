"""Exact frozen 670-request arm inventory; no confirmation or replacement."""
import argparse
import json
from pathlib import Path
import time

from frozen import require
from execution_budget import BUDGETS
from io_tools import save, aggregate_gate
from original_guard import Observation, OriginalGuard
from request_groups import run_group
from artifact_budget import BudgetMonitor, free_space_preflight

ARMS = ('A0', 'B', 'A2')


def arm_groups(arm):
    if arm not in ARMS:
        raise ValueError('only frozen initial A0/B/A2 matrix')
    matrix = json.loads(require('matrix.frozen.json').read_text())
    contract = json.loads(require('contract.json').read_text())
    if (matrix['contract_sha256'] != __import__('frozen').PINS['contract.json']
            or contract['initial_matrix_requests'] != 2010
            or contract['requests_per_arm_including_warmups_primes_cancels'] != 670):
        raise ValueError('frozen matrix contract differs')
    groups = [row for row in matrix['groups'] if row['arm'] == arm]
    if len(groups) != 142 or sum(len(g['requests']) + bool(g['prime']) for g in groups) != 670:
        raise ValueError('all assigned warmups/primes/cancels required')
    names = [(g['row_id'], g['phase'], g['ordinal']) for g in groups]
    if len(names) != len(set(names)):
        raise ValueError('duplicate group phase/ordinal')
    return groups


def run(arm, output, window, binding):
    groups = arm_groups(arm)
    root = Path(output)
    root.mkdir(mode=0o700)
    save(root / 'disk-preflight.json', free_space_preflight(window))
    save(root / 'INVENTORY.json', {'arm': arm, 'groups': len(groups),
        'requests_including_warmups_primes_cancels': 670,
        'order': [[g['row_id'], g['phase'], g['ordinal']] for g in groups]})
    guard = OriginalGuard(Observation(binding), root / 'guard')
    began = time.perf_counter()
    deadline = began + BUDGETS['matrix_per_arm']
    results = []
    monitor = BudgetMonitor(window, root / 'artifact-budget-samples.jsonl')
    try:
        monitor.tick(force=True)
        for number, group in enumerate(groups):
            destination = root / f'{number:03d}-{group["row_id"]}-{group["phase"]}-{group["ordinal"]}'
            value = run_group(group, destination, deadline, guard=guard,
                              window=window, budget_check=monitor.tick)
            results.append({'row_id': group['row_id'], 'phase': group['phase'],
                'ordinal': group['ordinal'], 'path': str(destination / 'group-result.json'),
                'status': value['status']})
            save(root / f'group-complete-{number:03d}.json', results[-1])
        value = {'status': 'COMPLETE_FROZEN_W2_MATRIX_ARM_NOT_ANALYSIS',
            'arm': arm, 'group_results': results, 'requests': 670,
            'elapsed_s': time.perf_counter() - began,
            'aggregate_measured_bytes': aggregate_gate(window)}
        save(root / 'RESULT.json', value)
        return value
    except BaseException as error:
        save(root / 'FAILURE.json', {'status': 'INCOMPLETE_FROZEN_MATRIX_STOP',
            'error': repr(error)[:4096], 'arm': arm, 'completed_groups': len(results),
            'assigned_groups': len(groups), 'current_group': len(results),
            'elapsed_s': time.perf_counter() - began})
        raise
    finally:
        monitor.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--binding', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--window', required=True)
    args = parser.parse_args()
    path = Path(args.binding)
    if path.stat().st_size > 1024**2:
        raise ValueError('startup binding size')
    run(args.arm, args.output, args.window, json.loads(path.read_text()))


if __name__ == '__main__':
    main()
