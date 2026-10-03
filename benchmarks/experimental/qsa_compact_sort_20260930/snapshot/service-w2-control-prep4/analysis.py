"""Complete frozen matched groups and CI, never replacement or best-arm choice."""
import argparse
import json
from pathlib import Path
import sys

from frozen import module, require
from io_tools import save, sha


def analyze(window):
    window = Path(window)
    frozen = json.loads(require('matrix.frozen.json').read_text())
    rows = []
    pins = {}
    for arm in ('A0', 'B', 'A2'):
        summary = window / 'control/attempt1' / arm / 'matrix/RESULT.json'
        value = json.loads(summary.read_text())
        if (value['status'] != 'COMPLETE_FROZEN_W2_MATRIX_ARM_NOT_ANALYSIS'
                or value['arm'] != arm or value['requests'] != 670
                or len(value['group_results']) != 142):
            raise ValueError('whole preassigned arm matrix incomplete')
        pins[str(summary)] = sha(summary)
        for registered in value['group_results']:
            path = Path(registered['path'])
            if (not path.resolve().is_relative_to(summary.parent.resolve())
                    or path.is_symlink() or path.stat().st_size > 4 * 1024**2):
                raise ValueError('group result outside exact arm or unbounded')
            row = json.loads(path.read_text())
            if (row['arm'] != arm or row['status'] != registered['status']
                    or any(row[key] != registered[key]
                           for key in ('row_id', 'phase', 'ordinal'))):
                raise ValueError('actual group path/index/phase binding differs')
            pins[str(path)] = sha(path)
            rows.append(row)
    analyzer = module('group_analyzer')
    # Warmup ordinals overlap measured ordinals. Validate complete separate
    # inventories, then exclude warmups from timing exactly as frozen.
    analyzer.bind_groups([row for row in frozen['groups'] if row['phase'] == 'warmup'],
                         [row for row in rows if row['phase'] == 'warmup'])
    triples = analyzer.bind_groups(
        [row for row in frozen['groups'] if row['phase'] == 'measured'],
        [row for row in rows if row['phase'] == 'measured'])
    definitions = {row['row_id']: row for row in frozen['groups']
                   if row['phase'] == 'measured'}
    results = [analyzer.analyze(name, values, kind=definitions[name]['kind'])
               for name, values in sorted(triples.items())]
    stable_path = window / 'control/attempt1/B/stability/RESULT.json'
    stable = json.loads(stable_path.read_text())
    if (stable['status'] != 'COMPLETE_FROZEN_STABILITY_NOT_PERFORMANCE_ADMISSION'
            or stable['completed'] < 320 or stable['actual_first_to_last_s'] < 600
            or not all(stable['checks'].values())):
        raise ValueError('frozen B stability insufficient')
    pins[str(stable_path)] = sha(stable_path)
    statuses = {row['status'] for row in results}
    status = ('NO_GO' if 'NO_GO' in statuses else 'INCONCLUSIVE'
              if 'INCONCLUSIVE' in statuses else
              'PASS_INITIAL_REQUIRES_SEPARATE_APPROVED_CONFIRMATION')
    result = {'status': status, 'scope': 'initial W2 matched-group estimate; not final admission',
        'rows': results, 'input_pins': pins, 'matrix_sha256': sha(require('matrix.frozen.json')),
        'analysis_contract_sha256': sha(require('ANALYSIS-CONTRACT.md')),
        'raw_count': {'groups': len(rows), 'warmups': 78, 'measured': 348,
                      'matrix_requests': 2010},
        'B_stability': stable, 'confirmation': 'NOT_EXECUTED_NOT_AUTHORIZED_BY_THIS_RUN',
        'runtime': {'executable': sys.executable, 'python': sys.version}}
    destination = window / 'analysis'
    destination.mkdir(mode=0o700)
    save(destination / 'RESULT.json', result, 4 * 1024**2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', required=True)
    args = parser.parse_args()
    result = analyze(args.window)
    print(result['status'])
    raise SystemExit({'NO_GO': 2, 'INCONCLUSIVE': 3}.get(result['status'], 0))


if __name__ == '__main__':
    main()
