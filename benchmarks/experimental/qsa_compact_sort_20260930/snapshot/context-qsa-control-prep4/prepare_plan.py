"""Bind exact frozen matrix and fresh staged baseline; root reviews output SHA."""
import argparse
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / 'control'))
import step58_control as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-pid', required=True, type=int)
    args = parser.parse_args()
    c.require(PACKAGE == c.BASE / PACKAGE.name, 'plan only at exact deployed package path')
    c.require(args.expected_pid > 0, 'fresh real original PID required')
    baseline = c.read(c.D / 'baseline/before-snapshot.json')
    c.require(int(baseline['systemd']['MainPID']) == args.expected_pid, 'fresh snapshot/PID mismatch')
    production = c.state(c.S)
    c.require(production['MainPID'] == str(args.expected_pid) and production['ActiveState'] == 'active'
              and c.re.fullmatch('[0-9a-f]{32}', production['InvocationID']) is not None,
              'fresh live production PID/invocation required')
    files = {}
    for row in c.read(PACKAGE / 'MANIFEST.json')['files']:
        files[str(PACKAGE / row['path'])] = row['sha256']
    files[str(PACKAGE / 'MANIFEST.json')] = c.sha(PACKAGE / 'MANIFEST.json')
    for row in c.read(PACKAGE / 'DEPENDENCIES.json')['files']:
        files[str(c.BASE / row['relative_path'])] = row['sha256']
    files.update({str(c.native.K / name): value for name, value in baseline['sources'].items()})
    files.update(baseline['files'])
    files.update(c.read(PACKAGE / 'source/restoration-external-reference-pins.json'))
    # Authenticate original SO/candidate/source/shim/inventory before stopping production.
    cfg = c.derive('A0', c.D)
    for name in ('candidate_binary', 'original_binary', 'original_shim', 'inventory_path'):
        expected = cfg['inventory_sha256'] if name == 'inventory_path' else cfg[name + '_sha256']
        c.require(c.sha(cfg[name]) == expected, 'pure runtime input differs: ' + name)
        files[cfg[name]] = expected
    for pin in cfg['source_pins'].values():
        c.require(c.sha(pin['path']) == pin['sha256'], 'deployed source pin differs')
        files[pin['path']] = pin['sha256']
    for path in (c.D / 'baseline/before-snapshot.json', c.D / 'baseline/before-unit.txt',
                 Path('/home/l/work/flash-next/run-flash-next-w12.sh'),
                 Path('/home/l/work/flash-next/diagnostics/logging.json')):
        files[str(path)] = c.sha(path)
    for folder in ('control', 'restoration', 'source'):
        for path in (c.D / folder).rglob('*'):
            if path.is_file():
                c.require(not path.is_symlink(), 'staged input symlink')
                files[str(path)] = c.sha(path)
    plan = {'schema': 'step58-contextqsa4-pure-ABA-control', 'reviewed': False,
        'window': str(c.D), 'units': c.UNITS, 'timer': c.TIMER, 'outer': c.OUTER,
        'expected_pid': args.expected_pid, 'budgets': c.BUDGETS,
        'expected_invocation_id': production['InvocationID'],
        'resource_contract': c.read(PACKAGE / 'CONTRACT.json')['resources'],
        'identity_contract_sha256': c.proof()['identity_module_sha256'],
        'arms': ['A0', 'B', 'A2'], 'modes': ['off', 'on', 'off'],
        'candidate_arms': 1, 'scheduler_observer': False,
        'inventory': c.read(c.MATRIX / 'CONTRACT.json')['inventory'],
        'matrix_path': str(c.MATRIX / 'matrix.frozen.json'),
        'pending_facts': [], 'files': dict(sorted(files.items()))}
    c.validate_plan(plan)
    path = c.D / 'control/plan.json'
    c.save(path, plan)
    print(json.dumps({'status': 'EXACT_PLAN_REQUIRES_ROOT_REVIEW', 'path': str(path),
        'sha256': c.sha(path), 'expected_pid': args.expected_pid,
        'source_only': True, 'inventory': plan['inventory']}))


if __name__ == '__main__':
    main()
