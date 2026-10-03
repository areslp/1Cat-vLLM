"""Affected selector/file-constructor boundaries only; no requests or live calls."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

P = Path(__file__).resolve().parent
BASE = P.parent
CANONICAL = Path('/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930')
sys.path.insert(0, str(P))
from guard_activation import activate, proof, require_binding
identity, guard = activate()
import http_runner
import startup
from transport_activation import owned, require_runner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not sys.dont_write_bytecode:
        raise ValueError('explicit managed CPU-only/-B contract required')
    root = args.output
    root.mkdir(mode=0o700)
    checks = []

    def passed(label):
        checks.append({'check': label, 'ok': True})

    def reject(label, call):
        try:
            call()
        except ValueError as error:
            checks.append({'check': label, 'ok': True, 'rejection': str(error)})
        else:
            raise AssertionError('unexpected acceptance: ' + label)

    module = owned()
    assert http_runner.original.Pump is module.Pump
    assert http_runner.original.run_group.__globals__['Pump'] is module.Pump
    assert module.Pump.add.__globals__['WORKER'] == BASE / 'context-qsa-transport-prep1/transport_worker.py'
    assert require_runner(http_runner.original) == proof()['transport_identity']
    assert sha(module.__file__) == sha(BASE / 'context-qsa-prep1/transport_deadline_context.py')
    passed('actual frozen run_group global Pump selects reviewed same-byte relative worker')

    previous = http_runner.original.Pump
    try:
        http_runner.original.Pump = object()
        reject('unknown/replaced actual runner Pump rejected', lambda: require_runner(http_runner.original))
    finally:
        http_runner.original.Pump = previous
    previous = module.WORKER
    try:
        module.WORKER = BASE / 'context-qsa-prep1/transport_worker.py'
        reject('same-source Pump with old512KiB worker path rejected', owned)
    finally:
        module.WORKER = previous

    # Actual stored producer shape + exact four native capture bytes, copied to
    # local fixture paths. No identity()/metrics()/live service method is called.
    actual = BASE / 'context-qsa-run2/control/attempt1/A0'
    old = json.loads((actual / 'guard-binding.json').read_text())
    receipt = json.loads((actual / 'startup-gate.json').read_text())
    pins = copy.deepcopy(old['capture_receipts'])
    for pin in pins:
        source = BASE / Path(pin['path']).relative_to(CANONICAL)
        assert sha(source) == pin['sha256']
        target = root / f'capture-rank{pin["rank"]}.json'
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        pin['path'] = str(target)
    receipt['capture_receipts'] = pins
    receipt['resource_guard_identity'] = proof()
    startup_path = root / 'startup-fixture.json'
    save(startup_path, receipt)
    preliminary = copy.deepcopy(old)
    for key in ('resource_guard_identity', 'startup_path', 'startup_sha256'):
        del preliminary[key]
    preliminary['capture_receipts'] = pins
    preliminary['source_files'] = dict(proof()['source_pins'])
    binding = startup.publish_binding(root, preliminary, startup_path)
    assert len(binding) == 12
    assert set(binding) == set(old) - {'resource_guard_identity'}
    observation = guard.Observation(binding)
    assert len(observation.descriptors) == len(binding['source_files'])
    assert require_binding(binding) == proof()
    passed('actual startup publication constructs unchanged OriginalObservation exact12 with nested transport proof')

    bad = copy.deepcopy(binding)
    bad['transport_identity'] = proof()['transport_identity']
    reject('unknown extra core key rejected by real constructor', lambda: guard.Observation(bad))
    bad = copy.deepcopy(binding)
    bad['startup_sha256'] = '0' * 64
    reject('startup receipt SHA checked before nested metadata', lambda: require_binding(bad))
    bad = copy.deepcopy(binding)
    worker_path = proof()['transport_identity']['worker_path']
    bad['source_files'][worker_path] = '0' * 64
    reject('worker source pin tamper rejected by requirement', lambda: require_binding(bad))
    reject('worker source pin tamper rejected by actual constructor', lambda: guard.Observation(bad))
    bad = copy.deepcopy(binding)
    pump_path = proof()['transport_identity']['Pump_module_path']
    del bad['source_files'][pump_path]
    reject('missing actual Pump source pin rejected', lambda: require_binding(bad))
    changed = copy.deepcopy(receipt)
    changed['resource_guard_identity']['transport_identity']['worker_sha256'] = '0' * 64
    path = root / 'metadata-tamper.json'
    save(path, changed)
    bad = dict(binding, startup_path=str(path), startup_sha256=sha(path))
    reject('transport metadata tamper rejected even after receiptSHA updated', lambda: require_binding(bad))

    entries = {}
    for entry in ('collect_startup', 'http_runner', 'final_guard', 'step58_control'):
        code = (f'import sys,json;sys.path.insert(0,{str(P)!r});'
                f'sys.path.insert(0,{str(P / "control")!r});import {entry};'
                'from guard_activation import proof;print(json.dumps(proof()))')
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        entries[entry] = json.loads(result.stdout)
        assert entries[entry] == proof()
    passed('four actual entry imports preserve identical owned transport/resource proof')
    tested = [P / name for name in (
        'transport_activation.py', 'guard_activation.py', 'http_runner.py',
        'startup.py', 'check_transport_binding_cpu.py')]
    save(root / 'PROFILE.json', {
        'python': sys.version, 'executable': sys.executable,
        'profile': 'uv-managed3.12.13/no-project/-I/-B/CVD-empty',
        'source_pins': {str(path): sha(path) for path in tested},
        'HTTP': False, 'identity_calls': False, 'GPU': False,
        'fixture_basis': str(actual),
        'old_capture_bytes_reused_not_new_runtime_proof': True,
    })
    save(root / 'RESULT.json', {
        'status': 'PASS_AFFECTED_CPU_SOURCE_FILE_BINDING_ONLY',
        'checks': checks, 'count': len(checks),
        'resource_guard_identity': proof(), 'entry_proofs': entries,
        'new_transport_wire_tests': 0, 'old_run3_result': 'UNCHANGED_FAIL',
    })
    print(json.dumps({'status': 'PASS', 'checks': len(checks)}))


if __name__ == '__main__':
    main()
