"""Only the newly approved client capacity gate, no live cgroup/HTTP."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P))
import http_runner as h


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not sys.dont_write_bytecode:
        raise ValueError('managed CPU-only/-B profile required')
    root = args.output
    root.mkdir(mode=0o700)
    checks = []
    actual = h.MATRIX.parent / 'review/context-qsa-transport-native1024-20261002/client/client-cgroup.json'
    native = json.loads(actual.read_text())
    files = copy.deepcopy(native['files'])
    assert files['memory.max'] == str(1024**3)
    assert all(h.client_checks(files).values())
    checks.append('actual native1024 cgroup files admit at exact1GiB/events.max0/stat/Swap0/OOM0')
    arm = 'A0'
    group = '/system.slice/step58-contextqsa4-http-a0.service'
    # Native file values are reused in a clearly synthetic scope/path receipt;
    # no claim that the future HTTP arm or kernel cgroup already ran.
    value = {'status': 'PASS_ACTUAL_CLIENT_CGROUP',
             'proc_self_cgroup_raw': '0::' + group + '\n',
             'cgroup_path': '/sys/fs/cgroup' + group,
             'files': files, 'checks': h.client_checks(files),
             'systemd_run_CLI_peak_used_as_actual_peak': False,
             'fixture_is_synthetic_future_unit_path': True}
    def validate(name, changed, reject):
        folder = root / name
        folder.mkdir()
        save(folder / 'client-resource.json', changed)
        try:
            h.validate_client_resource(folder, arm)
        except ValueError as error:
            assert reject, (name, str(error))
            checks.append({'reject': name, 'error': str(error)})
        else:
            assert not reject, name
            checks.append(name)
    validate('new1GiB readback positive', value, False)
    bad = copy.deepcopy(value)
    bad['files']['memory.max'] = str(512 * 1024**2)
    validate('old512 budget cannot be presented as new1GiB', bad, True)
    bad = copy.deepcopy(value)
    bad['files']['memory.peak'] = str(1024**3 + 1)
    validate('peak-over1GiB rejects despite forged alltrue checks', bad, True)
    bad = copy.deepcopy(value)
    bad['files']['memory.events'] = bad['files']['memory.events'].replace('max 0', 'max 1')
    validate('events.max pressure rejects despite forged alltrue checks', bad, True)
    bad = copy.deepcopy(value)
    bad['files']['memory.stat'] = ''
    validate('missing actual memory.stat rejects', bad, True)
    limits = json.loads((P / 'CONTRACT.json').read_text())['resources']
    assert limits['client_cgroup_bytes'] == h.CLIENT_CGROUP_BYTES == 1024**3
    tree = ast.parse((P / 'control/step58_control.py').read_text())
    timed = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'timed')
    properties = {n.value for n in ast.walk(timed) if isinstance(n, ast.Constant)
                  and isinstance(n.value, str) and n.value.startswith('--property=')}
    assert '--property=MemoryMax=1G' in properties and '--property=MemoryMax=512M' not in properties
    assert '--property=MemorySwapMax=0' in properties
    assert '--property=CPUQuota=200%' in properties and '--property=AllowedCPUs=14,42' in properties
    checks.append('contract/runtimeclient/readback1GiB consistent; CPU14,42/200%/Swap0 unchanged')
    save(root / 'RESULT.json', {'status': 'PASS_AFFECTED_CLIENT_BUDGET_CPU_ONLY',
        'checks': checks, 'count': len(checks), 'python': sys.version,
        'executable': sys.executable, 'actual_native_input_path': str(actual),
        'actual_native_input_sha256': hashlib.sha256(actual.read_bytes()).hexdigest(),
        'tested_source': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (P / 'http_runner.py', P / 'control/step58_control.py',
                         P / 'CONTRACT.json', Path(__file__).resolve())},
        'live_cgroup_or_model_claim': False})
    print(json.dumps({'status': 'PASS', 'checks': len(checks)}))


if __name__ == '__main__':
    main()
