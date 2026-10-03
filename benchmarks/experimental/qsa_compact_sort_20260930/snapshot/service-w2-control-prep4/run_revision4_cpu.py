"""One same-process managed runtime receipt for the targeted time consumers."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parent


def main():
    if sys.argv[1:] not in ([], ['--only-model-attempt2']):
        raise ValueError('only the one affected fixture corrective attempt is allowed')
    second = bool(sys.argv[1:])
    destination = ROOT / ('evidence/cpu-revision4-attempt2' if second else 'evidence/cpu-revision4-attempt1')
    destination.mkdir()
    os.environ['W2_CPU_EVIDENCE_DIR'] = str(destination)
    tested = ['execution_budget.py', 'execution-budget-revision.json', 'matrix.py',
              'ops.py', 'plan.py', 'prepare_plan.py', 'source_auth.py', 'common.py',
              'transport_deadline.py', 'run_revision4_cpu.py',
              'tests/test_time_revision4.py']
    tested += [str(p.relative_to(ROOT)) for p in (ROOT / 'control').glob('*.sh')]
    sources = {}
    for name in tested:
        data = (ROOT / name).read_bytes()
        sources[name] = hashlib.sha256(data).hexdigest()
        copy = destination / 'executed-source' / name
        copy.parent.mkdir(parents=True, exist_ok=True)
        with copy.open('xb') as stream:
            stream.write(data)
    runtime = {'executable': sys.executable, 'python': sys.version,
        'uv': subprocess.check_output(['/Users/l/.local/bin/uv', '--version'], text=True).strip(),
        'argv': sys.argv, 'profile': 'uv run --no-project --managed-python --python 3.12 --no-python-downloads',
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'source_sha256': sources}
    with (destination / 'runtime-and-source.json').open('x') as stream:
        json.dump(runtime, stream, indent=2, sort_keys=True)
        stream.write('\n')
    sys.path.insert(0, str(ROOT / 'tests'))
    import test_time_revision4
    suite = (unittest.TestSuite([test_time_revision4.TimeRevision4(
        'test_actual_model_start_argv_derives4800_5800_and_exactunits')]) if second else
        unittest.defaultTestLoader.loadTestsFromModule(test_time_revision4))
    with (destination / 'stderr.log').open('x') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    code = 0 if result.wasSuccessful() else 1
    value = {'status': 'PASS_TARGETED_TIME_SOURCE_CPU_NOT_LIVE_W2' if code == 0 else 'FAIL_CPU',
        'raw_exit': code, 'testsRun': result.testsRun, 'failures': len(result.failures),
        'errors': len(result.errors), 'runtime': runtime,
        'Torch_imported': 'torch' in sys.modules, 'VLLM_imported': 'vllm' in sys.modules,
        'scope': 'Actual matrix.run, ops.timed/start argv and plan.validate with explicit group/guard/OS/ready mocks. Actual pinned original contract/matrix/amendment reads and bash-n. No model/HTTP/systemd execution. Old suites not rerun.'}
    with (destination / 'RESULT.json').open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    with (destination / 'exit').open('x') as stream:
        stream.write(str(code) + '\n')
    print(json.dumps({k: value[k] for k in ('status', 'raw_exit', 'testsRun', 'failures', 'errors')}))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
