"""Same-process runtime/source receipt for only new identity boundary checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parent


def main():
    if sys.argv[1:] not in ([], ['--attempt2'], ['--attempt3']):
        raise ValueError('only named create-only CPU attempts')
    suffix = sys.argv[1].removeprefix('--') if sys.argv[1:] else 'attempt1'
    destination = ROOT / ('evidence/cpu-revision3-' + suffix)
    destination.mkdir()
    sys.path.insert(0, str(ROOT / 'tests'))
    import test_startup_revision3
    import test_identity_revision3
    import test_controller
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromModule(test_startup_revision3),
        unittest.defaultTestLoader.loadTestsFromModule(test_identity_revision3),
        test_controller.W2ControllerCPU('test_real_staged_import_resolves_exact_sibling_package')])
    tested = ['startup.py', 'common.py', 'original_guard.py', 'identity_contract.py',
              'service_binding.py', 'source_auth.py', 'ops.py',
              'control/step58_control.py', 'tests/test_startup_revision3.py',
              'tests/test_identity_revision3.py', 'tests/test_controller.py',
              'run_revision3_cpu.py']
    sources = {}
    for name in tested:
        path = ROOT / name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        saved = destination / 'executed-source' / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        with saved.open('xb') as stream:
            stream.write(path.read_bytes())
    runtime = {'executable': sys.executable, 'python': sys.version,
        'uv': subprocess.check_output(['/Users/l/.local/bin/uv', '--version'], text=True).strip(),
        'argv': sys.argv, 'profile': 'uv run --no-project --managed-python --python 3.12 --no-python-downloads',
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'source_sha256': sources}
    with (destination / 'runtime-and-source.json').open('x') as stream:
        json.dump(runtime, stream, indent=2, sort_keys=True)
        stream.write('\n')
    with (destination / 'stderr.log').open('x') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    code = 0 if result.wasSuccessful() else 1
    value = {'status': 'PASS_SOURCE_CPU_ONLY_NOT_LIVE_STARTUP' if code == 0 else 'FAIL_CPU',
        'testsRun': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'runtime': runtime, 'Torch_imported': 'torch' in sys.modules,
        'VLLM_imported': 'vllm' in sys.modules, 'raw_exit': code,
        'scope': 'Actual collector+capture contract+Observation.actual_identity over closed run2 bytes; explicit synthetic OS, native-source I/O and HTTP adapters. Actual staged-layout import is a separate bounded CPU child.'}
    with (destination / 'RESULT.json').open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    (destination / 'exit').write_text(str(code) + '\n')
    print(json.dumps({k: value[k] for k in ('status', 'testsRun', 'failures', 'errors', 'raw_exit')}))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
