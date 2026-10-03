"""Reproduce only the real failed startup boundary; no host/network calls."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'service-w2-control-prep2'
ACTUAL = ROOT / 'evidence/run2-actual'


def main():
    destination = ROOT / 'evidence/reproduce-run2-attempt1'
    destination.mkdir()
    source = OLD / 'startup.py'
    text = source.read_text()
    node = next(n for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name == 'collect')
    def forbidden():
        raise AssertionError('must fail before pin/proc/GPU/network operations')
    namespace = {'pinned_inputs': forbidden}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'),
         namespace)
    invocation = json.loads((ACTUAL /
        'control/attempt1/A0/startup-invocation.json').read_text())
    readiness = json.loads((ACTUAL /
        'control/attempt1/A0/readiness.json').read_text())
    try:
        namespace['collect']('A0', invocation['unit'], invocation['config_path'],
                            ACTUAL / 'control/attempt1/A0/api-task-config.private.json',
                            ACTUAL / 'control/attempt1/A0')
    except ValueError as error:
        if str(error) != 'exact first W2 arm/unit identity':
            raise
        trace = traceback.format_exc()
    else:
        raise AssertionError('old defect not reproduced')
    (destination / 'old-traceback.txt').write_text(trace)
    value = {'status': 'REPRODUCED_OLD_RUN2_STARTUP_IDENTITY_REJECTION',
        'unit': invocation['unit'], 'actual_invocation_id': invocation['invocation_id'],
        'actual_maturity': readiness,
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'scope': 'Actual old collect function AST with real run2 arguments; pin I/O forbidden. No new service observation.',
        'runtime': {'executable': sys.executable, 'python': sys.version,
                    'uv': subprocess.check_output(['/Users/l/.local/bin/uv', '--version'], text=True).strip()},
        'Torch_imported': 'torch' in sys.modules}
    (destination / 'RESULT.json').write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': value['status'], 'unit': value['unit'],
                      'python': sys.version.split()[0]}))


if __name__ == '__main__':
    main()
