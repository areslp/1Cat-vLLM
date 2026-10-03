"""Read-only native CPU validation; receipts go to stdout, never this package."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
BASE = PACKAGE.parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or sys.dont_write_bytecode is not True:
        raise ValueError('native CPU-only check requires CUDA_VISIBLE_DEVICES= and -B')
    manifest = json.loads((PACKAGE / 'MANIFEST.json').read_text())
    for row in manifest['files']:
        path = PACKAGE / row['path']
        if path.is_symlink() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('control source differs: ' + str(path))
    dependencies = json.loads((PACKAGE / 'DEPENDENCIES.json').read_text())
    for row in dependencies['files']:
        path = BASE / row['relative_path']
        if path.is_symlink() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('exact dependency differs: ' + str(path))
    python_files, shells = 0, []
    for path in PACKAGE.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
        python_files += 1
    for path in sorted(PACKAGE.rglob('*.sh')):
        subprocess.run(['/bin/bash', '-n', str(path)], check=True, timeout=10,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        shells.append(str(path.relative_to(PACKAGE)))
    sys.path.insert(0, str(PACKAGE / 'control'))
    import step58_control as controller
    sys.path.insert(0, str(PACKAGE))
    import startup
    if Path(startup.__file__).resolve() != PACKAGE / 'startup.py':
        raise ValueError('owned startup collector import selected wrong source')
    import final_guard
    if Path(final_guard.__file__).resolve() != PACKAGE / 'final_guard.py':
        raise ValueError('owned final collector import selected wrong source')
    import collect_startup
    import http_runner
    from guard_activation import proof
    guard_proof = proof()
    print(json.dumps({'resource_guard_identity': guard_proof, 'entry_modules': {
        m.__name__: str(Path(m.__file__).resolve()) for m in
        (controller, startup, final_guard, collect_startup, http_runner)}, 'status': 'PASS_READ_ONLY_SOURCE_CPU_IMPORTS',
        'source_files': len(manifest['files']), 'dependencies': len(dependencies['files']),
        'AST_files': python_files, 'bash_syntax': shells,
        'budget': controller.budget_validate(), 'python': sys.version,
        'executable': sys.executable, 'profile': 'caller_selected_stdlib_source_check',
        'declared_host_python': controller.PY,
        'interpreter_is_declared_host_venv': sys.executable == controller.PY,
        'package_writes': False, 'HTTP': False, 'SSH': False, 'GPU': False,
        'not_a_live_preflight_or_service_claim': True}, sort_keys=True))


if __name__ == '__main__':
    main()
