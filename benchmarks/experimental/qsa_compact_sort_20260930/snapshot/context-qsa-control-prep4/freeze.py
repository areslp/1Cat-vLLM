"""Local source/dependency seal; never accesses host, service or devices."""
import ast
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
BASE = PACKAGE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    matrix = BASE / 'context-qsa-prep1'
    matrix_manifest = matrix / 'MANIFEST.json'
    if not matrix_manifest.is_file():
        raise ValueError('final reviewed matrix source seal absent')
    paths = {matrix_manifest}
    for row in json.loads(matrix_manifest.read_text())['files']:
        path = matrix / row['path']
        if (path.is_symlink() or not path.is_relative_to(matrix)
                or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']):
            raise ValueError('reviewed matrix manifest/input bytes changed')
        paths.add(path)
    paths.update(path for path in (BASE / 'service-w2-control-prep4').iterdir()
                 if path.is_file() and path.suffix in ('.py', '.json'))
    service = BASE / 'service-implementation-retry2'
    paths.add(service / 'manifest.json')
    paths.update(service / row['path'] for row in json.loads((service / 'manifest.json').read_text())['files'])
    paths.add(BASE / 'numeric-run7/control/configs/A0.service.json')
    # The original OriginalGuard imports frozen parsers/fence; pin its entire
    # existing declared dependency closure without importing old run/matrix.
    for row in json.loads((BASE / 'mixed-order-diagnostic-prep1/DEPENDENCIES.json').read_text())['files']:
        paths.add(BASE / row['relative_path'])
    for row in json.loads((PACKAGE / 'CONTRACT.json').read_text())['prior_evidence_pins']:
        path = BASE / row['relative_path']
        if sha(path) != row['sha256']:
            raise ValueError('actual prior resource observation changed')
        paths.add(path)
    paths.add(BASE / 'context-qsa-resource-control-prep1/nvml_capture.py')
    transport = BASE / 'context-qsa-transport-prep1'
    paths.add(transport / 'MANIFEST.json')
    for row in json.loads((transport / 'MANIFEST.json').read_text())['files']:
        path = transport / row['path']
        if (path.is_symlink() or path.stat().st_size != row['bytes']
                or sha(path) != row['sha256']):
            raise ValueError('sealed transport payload changed')
        paths.add(path)
    for row in json.loads((transport / 'DEPENDENCIES.json').read_text())['files']:
        paths.add(BASE / row['relative_path'])
    dependencies = {'files': [{'relative_path': str(path.relative_to(BASE)),
        'sha256': sha(path), 'bytes': path.stat().st_size} for path in sorted(paths)]}
    with (PACKAGE / 'DEPENDENCIES.json').open('x') as stream:
        stream.write(json.dumps(dependencies, indent=2) + '\n')
    files = []
    active = {'evidence/FREEZE.stdout', 'evidence/FREEZE.stderr', 'evidence/FREEZE.exit'}
    for path in sorted(PACKAGE.rglob('*')):
        if path.is_symlink():
            raise ValueError('source symlink prohibited')
        if not path.is_file() or path.name == 'MANIFEST.json' or '__pycache__' in path.parts:
            continue
        if str(path.relative_to(PACKAGE)) in active:
            continue
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
        files.append({'path': str(path.relative_to(PACKAGE)), 'sha256': sha(path),
                      'bytes': path.stat().st_size})
    value = {'status': 'LOCAL_SOURCE_CPU_ONLY_NOT_LAUNCHED', 'files': files,
        'inventory': json.loads((matrix / 'CONTRACT.json').read_text())['inventory'],
        'no_live_claim': True, 'excludes': ['MANIFEST.json', '__pycache__', *sorted(active)]}
    with (PACKAGE / 'MANIFEST.json').open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'status': value['status'], 'files': len(files),
        'dependencies': len(paths), 'manifest_sha256': sha(PACKAGE / 'MANIFEST.json')}))


if __name__ == '__main__':
    main()
