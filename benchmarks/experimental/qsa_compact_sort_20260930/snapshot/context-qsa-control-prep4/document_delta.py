"""Fixed source comparison only, after affected checks and before final seal."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

P = Path(__file__).resolve().parent
OLD = P.parent / 'context-qsa-control-prep3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def normalized(data):
    for before, after in (
        (b'context-qsa-control-prep3', b'context-qsa-control-prep4'),
        (b'context-qsa-run3', b'context-qsa-run4'),
        (b'step58-contextqsa3', b'step58-contextqsa4'),
        (b'STEP58_CONTEXTQSA3', b'STEP58_CONTEXTQSA4'),
    ):
        data = data.replace(before, after)
    return data


def main():
    reuse, differences, new = [], [], []
    old = {row['path']: row for row in json.loads((OLD / 'MANIFEST.json').read_text())['files']}
    for path in sorted(P.rglob('*')):
        if (not path.is_file() or path.relative_to(P).parts[0] in ('evidence',)
                or path.name in ('MANIFEST.json', 'DEPENDENCIES.json')
                or str(path.relative_to(P)).startswith('source/INTEGRATION')):
            continue
        name = str(path.relative_to(P))
        if name not in old:
            new.append(name)
            continue
        previous = OLD / name
        row = {'path': name, 'source_sha256': old[name]['sha256'],
               'current_sha256': sha(path),
               'scope_only_or_exact': normalized(previous.read_bytes()) == path.read_bytes()}
        reuse.append(row)
        if not row['scope_only_or_exact']:
            differences.extend(difflib.unified_diff(
                normalized(previous.read_bytes()).decode().splitlines(True),
                path.read_text().splitlines(True),
                fromfile='normalized-prep3/' + name, tofile='prep4/' + name))
    restorations = [row for row in reuse if row['path'].startswith('restoration/')]
    assert len(restorations) == 14 and all(row['scope_only_or_exact'] for row in restorations)
    assert (P / 'execution_budget.py').read_bytes() == (OLD / 'execution_budget.py').read_bytes()
    for name in ('identity_contract.py', 'nvml_capture.py', 'startup.py'):
        assert (P / name).read_bytes() == (OLD / name).read_bytes()
    shells = sorted(P.rglob('*.sh'))
    for path in shells:
        subprocess.run(['/bin/bash', '-n', str(path)], check=True,
                       capture_output=True, timeout=10)
    python_files = sorted(P.rglob('*.py'))
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path))
    (P / 'source/INTEGRATION.diff').write_text(''.join(differences))
    save(P / 'source/INTEGRATION-REVIEW.json', {
        'source_package': OLD.name, 'source_manifest_sha256': sha(OLD / 'MANIFEST.json'),
        'files': reuse, 'new_files': new,
        'restoration_scope_only': True, 'restoration_files': len(restorations),
        'budgets_exact_unchanged': True,
        'model_resource_and_identity_contract_bytes_unchanged': True,
        'startup_producer_constructor_bytes_unchanged': True,
        'matrix_body_order_salt_metrics_not_modified': True,
        'AST': len(python_files), 'bash_individually_checked': len(shells),
        'diff_sha256': sha(P / 'source/INTEGRATION.diff'),
        'old_files_modified': False,
    })
    print(json.dumps({'status': 'PASS_MINIMAL_SOURCE_DELTA', 'AST': len(python_files),
                      'bash': len(shells), 'restoration': len(restorations)}))


if __name__ == '__main__':
    main()
