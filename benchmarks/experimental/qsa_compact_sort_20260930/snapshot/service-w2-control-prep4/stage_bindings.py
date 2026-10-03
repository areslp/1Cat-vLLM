"""Bind every actual runnable window copy to exact frozen package bytes."""
from pathlib import Path
from common import read,sha


def declared_copies(package,window):
    package=Path(package);window=Path(window)
    rows=read(package/'manifest.json')['files'];mapping={}
    for row in rows:
        relative=Path(row['path'])
        if relative.parts[0] in ('control','restoration'):
            dest=window/relative
        elif str(relative)=='source/run3_snapshot.production.py':
            dest=window/'baseline/snapshot.py'
        else:continue
        source=package/relative
        if (not source.resolve().is_relative_to(package.resolve())
                or not dest.resolve().is_relative_to(window.resolve())
                or sha(source)!=row['sha256']):
            raise ValueError('declared stage source identity differs')
        mapping[str(dest)]=(str(source),row['sha256'])
    if not mapping or str(window/'baseline/snapshot.py') not in mapping:
        raise ValueError('incomplete frozen staged source inventory')
    return mapping


def validate_staged(package,window,plan_files=None):
    package=Path(package);window=Path(window)
    expected=declared_copies(package,window)
    path=window/'control/staged-receipt.json';receipt=read(path)
    if (receipt['status']!='EMPTY_WINDOW_NO_HTTP_NO_SERVICE'
            or receipt['controller_manifest_sha256']!=sha(package/'manifest.json')
            or set(receipt['copies'])!=set(expected)):
        raise ValueError('staged receipt is not the COMPLETE frozen mapping')
    required={str(path):sha(path)}
    for dest,(_,digest) in expected.items():
        if receipt['copies'][dest]!=digest or sha(dest)!=digest:
            raise ValueError('staged executable bytes/source pin differ: '+dest)
        required[dest]=digest
    if plan_files is not None and any(plan_files.get(p)!=h for p,h in required.items()):
        raise ValueError('plan omitted/altered actual staged executable binding')
    return required
