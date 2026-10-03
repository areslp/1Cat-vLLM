"""Create-only host staging; no service/HTTP/GPU/SSH operations."""
import json
from pathlib import Path
import shutil
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / 'control'))
import step58_control as c


def main():
    c.require(PACKAGE == c.BASE / PACKAGE.name, 'stage only at exact deployed package path')
    for row in c.read(PACKAGE / 'MANIFEST.json')['files']:
        c.require(c.sha(PACKAGE / row['path']) == row['sha256'], 'control source changed')
    c.D.mkdir(mode=0o700)
    (c.D / 'baseline').mkdir(mode=0o700)
    shutil.copytree(PACKAGE / 'control', c.D / 'control', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(PACKAGE / 'restoration', c.D / 'restoration', ignore=shutil.ignore_patterns('__pycache__'))
    (c.D / 'source').mkdir(mode=0o700)
    shutil.copyfile(PACKAGE / 'source/restoration-external-reference-pins.json',
                    c.D / 'source/restoration-external-reference-pins.json')
    c.CFG_DIR.mkdir(mode=0o700)
    for arm in ('A0', 'A2'):
        c.save(c.CFG_DIR / (arm + '.service.json'), c.derive(arm, c.D))
    import restoration_dependencies
    restoration_dependencies.validate(c.D)
    print(json.dumps({'status': 'STAGED_ONLY_NOT_STARTED', 'window': str(c.D),
                      'next': 'fresh baseline snapshot/unit then prepare_plan.py'}))


if __name__ == '__main__':
    main()
