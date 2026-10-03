"""Create-only staged copies of the exact frozen W2 controller manifest."""
import argparse
import shutil
from pathlib import Path
from common import PACKAGE, WINDOW, read, save, sha
from restoration_dependencies import validate as validate_recovery
from stage_bindings import declared_copies, validate_staged
import source_auth


def stage():
    source_auth.authenticate()
    validate_recovery(PACKAGE)
    source_auth.controller_pins()
    mapping = declared_copies(PACKAGE, WINDOW)
    WINDOW.mkdir(mode=0o700, exist_ok=False)
    for destination, (source, digest) in mapping.items():
        destination, source = Path(destination), Path(source)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with destination.open('xb') as out, source.open('rb') as stream:
            shutil.copyfileobj(stream, out)
        if sha(destination) != digest:
            raise ValueError('actual copy byte mismatch')
    (WINDOW / 'control/configs').mkdir(mode=0o700)
    save(WINDOW / 'control/staged-receipt.json', {
        'status': 'EMPTY_WINDOW_NO_HTTP_NO_SERVICE',
        'controller_manifest_sha256': sha(PACKAGE / 'manifest.json'),
        'copies': {dest: digest for dest, (_, digest) in mapping.items()}})
    validate_staged(PACKAGE, WINDOW)
    return mapping


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--create-only-window', action='store_true', required=True)
    parser.parse_args()
    stage()
    print('CREATE_ONLY_W2_WINDOW_NO_MODEL_START_NO_GENERATION')
