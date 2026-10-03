"""Bind the frozen run_group's global Pump to the reviewed relative worker."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
TRANSPORT = PACKAGE.parent / 'context-qsa-transport-prep1'
MATRIX = PACKAGE.parent / 'context-qsa-prep1'
NAME = '_context4_owned_transport'
MANIFEST_SHA = '6bda93a24d43514e44a5697fa4d5bc92f58b59bc1ae413e7164ba9466c56af66'
PINS = {
    'check_wire.py': '738297f1529826dd9796d95c2f9d22f0444908d486e3c8443855b28bbbdbeff3',
    'transport_worker.py': 'a309a920992e9be021026b293cf4fd7a40349b1755103acd53dde9ea5aa2b998',
    'transport_deadline_context.py': 'a60c9910a492b2666a33e32ed705a6085644c9626065c313a6d4474954ea7c14',
    'dependencies.py': '24d0fc96214ab14208543a5e84db21496779dbbaaa04e7002e294aaf8ef46149',
    'DEPENDENCIES.json': 'c7509ace6279d3048d55c9815ac0d131c6e25fa6ed1ca597d36004a2b2ec7b04',
}
FROZEN_RUNNER_SHA = '578bb48ae5b8b9d33f08454f86f27181892f6e4aef6197f479177e2a1d0483ba'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owned():
    if sha(TRANSPORT / 'MANIFEST.json') != MANIFEST_SHA:
        raise ValueError('reviewed transport manifest changed')
    for name, digest in PINS.items():
        path = TRANSPORT / name
        if path.is_symlink() or sha(path) != digest:
            raise ValueError('reviewed transport source changed: ' + name)
    for name in ('transport_deadline_context.py', 'dependencies.py', 'DEPENDENCIES.json'):
        if (TRANSPORT / name).read_bytes() != (MATRIX / name).read_bytes():
            raise ValueError('Pump/dependency bytes must match frozen matrix source')
    if json.loads((TRANSPORT / 'MANIFEST.json').read_text())['stable_sources'] != PINS:
        raise ValueError('transport stable source declaration differs')
    # Load before the frozen runner imports the generic dependency module.
    # A conflicting preloaded dependency is rejected, not silently overwritten.
    dependencies = sys.modules.get('dependencies')
    if dependencies is not None and Path(dependencies.__file__).resolve() != TRANSPORT / 'dependencies.py':
        raise ValueError('transport dependency module was already selected elsewhere')
    sys.path.insert(0, str(TRANSPORT))
    module = sys.modules.get(NAME)
    path = TRANSPORT / 'transport_deadline_context.py'
    if module is None:
        spec = importlib.util.spec_from_file_location(NAME, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[NAME] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[NAME]
            raise
    if (Path(module.__file__).resolve() != path
            or module.WORKER != TRANSPORT / 'transport_worker.py'
            or module.Pump.__module__ != NAME
            or Path(module.Pump.add.__code__.co_filename).resolve() != path
            or Path(module.Job.signal_owned.__code__.co_filename).resolve() != path):
        raise ValueError('actual owned Pump/relative worker binding differs')
    # Restore control helper priority; the selected dependency is cached/pinned.
    sys.path.insert(0, str(PACKAGE))
    return module


def proof():
    module = owned()
    paths = [TRANSPORT / name for name in PINS]
    paths.extend((TRANSPORT / 'MANIFEST.json', PACKAGE / 'transport_activation.py'))
    return {
        'status': 'OWNED_SAME_SOURCE_PUMP_AND_2MiB_WORKER_SELECTED',
        'Pump_module_path': str(Path(module.__file__).resolve()),
        'Pump_module_sha256': PINS['transport_deadline_context.py'],
        'Pump_class_module': module.Pump.__module__,
        'Pump_add_code_path': str(Path(module.Pump.add.__code__.co_filename).resolve()),
        'worker_path': str(module.WORKER),
        'worker_sha256': PINS['transport_worker.py'],
        'dependency_module_path': str(Path(sys.modules['dependencies'].__file__).resolve()),
        'transport_manifest_sha256': MANIFEST_SHA,
        'source_pins': {str(path): sha(path) for path in paths},
        'source_selection_not_actual_HTTP_or_performance': True,
    }


def bind(runner):
    module = owned()
    path = MATRIX / 'runner.py'
    if (Path(runner.__file__).resolve() != path or sha(path) != FROZEN_RUNNER_SHA
            or runner.run_group.__globals__ is not runner.__dict__):
        raise ValueError('exact frozen runner/global Pump owner required')
    previous = runner.Pump
    if previous is not module.Pump:
        original_path = MATRIX / 'transport_deadline_context.py'
        if (Path(previous.add.__code__.co_filename).resolve() != original_path
                or Path(sys.modules[previous.__module__].__file__).resolve() != original_path):
            raise ValueError('unknown preexisting runner Pump; no silent replacement')
        runner.Pump = module.Pump
    if runner.run_group.__globals__['Pump'] is not module.Pump:
        raise ValueError('actual run_group did not bind reviewed owned Pump')
    return proof()


def require_runner(runner):
    module = owned()
    if runner.Pump is not module.Pump or runner.run_group.__globals__['Pump'] is not module.Pump:
        raise ValueError('frozen run_group actual Pump changed after explicit binding')
    return proof()
