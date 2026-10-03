"""Select the owned one-constant resource contract before frozen guard imports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
LEGACY = PACKAGE.parent / 'service-w2-control-prep4'
IDENTITY_SHA = '7a96e11602306ef2a6df3aadbb14626b8ef2fb835f8ae310fab6ef83f5b7c5ea'
LEGACY_SHA = '5b8d193aa930363299ce930b87a344a686bf250a1e35734a89c456c5d2cf91c7'
GUARD_SHA = '66a5e2220501e9f522304628218bfe99a76bb8f7877a33571131411b679d32c9'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def activate():
    path = PACKAGE / 'identity_contract.py'
    old = LEGACY / 'identity_contract.py'
    if (sha(path) != IDENTITY_SHA or sha(old) != LEGACY_SHA
            or path.read_bytes() != old.read_bytes().replace(
                b'PROCESS_NVML_MIB = 32000', b'PROCESS_NVML_MIB = 32384')):
        raise ValueError('exact owned process-only constant delta required')
    identity = sys.modules.get('identity_contract')
    if identity is None:
        spec = importlib.util.spec_from_file_location('identity_contract', path)
        identity = importlib.util.module_from_spec(spec)
        sys.modules['identity_contract'] = identity
        spec.loader.exec_module(identity)
    if (Path(identity.__file__).resolve() != path
            or identity.PROCESS_NVML_MIB != 32384 or identity.DEVICE_NVML_MIB != 32384
            or identity.MODEL_CGROUP_BYTES != 120 * 1024**3):
        raise ValueError('wrong runtime identity module selected; no silent rebind')
    sys.path.insert(0, str(LEGACY))
    import original_guard
    if (Path(original_guard.__file__).resolve() != LEGACY / 'original_guard.py'
            or sha(original_guard.__file__) != GUARD_SHA
            or original_guard.gpu_identity is not identity.gpu_identity
            or original_guard.listener_identity is not identity.listener_identity):
        raise ValueError('frozen OriginalGuard must use actual owned functions')
    # Only the shared guard comes from LEGACY; owned controller helpers keep
    # precedence after its imports, including the full 798-arm budget module.
    sys.path.insert(0, str(PACKAGE))
    return identity, original_guard


def proof():
    identity, guard = activate()
    from transport_activation import proof as transport_proof
    transport = transport_proof()
    paths = [PACKAGE / name for name in (
        'identity_contract.py', 'guard_activation.py', 'nvml_capture.py', 'http_runner.py',
        'CONTRACT.json')]
    paths.append(LEGACY / 'original_guard.py')
    pins = {str(path): sha(path) for path in paths}
    pins.update(transport['source_pins'])
    return {'status': 'OWNED_RESOURCE_CONTRACT_ACTUALLY_BOUND',
        'identity_module_path': str(Path(identity.__file__).resolve()),
        'identity_module_sha256': IDENTITY_SHA,
        'original_guard_module_path': str(Path(guard.__file__).resolve()),
        'original_guard_module_sha256': GUARD_SHA,
        'gpu_identity_function_is_owned': guard.gpu_identity is identity.gpu_identity,
        'listener_identity_function_is_owned': guard.listener_identity is identity.listener_identity,
        'process_limit_MiB': identity.PROCESS_NVML_MIB,
        'device_limit_MiB': identity.DEVICE_NVML_MIB,
        'model_cgroup_bytes': identity.MODEL_CGROUP_BYTES,
        'transport_identity': transport,
        'source_pins': pins}


def require_binding(binding):
    path = Path(binding['startup_path'])
    if sha(path) != binding['startup_sha256']:
        raise ValueError('startup receipt bytes changed')
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError('bounded startup receipt required')
    receipt = json.loads(path.read_text())
    value = proof()
    if (receipt.get('resource_guard_identity') != value
            or any(binding['source_files'].get(path) != digest
                   for path, digest in value['source_pins'].items())):
        raise ValueError('actual startup resource guard binding missing/different')
    return value
