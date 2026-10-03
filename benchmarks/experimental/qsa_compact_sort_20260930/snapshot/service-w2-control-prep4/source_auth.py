"""Literal prerequisite authentication, repeated before staging and launch."""
import json
from pathlib import Path

from common import BASE, PACKAGE, read, sha
from frozen import BASE as LOCAL_BASE, PINS as W2_PINS
from service_binding import SERVICE_MANIFEST, OFF_SOURCE

PRIOR = {
    'review/W2-PLAN3-FAILURE-CLOSURE-REVIEW.json': ('19c75b611bcde8bfd72e64ec4d3ed3f29598c0a310b7412a30a61285edab78e4', 'PASS_PARENT_W2_RUN3_FAILURE_AND_ARCHIVE_REVIEW_ORIGINAL_RESTORED'),
    'review/W2-PLAN2-FAILURE-CLOSURE-REVIEW.json': ('e1d2d511542196ad1a35cbea9881db61d04d37690e093305a01e5aa73cc02929', 'PASS_PARENT_W2_RUN2_FAILURE_AND_ARCHIVE_REVIEW_ORIGINAL_RESTORED'),
    'review/W2-GUARD2-SOURCE-REVIEW.json': ('88d7cf6c92276c87b29356d0693176816c4eba035a6f53ebe39f5613740ff4be', 'PASS_PARENT_GUARD2_FINITE_SPARSE_E7_AND_MANAGED_CPU_CORRECTION'),
    'review/W2-PLAN1-FAILURE-REVIEW.json': ('02b173999b0c6edf2872b8c31a049b192db5e09afbd6dda0812d5f5d49fe5aa4', 'PARENT_CONFIRMED_A0_GUARD_SPARSE_COUNTER_FAILURE_NOT_CANDIDATE_FAILURE'),
    'review/W2-PLAN1-RESTORATION-REVIEW.json': ('6ba636d85ce89f0336067c2c7a86cef1e563f8694497c637993d53a0e2fa4574', 'PASS_ORIGINAL_RESTORED_AFTER_W2_PLAN1'),
    'review/NUMERIC-B1-BOTH-REVIEW.json': ('c262b67af93b942233dd9ea40d6492e1315851b02da59285c9e66a098321cbdd', 'PASS_PARENT_FINITE_B_EXACT_AGAINST_BOTH_ORIGINALS'),
    'review/NUMERIC-B1-RESTORATION-REVIEW.json': ('c5e7dda686d2c057ee1392411415717ddcdbe8fde5119f94988b9dcb4560de25', 'PASS_ORIGINAL_RESTORED_AFTER_NUMERIC_B1'),
    'review/W1-RESULT-REVIEW.json': ('1450c7866d1b58661c0df0180e4415351d32d825e308e11db1eb71c19a3f3cb4', 'PASS_OFFLINE_W1_SERVICE_UNVERIFIED'),
    'review/W2-TRANSPORT2-SOURCE-REVIEW.json': ('4e5182e6406f0a0faaf84309c44cfa26bf35c4452d8931a52c73546249d045b9', None),
    'review/W2-GUARD1-SOURCE-REVIEW.json': ('d12e814b1ebf4bc50add5b259b4e2e25a3eb0c3de74374943816897bf651068a', None),
    'review/W2-WIRE2-ACTUAL-REVIEW.json': ('a2d760edc96cd6e67a930624231671a52581681448f78fab1a7e456351d4297c', None),
}
NATIVE = {'service-numeric-candidate-B-control-prep1/source/run3_control.production.py':
          '089f362e32253c2d2e3bcd136e51235f8dd95a4c69dac548493fe18854ee520e'}
WIRE_MANIFEST = '35bab48e916d028d278c9680811f5a4f9941dadca232a8666f0d1b71846279d6'
GUARD2_MANIFEST = '2c3cef0e9ea1c8ab6b1f78d3052049e33db3f079aa087f64a10ce51375077977'
CONTROLLER1_MANIFEST = '2558256a83a316d37991d0170c4916ec7a910cdec5dac2701d95b84cd00aa912'
CONTROLLER2_MANIFEST = '3ccf2cd74191afabf6774a866598196fa2d339584ecf29ce3700820049b56060'
CONTROLLER3_MANIFEST = '59cc6850c6f35e66db4a7fc48aa5752b8dda262e8efc4e7bbd6702b4153a523b'
GUARD2_RUNTIME_CORRECTION_MANIFEST = '4a27aa0618216a59bac3ed077cba5fca94b67ccddd3a1b1701b71bb19be837ba'


def guard2_pins():
    root = LOCAL_BASE / 'service-w2-guard-prep2'
    path = root / 'manifest.json'
    if sha(path) != GUARD2_MANIFEST:
        raise ValueError('literal corrected guard manifest changed')
    value = read(path)
    result = {str(BASE / root.name / 'manifest.json'): GUARD2_MANIFEST}
    seen = set()
    for row in value['files']:
        if set(row) != {'path', 'size', 'sha256'}:
            raise ValueError('corrected guard manifest entry schema changed')
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen:
            raise ValueError('corrected guard manifest foreign/duplicate path')
        seen.add(str(relative))
        source = root / relative
        if (source.is_symlink() or not source.resolve().is_relative_to(root.resolve())
                or source.stat().st_size != row['size'] or sha(source) != row['sha256']):
            raise ValueError('corrected guard payload changed: ' + str(source))
        result[str(BASE / root.name / relative)] = row['sha256']
    if len(seen) != value['file_count']:
        raise ValueError('corrected guard manifest count differs')
    for name in ('original_guard.py', 'e7_counters.py'):
        if sha(PACKAGE / name) != sha(root / name):
            raise ValueError('controller corrected guard differs from frozen packet')
    return result


def entries(manifest):
    rows = manifest['files']
    if isinstance(rows, dict):
        rows = [dict(row, path=name) for name, row in rows.items()]
    seen = set()
    for row in rows:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen:
            raise ValueError('manifest foreign/duplicate path')
        seen.add(str(relative))
        yield relative, row['sha256'], row['bytes']


def manifest_pins(name, digest):
    root = LOCAL_BASE / name
    path = root / 'manifest.json'
    if sha(path) != digest:
        raise ValueError('literal manifest pin differs: ' + name)
    result = {str(BASE / name / 'manifest.json'): digest}
    for relative, expected, size in entries(read(path, 4 * 1024**2)):
        source = root / relative
        if (source.is_symlink() or not source.resolve().is_relative_to(root.resolve())
                or source.stat().st_size != size or sha(source) != expected):
            raise ValueError('manifest payload changed: ' + str(source))
        result[str(BASE / name / relative)] = expected
    return result


def authenticate():
    pins = guard2_pins()
    pins.update(manifest_pins('service-w2-control-prep1', CONTROLLER1_MANIFEST))
    pins.update(manifest_pins('service-w2-control-prep2', CONTROLLER2_MANIFEST))
    pins.update(manifest_pins('service-w2-control-prep3', CONTROLLER3_MANIFEST))
    pins.update(manifest_pins('service-w2-guard-prep2-runtime-correction1',
                              GUARD2_RUNTIME_CORRECTION_MANIFEST))
    for name, (digest, status) in PRIOR.items():
        source = LOCAL_BASE / name
        if sha(source) != digest or (status and read(source, 4 * 1024**2)['status'] != status):
            raise ValueError('literal prior review pin/status differs: ' + name)
        pins[str(BASE / name)] = digest
    for name, digest in NATIVE.items():
        if sha(LOCAL_BASE / name) != digest:
            raise ValueError('operational source differs')
        pins[str(BASE / name)] = digest
    pins.update(manifest_pins('service-implementation-retry2', SERVICE_MANIFEST))
    pins.update(manifest_pins('service-w2-prep', W2_PINS['manifest.json']))
    pins.update(manifest_pins('service-w2-wire-prep2', WIRE_MANIFEST))
    off = 'numeric-run7/control/configs/A0.service.json'
    if sha(LOCAL_BASE / off) != OFF_SOURCE:
        raise ValueError('exact pure-off template changed')
    pins[str(BASE / off)] = OFF_SOURCE
    service_config = read(LOCAL_BASE / off)
    for row in service_config['source_pins'].values():
        if sha(row['path']) != row['sha256']:
            raise ValueError('active native QSA/dispatch/stage/API source changed')
        pins[row['path']] = row['sha256']
    for name in ('candidate_binary', 'original_binary'):
        path, digest = service_config[name], service_config[name + '_sha256']
        if sha(path) != digest:
            raise ValueError('unchanged candidate/original library differs')
        pins[path] = digest
    runtime = read(LOCAL_BASE / 'service-w2-wire-prep2/runtime-pins.json')
    for row in runtime['files']:
        path = Path(row['path'])
        if sha(path) != row['sha256']:
            raise ValueError('actual requests/urllib3 deployed source changed')
        pins[str(path)] = row['sha256']
    return pins


def controller_pins():
    path = PACKAGE / 'manifest.json'
    return manifest_pins(PACKAGE.name, sha(path))


def wire_proof():
    # A plan cannot be prepared until the separately controlled corrective
    # CPU wire execution really passes. No substitute/mocked proof accepted.
    root = BASE / 'service-w2-wire-prep2'
    path = root / 'attempt1/cases/RESULT.json'
    if sha(path) != '78f0d7e35ff8cc173bdfa0990126c35d7e0f58b32bcc26014e082d2d39acaec8':
        raise ValueError('the one reviewed actual corrective wire result changed')
    value = read(path)
    if (value['status'] != 'PASS_REAL_REQUESTS_SOCKET_SYNTHETIC_NOT_MODEL_W2'
            or value['synthetic_HTTP_requests'] != 3 or value['model_HTTP_requests'] != 0
            or value['server_and_request_processes_ended'] is not True
            or [r['case'] for r in value['cases']] != ['normal', 'cancel', 'trickle-no-newline']
            or any(r['reaped'] is not True for r in value['cases'])
            or [r['actual_child_exit'] for r in value['cases'][:2]] != [0, 0]
            or value['cases'][2]['actual_child_exit'] not in (-15, -9)):
        raise ValueError('real native corrective wire proof incomplete')
    normal = value['cases'][0]['native_socket_descriptor_observations']
    if any(type(normal[k]) is not int or normal[k] <= 0 for k in
           ('live_socket_timeout_updates', 'confirmed_native_closed_buffer_iterations')):
        raise ValueError('wire EOF branch proof is vacuous')
    required = {str(path): sha(path)}
    actual_review = read(BASE / 'review/W2-WIRE2-ACTUAL-REVIEW.json')
    for filename, digest in actual_review['sha256'].items():
        if sha(filename) != digest:
            raise ValueError('parent reviewed actual/source wire input differs')
        required[filename] = digest
    for name in ('child.exit', 'systemd-wait.exit', 'original-identity.exit'):
        receipt = root / 'attempt1' / name
        if receipt.read_text().strip() != '0':
            raise ValueError('wire child/unit/original identity did not close')
        required[str(receipt)] = sha(receipt)
    cleanup = root / 'attempt1/unit-cleanup.txt'
    properties = dict(row.split('=', 1) for row in cleanup.read_text().splitlines() if '=' in row)
    if (properties.get('ActiveState') != 'inactive' or properties.get('MainPID') != '0'
            or properties.get('LoadState') != 'not-found'
            or properties.get('ControlGroup') != ''):
        raise ValueError('wire owned unit/cgroup not fully closed')
    required[str(cleanup)] = sha(cleanup)
    for row in value['cases']:
        directory = root / 'attempt1/cases' / row['case']
        for name, key in (('process-exit.json', 'process_exit_sha256'),
                          ('actual-socket-adapter.json', 'actual_socket_sha256')):
            receipt = directory / name
            if sha(receipt) != row[key]:
                raise ValueError('wire actual process/socket receipt changed')
            required[str(receipt)] = row[key]
    return required
