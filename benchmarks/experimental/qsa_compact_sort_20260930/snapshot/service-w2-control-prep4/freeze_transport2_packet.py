"""Freeze source-only native EOF correction; preserve the actual failed wire1."""
import json
from pathlib import Path
import shutil
import sys

from io_tools import save, sha, tree_bytes
from frozen import PINS, require

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
OUT = BASE / 'service-w2-transport-prep2'
NAMES = ['frozen.py', 'io_tools.py', 'transport_worker.py',
         'transport_deadline.py', 'wire_preflight.py', 'tests/test_native_eof.py']


def main():
    for name in PINS:
        require(name)
    OUT.mkdir(mode=0o700)
    for name in NAMES:
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write((ROOT / name).read_bytes())
    for name in ('cpu-native-eof-attempt1', 'cpu-native-eof-attempt2',
                 'wire1-transport-attempt1', 'transport-EOF-defect1'):
        shutil.copytree(ROOT / 'evidence' / name, OUT / 'evidence' / name)
    runtime = json.loads((BASE / 'service-w2-prep/deployed-http-source-receipt.json').read_text())
    native_pins = {}
    for name, module in (('requests_models.py', 'requests.models'),
                         ('urllib3_response.py', 'urllib3.response')):
        row = next(row for row in runtime['files'] if row['module'] == module)
        source = BASE / 'service-w2-prep/source' / name
        if sha(source) != row['sha256']:
            raise ValueError('actual native HTTP source snapshot mismatch')
        (OUT / 'source').mkdir(exist_ok=True)
        with (OUT / 'source' / name).open('xb') as stream:
            stream.write(source.read_bytes())
        native_pins[module] = row
    attempt = ROOT / 'evidence/wire1-transport-attempt1/host-attempt1'
    failure = json.loads((attempt / 'cases/normal/FAILURE.json').read_text())
    process = json.loads((attempt / 'cases/normal/process-exit.json').read_text())
    if (failure['error'] != "OSError(9, 'Bad file descriptor')"
            or process['actual_child_exit'] != 1 or process['process_reaped'] is not True
            or (attempt / 'original-identity.exit').read_text().strip() != '0'
            or (ROOT / 'evidence/cpu-native-eof-attempt2/exit').read_text().strip() != '0'):
        raise ValueError('actual failure or CPU correction evidence differs')
    save(OUT / 'WIRE1-RECEIPT.json', {
        'status': 'ACTUAL_SYNTHETIC_WIRE_FAIL_CLOSED_NOT_MODEL_FAILURE',
        'actual_synthetic_requests': 1, 'model_requests': 0,
        'completed_cases': [], 'cancel_and_trickle': 'NOT_EXECUTED',
        'source_manifest_sha256': sha(BASE / 'service-w2-wire-prep1/manifest.json'),
        'source_plan_sha256': sha(BASE / 'service-w2-wire-prep1/PLAN.json'),
        'exits': {'dependency_hash': 0, 'child': 1, 'systemd_wait': 1,
                  'foreground_SSH': 1, 'download': 0, 'original_identity_cmp': 0},
        'first_error': failure, 'process_exit': process,
        'unit_cleanup': (attempt / 'unit-cleanup.txt').read_text(),
        'unit_resource_properties_after_GC': (attempt / 'unit-final.txt').read_text(),
        'cgroup_RAM_MemoryPeak': 'Unavailable after systemd GC; raw journal rounded1.5M retained, not a zero peak or GPU memory claim.',
        'normal_child_peak_RSS_bytes': json.loads(
            (attempt / 'cases/normal/native-close.json').read_text())['peak_RSS_bytes'],
        'artifact_bytes': tree_bytes(attempt),
        'closed_semantics_limit': 'Initial actual socket FD4/peer39411, native server EOF and EBADF were observed. raw.closed at failure was not exported; pinned source plus method-body CPU fixture establishes why buffered lines can remain after native close, without inventing that missing runtime flag.'},
        256 * 1024)
    save(OUT / 'contract.json', {
        'schema': 1, 'status': 'SOURCE_CPU_NATIVE_EOF_CORRECTION_NOT_WIRE_PASS',
        'immutable_previous': {'transport1_manifest': sha(BASE / 'service-w2-transport-prep1/manifest.json'),
                               'wire1_manifest': sha(BASE / 'service-w2-wire-prep1/manifest.json')},
        'external_W2_pins': PINS, 'native_HTTP_pins': native_pins,
        'runtime': {'executable': sys.executable, 'Python': sys.version,
                    'profile': 'uv managed3.12 --no-project --no-python-downloads'},
        'minimal_runtime_delta': 'Before next(iterator), update timeout only on same initial live FD. FD=-1 is allowed only when actual urllib3 raw.closed is exactlyTrue. Other errors/changedFD/unknownnegative/falseclosed STOP; parent absolute deadline/kill/reap and frozen completion parser remain.',
        'source_evidence': {'requests_models.py': 'iter_lines990-1028 yields all lines from a previously read chunk',
                            'urllib3_response.py': '_raw_read1007-1059 closes native fp at EOF; stream1238-1273 allows decoded buffer afterfpclose; closed1298-1308 uses native HTTPResponse.isclosed'},
        'CPU_tests': {'attempt1': {'exit': 1, 'tests': 2,
            'error': 'test-local class variableNameError, source preserved; EOF method check not reached'},
            'attempt2': {'exit': 0, 'tests': 2,
                'scope': 'Exact pinned iter_lines/closed AST bodies with explicit underlying reader/socket stubs; oldEBADF reproduced/new buffer path and strict negative/error propagation checked'}},
        'live_tests_not_run': ['Corrected actual requests/urllib3 EOF',
                              'Actual two-positive wire close',
                              'Actual trickle/no-newline absolute deadline'],
        'new_host_unit_authorized': False,
        'no_float_parser_arrival_matrix_or_sampling_change': True,
        'no_generation_model_or_GPU_operation': True}, 1024**2)
    entries = {str(path.relative_to(OUT)): {'bytes': path.stat().st_size,
        'sha256': sha(path)} for path in sorted(OUT.rglob('*')) if path.is_file()}
    save(OUT / 'manifest.json', {'files': entries, 'count': len(entries),
        'bytes': sum(row['bytes'] for row in entries.values())}, 1024**2)
    print(json.dumps({'root': str(OUT), 'count': len(entries),
        'manifest_sha256': sha(OUT / 'manifest.json'),
        'contract_sha256': sha(OUT / 'contract.json'),
        'wire1_receipt_sha256': sha(OUT / 'WIRE1-RECEIPT.json')}))


if __name__ == '__main__':
    main()
