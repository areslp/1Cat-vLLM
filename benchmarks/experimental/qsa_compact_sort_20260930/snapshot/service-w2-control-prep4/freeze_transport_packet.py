"""Create one immutable CPU-reviewed transport packet, not a live admission."""
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

from frozen import PINS, require


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'service-w2-transport-prep1'
SOURCES = ['frozen.py', 'io_tools.py', 'transport_deadline.py',
           'transport_worker.py', 'tests/stub_child.py',
           'tests/test_transport_process.py', 'tests/test_collector.py']


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('xb') as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2) + '\n').encode())


def main():
    for name in PINS:
        require(name)
    OUT.mkdir(mode=0o700)
    for name in SOURCES:
        destination = OUT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write((ROOT / name).read_bytes())
    for name in ('cpu-transport-attempt1', 'cpu-collector-attempt1'):
        origin = ROOT / 'evidence' / name
        if (origin / 'exit').read_text().strip() != '0':
            raise ValueError('executed CPU evidence not PASS: ' + name)
        shutil.copytree(origin, OUT / 'evidence' / name)
    write(OUT / 'contract.json', {
        'schema': 1, 'status': 'CPU_TRANSPORT_PACKET_NOT_LIVE_W2',
        'ownership': 'service-w2-transport-prep1 only; create-only frozen snapshot',
        'external_pins': PINS,
        'runtime': {'host_python': '/home/l/work/1Cat-vLLM/.venv/bin/python',
                    'host_expected_python': '3.12.13',
                    'local_executed_python': sys.executable,
                    'local_python_version': platform.python_version(),
                    'local_profile': 'uv managed 3.12; no project or downloads'},
        'checks': {'actual_OS_subprocess_tests': 5, 'frozen_parser_stub_tests': 4,
                   'exits': [0, 0], 'actual_network_executed': False,
                   'GPU_or_model_operations': 0},
        'bounds': {'ready_preparation_seconds': 30,
                   'request_deadline_seconds': 120,
                   'max_live_request_children': 8,
                   'term_to_kill_seconds': 1, 'owner_cleanup_seconds': 4,
                   'received_bytes_per_request': 16 * 1024**2,
                   'retained_request_tree_bytes': 34 * 1024**2,
                   'line_bytes': 512 * 1024, 'line_count': 4096,
                   'stderr_bytes': 65536,
                   'aggregate_window_measured_cap_bytes': 8 * 1024**3},
        'timing': 'All child imports/READY precede release; mixed release uses first positive event. Native SSE/perf38 definitions preserved. Actual submission/release/close/reap times retained.',
        'cleanup': 'Timeout is failure with owned process-group TERM/KILL and actual reap. Cleanup failure is preserved and requires outer cgroup drain; successful parser result alone never proves termination or engine cancellation.',
        'unverified': [
            'Actual deployed requests/urllib3 adapter socket access and wire path',
            'Native engine cancel/queue/terminal publication after child close',
            'Pure off/on startup identity and resource gate',
            'Aggregate monitor interval and in-flight overshoot integration',
            'Whole matrix, stability, matched-group analysis and live A/B/A'],
        'parent_decisions': {'outer_memory_bytes': 8 * 1024**3,
                             'HTTP_group_memory_bytes': 2 * 1024**3,
                             'aggregate_cap_is_filesystem_quota': False,
                             'theoretical_4066_request_worst_case_supported': False,
                             'no_delete_or_retry_to_fit_cap': True},
    })
    entries = {str(path.relative_to(OUT)): {
        'sha256': digest(path), 'bytes': path.stat().st_size}
        for path in sorted(OUT.rglob('*')) if path.is_file()}
    write(OUT / 'manifest.json', {'schema': 1, 'files': entries,
          'count': len(entries), 'bytes': sum(row['bytes'] for row in entries.values())})
    print(json.dumps({'root': str(OUT), 'files_without_manifest': len(entries),
                      'manifest_sha256': digest(OUT / 'manifest.json'),
                      'contract_sha256': digest(OUT / 'contract.json')}))


if __name__ == '__main__':
    main()
