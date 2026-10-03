"""Create-only review checkpoint. No host/service/network operation."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

from frozen import PINS, require
from io_tools import save, sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'service-w2-guard-prep1'
FILES = ['frozen.py', 'io_tools.py', 'identity_contract.py', 'original_guard.py',
         'service_binding.py', 'startup.py', 'tests/test_startup_guard.py',
         'tests/test_service_binding.py', 'tests/test_cancel_policy.py',
         'source/vllm/v1/engine/output_processor.py',
         'source/vllm/v1/metrics/stats.py', 'source/vllm/v1/metrics/loggers.py',
         'evidence/NATIVE-CANCEL-SOURCE-FACTS.json',
         'evidence/prepare_source_cancel_note.py']
EXITS = ('cpu-startup-guard-attempt1', 'cpu-service-schema-attempt1',
         'cpu-cancel-policy-attempt1')


def main():
    for name in PINS:
        require(name)
    OUT.mkdir(mode=0o700)
    for name in FILES:
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write((ROOT / name).read_bytes())
    for name in EXITS:
        origin = ROOT / 'evidence' / name
        if (origin / 'exit').read_text().strip() != '0':
            raise ValueError('CPU check did not PASS: ' + name)
        shutil.copytree(origin, OUT / 'evidence' / name)
    native = json.loads((ROOT / 'evidence/NATIVE-CANCEL-SOURCE-FACTS.json').read_text())
    save(OUT / 'contract.json', {
        'schema': 1, 'status': 'SOURCE_CPU_CHECKPOINT_NOT_LIVE_W2',
        'external_W2_pins': PINS, 'source_cancel_facts_sha256':
            sha(ROOT / 'evidence/NATIVE-CANCEL-SOURCE-FACTS.json'),
        'fixed_service_manifest_sha256': __import__('service_binding').SERVICE_MANIFEST,
        'runtime': {'local_executable': sys.executable, 'local_python': sys.version,
            'uv': subprocess.check_output(['/Users/l/.local/bin/uv', '--version'],
                                         text=True).strip(),
            'profile': 'uv managed 3.12 --no-project --no-python-downloads',
            'host_profile': '/home/l/work/1Cat-vLLM/.venv/bin/python'},
        'checks': {'startup_binding_CPU_tests': 4, 'actual_saved_capture_schema_tests': 5,
            'client_cancel_policy_tests': 3, 'raw_exits': [0, 0, 0],
            'test_scope': 'CPU tests and 8 historical ready JSONs; OS/telemetry adapters are stubs in wrapper checks; no live W2 startup PASS'},
        'cancellation': {'original_contract_unchanged': True,
            'two_positive_cut_native_close_reap_and_independent_queue_drain': True,
            'native_length_counter_6_to_8': 'Source-proven natural-finish race; actual full values retained, no fabricated abort +2',
            'engine_cancel_ack': 'UNVERIFIED', 'engine_cancelled_count': None,
            'cancellation_performance_benefit_claim': False},
        'timing': 'Original health/cache/full-route counters only outside timed group release/completion; failed native publication observations retained.',
        'resource_bounds': {'model_cgroup_bytes': 120 * 1024**3,
            'model_swap_bytes': 0, 'sampled_process_NVML_MiB': 32000,
            'sampled_device_NVML_MiB': 32384,
            'startup_endpoint_bytes': 2 * 1024**2,
            'startup_journal_measured_bytes': 8 * 1024**2,
            'publication_seconds': 15, 'publication_trace_records': 64},
        'startup_binding': 'Actual same API PID/starttime/invocation, four worker rank/PID/starttime/UUID/cgroup/library/capture bytes, preserved source/stage/consumer bindings, actual mature readiness+capacity.',
        'raw_flags': 'Worker raw selected env observations are read and retained. Their bool may be false; loaded native source/config/capture binding establishes exact approved mode separately.',
        'not_executed': ['Actual pure W2 startup/systemd/proc/NVML/HTTP health gate',
            'Actual requests/urllib3 socket synthetic wire',
            'Live group route/cache/cancellation publication',
            'W2 matrix/stability/CI/controller/model window'],
        'next': 'Finite controller with complete staged-copy/source bindings, unique stop receipts, best-effort unconditional original restoration before error aggregation; synthetic wire in its own approved CPU slot.'
    }, 1024**2)
    entries = {str(path.relative_to(OUT)): {'bytes': path.stat().st_size,
               'sha256': sha(path)} for path in sorted(OUT.rglob('*')) if path.is_file()}
    save(OUT / 'manifest.json', {'schema': 1, 'files': entries, 'count': len(entries),
        'bytes': sum(row['bytes'] for row in entries.values())}, 1024**2)
    print(json.dumps({'root': str(OUT), 'files_without_manifest': len(entries),
        'manifest_sha256': sha(OUT / 'manifest.json'),
        'contract_sha256': sha(OUT / 'contract.json')}))


if __name__ == '__main__':
    main()
