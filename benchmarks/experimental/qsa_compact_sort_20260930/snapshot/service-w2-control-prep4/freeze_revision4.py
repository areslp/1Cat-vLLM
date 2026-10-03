"""Create-only source packet; no host stage, model, HTTP or test rerun."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
OLD = BASE / 'service-w2-control-prep3'
REMOTE = Path('/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930')
PACKAGE = REMOTE / ROOT.name
WINDOW = REMOTE / 'service-w2-run4'
OLD_MANIFEST = '59cc6850c6f35e66db4a7fc48aa5752b8dda262e8efc4e7bbd6702b4153a523b'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def publish(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def replace_paths(value):
    if isinstance(value, dict):
        return {key: replace_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_paths(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(REMOTE / OLD.name), str(PACKAGE)).replace(
            str(REMOTE / 'service-w2-run3'), str(WINDOW))
    return value


def main():
    for name in ('CPU-RECEIPT.json', 'contract.json', 'command-contract.json', 'manifest.json'):
        if (ROOT / name).exists():
            raise FileExistsError('immutable output exists: ' + name)
    if sha(OLD / 'manifest.json') != OLD_MANIFEST:
        raise ValueError('frozen previous controller changed')
    from common import MODELS, HTTP, UNITS, TIMER, OUTER
    from execution_budget import BUDGETS, ORIGINAL_BUDGETS, binding, effective_budgets
    import source_auth
    effective_budgets()
    initial = json.loads((ROOT / 'evidence/initial-copy-map4.json').read_text())
    rows, diff, counts = [], [], {}
    explicit = {'matrix.py', 'ops.py', 'plan.py', 'prepare_plan.py', 'source_auth.py'}
    for destination, original in sorted(initial['copied_files'].items()):
        source, current = OLD / original['path'], ROOT / destination
        if sha(source) != original['sha256']:
            raise ValueError('old source changed after clone')
        if current.read_bytes() == source.read_bytes():
            kind = 'HISTORICAL_RELOCATED_EXACT' if destination != original['path'] else 'EXACT'
        elif destination in explicit:
            kind = 'EXPLICIT_TIME_CONSUMER_OR_PREREQUISITE_BINDING'
        else:
            text = source.read_text()
            for old, new in initial['identity_replacements']:
                text = text.replace(old, new)
            shell_budget = {
                'control/launch_outer.sh': ('RuntimeMaxSec=12900s', 'RuntimeMaxSec=18300s'),
                'control/run_and_restore.sh': ('timeout --kill-after=20 9900 bash', 'timeout --kill-after=20 15300 bash'),
                'control/offline_window.sh': ('--on-active=167m', '--on-active=257m'),
            }
            if destination in shell_budget:
                before, after = shell_budget[destination]
                text = text.replace(before, after)
            if current.read_text() != text:
                raise ValueError('unexplained change: ' + destination)
            kind = ('RUN4_IDENTITY_PLUS_REGISTERED_SHELL_TIME' if destination in shell_budget
                    else 'RUN4_IDENTITY_ONLY')
        counts[kind] = counts.get(kind, 0) + 1
        rows.append({'old_path': original['path'], 'new_path': destination,
                     'old_sha256': original['sha256'], 'new_sha256': sha(current),
                     'comparison': kind})
        if current.read_bytes() != source.read_bytes():
            diff.extend(difflib.unified_diff(source.read_text().splitlines(True),
                current.read_text().splitlines(True), fromfile=str(source), tofile=str(current)))
    with (ROOT / 'evidence/runtime-and-assembly4.diff').open('x') as stream:
        stream.write(''.join(diff))
    publish('evidence/old450-to-new-map.json', {'old_manifest_sha256': OLD_MANIFEST,
        'counts': counts, 'files': rows, 'old_payload_count': len(rows)})
    # Explicit active set, not historical saved generators or executed-source.
    active_python = [ROOT / name for name in (
        'analysis.py', 'artifact_budget.py', 'artifact_check.py', 'common.py',
        'e7_counters.py', 'execution_budget.py', 'flow.py', 'fresh_identity.py',
        'frozen.py', 'identity_contract.py', 'io_tools.py', 'matrix.py', 'ops.py',
        'original_guard.py', 'plan.py', 'prepare_plan.py', 'request_groups.py',
        'restoration_dependencies.py', 'run_w2.py', 'service_binding.py',
        'source_auth.py', 'stability.py', 'stage_bindings.py', 'stage_window.py',
        'startup.py', 'transport_deadline.py', 'transport_worker.py')]
    active_python += list((ROOT / 'control').glob('*.py'))
    active_python += list((ROOT / 'restoration').rglob('*.py'))
    active_shell = [*(ROOT / 'control').glob('*.sh'), ROOT / 'restoration/verify_original.sh']
    for path in active_python:
        ast.parse(path.read_text(), filename=str(path))
    for path in active_shell:
        subprocess.run(['/bin/bash', '-n', str(path)], timeout=5, check=True)
    for path in active_python + active_shell:
        text = path.read_text()
        if any(token in text for token in (str(REMOTE / OLD.name),
                 str(REMOTE / 'service-w2-run3'), 'step58-w2-a03.service',
                 'step58-w2-b3.service', 'step58-w2-a23.service')):
            # source_auth pins the complete old package as historical proof.
            if path.name != 'source_auth.py':
                raise ValueError('old actual execution identity: ' + str(path))
    prior = {}
    for name, (digest, status) in source_auth.PRIOR.items():
        path = BASE / name
        if sha(path) != digest or (status and json.loads(path.read_text())['status'] != status):
            raise ValueError('literal prerequisite review mismatch: ' + name)
        prior[str(REMOTE / name)] = digest
    attempts = []
    for number, expected_tests, expected_exit in ((1, 7, 1), (2, 1, 0)):
        directory = ROOT / f'evidence/cpu-revision4-attempt{number}'
        result = json.loads((directory / 'RESULT.json').read_text())
        if (result['raw_exit'] != expected_exit or result['testsRun'] != expected_tests
                or int((directory / 'exit').read_text()) != expected_exit
                or result['Torch_imported'] or result['VLLM_imported']
                or not result['runtime']['python'].startswith('3.12.13 ')
                or result['runtime']['uv'].split()[:2] != ['uv', '0.11.16']):
            raise ValueError('same-process managed targeted CPU evidence differs')
        if number == 1 and (result['failures'] != 0 or result['errors'] != 1):
            raise ValueError('original missing-fixture failure evidence differs')
        if number == 2 and (result['failures'] or result['errors']):
            raise ValueError('single corrective model-consumer check failed')
        for name, digest in result['runtime']['source_sha256'].items():
            if sha(directory / 'executed-source' / name) != digest:
                raise ValueError('CPU executed bytes differ')
            if name not in ('tests/test_time_revision4.py', 'run_revision4_cpu.py') and sha(ROOT / name) != digest:
                raise ValueError('runtime changed after targeted check: ' + name)
        # Closed transport must be preserved rather than self-pinning open logs.
        outside = BASE / 'service-w2-control-prep4-transport'
        for suffix in ('stdout', 'stderr', 'exit'):
            source = outside / f'cpu-attempt{number}.{suffix}'
            target = directory / 'transport' / source.name
            target.parent.mkdir(exist_ok=True)
            with target.open('xb') as stream:
                stream.write(source.read_bytes())
        shell = outside / ('run_cpu4.sh' if number == 1 else 'run_cpu4_attempt2.sh')
        with (directory / 'transport' / shell.name).open('xb') as stream:
            stream.write(shell.read_bytes())
        attempts.append({'attempt': number, 'testsRun': expected_tests,
            'actual_exit': expected_exit, 'RESULT_sha256': sha(directory / 'RESULT.json'),
            'runtime': result['runtime']})
    publish('CPU-RECEIPT.json', {'status': 'SEVEN_TARGETED_CHECKS_CLOSED_SOURCE_CPU_NOT_LIVE_W2',
        'attempts': attempts, 'checks': 'attempt1 sixPASS/one missing synthetic config fixture error; attempt2 only affected model-start test PASS',
        'successful_distinct_checks': 7, 'no_old_passed_suite_rerun': True,
        'freeze_runtime': {'executable': sys.executable, 'python': sys.version,
            'uv': subprocess.check_output(['/Users/l/.local/bin/uv', '--version'], text=True).strip()},
        'static': {'AST': len(active_python), 'bash': len(active_shell),
                  'active_files_sha256': {str(p.relative_to(ROOT)): sha(p) for p in active_python + active_shell}},
        'literal_prerequisites': prior,
        'limitations': 'Actual native consumers with explicit no-network/group/ready/systemd mocks; no new W2/model/GPU proof; empirical time allowance is not worst-case completion guarantee.'})
    contract = replace_paths(json.loads((OLD / 'contract.json').read_text()))
    contract['budgets_seconds'] = BUDGETS
    amendment = json.loads((ROOT / 'execution-budget-revision.json').read_text())
    contract['execution_budget_amendment'] = binding()
    contract['original_budgets_seconds_unchanged'] = ORIGINAL_BUDGETS
    contract['time_audit'] = amendment['time_audit']
    contract['revision'] = {'change': 'Four unique execution time replacements plus exact derived model/HTTP lifetimes; original semantic W2 contract and full matrix unchanged',
        'historical_controller3_manifest': OLD_MANIFEST,
        'parent_failure_restore_closure_review_sha256': prior[str(REMOTE / 'review/W2-PLAN3-FAILURE-CLOSURE-REVIEW.json')],
        'partial_arm_reused': False, 'empirical_estimate_limitations': amendment['estimate'],
        'source_finding': 'evidence/budget-native-chain-prefreeze1/FINDING.json',
        'CPU_first_failure': 'actual config SHA read in synthetic ops.start fixture lacked its config file; original attempt retained',
        'CPU_corrective_check': 'only the affected one model-start test; runtime unchanged'}
    publish('contract.json', contract)
    command = replace_paths(json.loads((OLD / 'command-contract.json').read_text()))
    command['commands']['launch']['timeout_s'] = BUDGETS['outer']
    command['execution_budget_amendment'] = binding()
    command['budgets_seconds'] = BUDGETS
    future = {str(WINDOW / p.relative_to(ROOT)) for p in
        [*(ROOT / 'control').glob('*'), *(ROOT / 'restoration').rglob('*')] if p.is_file()}
    future.add(str(WINDOW / 'baseline/snapshot.py'))
    for value in command['commands'].values():
        for argument in value['argv']:
            if argument.startswith(str(PACKAGE) + '/'):
                if not (ROOT / Path(argument).relative_to(PACKAGE)).is_file():
                    raise ValueError('canonical package CLI absent')
            elif argument.startswith(str(WINDOW) + '/') and argument.endswith(('.py', '.sh')):
                if argument not in future:
                    raise ValueError('canonical staged CLI absent')
    files = []
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if '__pycache__' in relative.parts or path.suffix == '.pyc':
            continue
        if path.is_symlink():
            raise ValueError('unexpected source/test symlink: ' + str(relative))
        if path.is_file() and str(relative) not in ('command-contract.json', 'manifest.json'):
            files.append(str(relative))
    command['upload_files'] = sorted(files + ['command-contract.json', 'manifest.json'])
    publish('command-contract.json', command)
    payloads = [{'path': name, 'bytes': (ROOT / name).stat().st_size, 'sha256': sha(ROOT / name)}
                for name in command['upload_files'] if name != 'manifest.json']
    publish('manifest.json', {'schema': 'step58-pure-w2-controller-source-manifest-v1',
        'status': 'IMMUTABLE_SOURCE_CPU_PACKET_NOT_LIVE_W2', 'files': payloads,
        'payload_file_count': len(payloads), 'payload_bytes': sum(row['bytes'] for row in payloads),
        'historical_controller3_manifest_sha256': OLD_MANIFEST,
        'excludes': ['manifest.json self', '__pycache__', '*.pyc', 'external freeze stdout/stderr/raw exit outside immutable root']})
    print(json.dumps({'status': 'FROZEN_SOURCE_CPU_NOT_MODEL_LAUNCH',
        'manifest_sha256': sha(ROOT / 'manifest.json'), 'payload_files': len(payloads),
        'payload_bytes': sum(row['bytes'] for row in payloads), 'old450_comparison': counts,
        'contract_sha256': sha(ROOT / 'contract.json'), 'command_contract_sha256': sha(ROOT / 'command-contract.json'),
        'amendment_sha256': sha(ROOT / 'execution-budget-revision.json'),
        'CPU_receipt_sha256': sha(ROOT / 'CPU-RECEIPT.json')}))


if __name__ == '__main__':
    main()
