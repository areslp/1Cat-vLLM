"""Run4 raw audit; unchanged SSE metrics, original12 core-binding schema."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback

LEGACY_SHA = '80ffc5e2d03e907aa1629368ce2d5a66ad5b9a5cb7c3db05e60f754f4ddfffcf'
PARTIAL_SHA = '6f234a91f55942c3f16ba8bee31c69159bd3740700e481b12dd277622e716b0f'
MATRIX_SHA = '728a58ebe0269cb48e9d3dd57dd60fe640a6598c5659066185450a2b20435b59'
CONTROL_SHA = '31efd893097bf3768f9d6b958143f82b59e59cf1e8efa251189875d6fb15a079'
IDENTITY_SHA = '7a96e11602306ef2a6df3aadbb14626b8ef2fb835f8ae310fab6ef83f5b7c5ea'
ARMS = ('A0', 'B', 'A2')
CORE_KEYS = {'unit', 'invocation_id', 'api_pid', 'api_starttime', 'workers',
             'selected_API_environment', 'telemetry_dir', 'source_files',
             'startup_path', 'startup_sha256', 'capture_receipts', 'E7_identity'}
TRANSPORT_SHA = '6bda93a24d43514e44a5697fa4d5bc92f58b59bc1ae413e7164ba9466c56af66'
TRANSPORT_PINS = {
    'check_wire.py': '738297f1529826dd9796d95c2f9d22f0444908d486e3c8443855b28bbbdbeff3',
    'transport_worker.py': 'a309a920992e9be021026b293cf4fd7a40349b1755103acd53dde9ea5aa2b998',
    'transport_deadline_context.py': 'a60c9910a492b2666a33e32ed705a6085644c9626065c313a6d4474954ea7c14',
    'dependencies.py': '24d0fc96214ab14208543a5e84db21496779dbbaaa04e7002e294aaf8ef46149',
    'DEPENDENCIES.json': 'c7509ace6279d3048d55c9815ac0d131c6e25fa6ed1ca597d36004a2b2ec7b04',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(value, text):
    if not value:
        raise ValueError(text)


def read(path):
    require(path.is_file() and not path.is_symlink()
            and path.stat().st_size <= 8 * 1024**2, 'bounded regular JSON: ' + str(path))
    return json.loads(path.read_bytes())


def helpers(base):
    folder = base / 'review/context-qsa-independent-20261002'
    require(sha(folder / 'independent_review.py') == LEGACY_SHA
            and sha(folder / 'partial_raw_review.py') == PARTIAL_SHA,
            'immutable independent helper SHA changed')
    sys.path.insert(0, str(folder))
    import independent_review as raw
    import partial_raw_review as partial
    require(Path(raw.__file__).resolve() == folder / 'independent_review.py'
            and Path(partial.__file__).resolve() == folder / 'partial_raw_review.py',
            'wrong independent helper import')
    return raw, partial


def resource_binding(base, remote, arm, local_arm):
    binding = read(local_arm / 'model/binding.json')
    require(set(binding) == CORE_KEYS and len(binding['workers']) == 4
            and binding['unit'] == 'step58-contextqsa4-' + arm.lower() + '.service',
            'strict original12 stored core-binding keys')
    require(sha(local_arm / 'model/binding.json') == sha(local_arm / 'guard-binding.json'),
            'published core-binding bytes differ from actual client input')
    startup_path = Path(binding['startup_path'])
    require(startup_path == remote / 'control/attempt1' / arm / 'startup-gate.json',
            'exact arm startup path')
    require(sha(local_arm / 'startup-gate.json') == binding['startup_sha256'],
            'startup SHA must match before reading resource proof')
    startup = read(local_arm / 'startup-gate.json')
    proof = startup['resource_guard_identity']
    package = remote.parent / 'context-qsa-control-prep4'
    legacy = remote.parent / 'service-w2-control-prep4'
    expected = [package / n for n in ('identity_contract.py', 'guard_activation.py',
                                     'nvml_capture.py', 'http_runner.py', 'CONTRACT.json')]
    expected += [legacy / 'original_guard.py']
    transport = remote.parent / 'context-qsa-transport-prep1'
    transport_local = base / transport.name
    require(sha(transport_local / 'MANIFEST.json') == TRANSPORT_SHA,
            'exact transport source manifest')
    transport_paths = [transport / name for name in TRANSPORT_PINS]
    transport_paths += [transport / 'MANIFEST.json', package / 'transport_activation.py']
    source_pins = {str(p): sha(base / p.relative_to(remote.parent)) for p in transport_paths}
    require(all(source_pins[str(transport / name)] == digest
                for name, digest in TRANSPORT_PINS.items()), 'fixed transport source pins')
    expected_transport = {
        'status': 'OWNED_SAME_SOURCE_PUMP_AND_2MiB_WORKER_SELECTED',
        'Pump_module_path': str(transport / 'transport_deadline_context.py'),
        'Pump_module_sha256': TRANSPORT_PINS['transport_deadline_context.py'],
        'Pump_class_module': '_context4_owned_transport',
        'Pump_add_code_path': str(transport / 'transport_deadline_context.py'),
        'worker_path': str(transport / 'transport_worker.py'),
        'worker_sha256': TRANSPORT_PINS['transport_worker.py'],
        'dependency_module_path': str(transport / 'dependencies.py'),
        'transport_manifest_sha256': TRANSPORT_SHA,
        'source_pins': source_pins,
        'source_selection_not_actual_HTTP_or_performance': True,
    }
    require(proof['transport_identity'] == expected_transport,
            'startup actual Pump/relative worker source selection differs')
    expected += transport_paths
    require(proof['status'] == 'OWNED_RESOURCE_CONTRACT_ACTUALLY_BOUND'
            and proof['identity_module_path'] == str(package / 'identity_contract.py')
            and proof['identity_module_sha256'] == IDENTITY_SHA
            and proof['original_guard_module_path'] == str(legacy / 'original_guard.py')
            and proof['gpu_identity_function_is_owned'] is True
            and proof['listener_identity_function_is_owned'] is True
            and proof['process_limit_MiB'] == proof['device_limit_MiB'] == 32384
            and proof['model_cgroup_bytes'] == 120 * 1024**3,
            'new actual guard source/function/cap proof differs')
    require(set(proof['source_pins']) == {str(p) for p in expected}, 'resource proof source inventory')
    for path in expected:
        local = base / path.relative_to(remote.parent)
        require(proof['source_pins'][str(path)] == sha(local)
                == binding['source_files'][str(path)], 'resource proof actual source pin')
    require(proof['original_guard_module_sha256'] == sha(base / 'service-w2-control-prep4/original_guard.py'),
            'original guard SHA')
    require(startup['arm'] == arm
            and startup['status'] == 'PASS_W2_PURE_SERVICE_STARTUP_NOT_PERFORMANCE'
            and startup['unit'] == binding['unit']
            and startup['api_pid'] == binding['api_pid']
            and startup['invocation_id'] == binding['invocation_id']
            and startup['worker_pids'] == {str(w['rank']): w['pid'] for w in binding['workers']}
            and startup['identity']['properties']['MainPID'] == str(binding['api_pid'])
            and startup['capture_receipts'] == binding['capture_receipts'],
            'new resource proof/startup receipt binding')
    require(len(binding['capture_receipts']) == 4
            and {p['rank'] for p in binding['capture_receipts']} == {0, 1, 2, 3},
            'exact four actual constructor capture receipts')
    for pin in binding['capture_receipts']:
        path = Path(pin['path'])
        require(path == remote / 'control/attempt1' / arm / 'service-capture'
                / f'capture-ready-rank{pin["rank"]}.json'
                and sha(local_arm / 'service-capture' / path.name) == pin['sha256'],
                'actual stored constructor capture bytes changed')
    constructor = read(local_arm / 'binding-constructor-gate.json')
    require(constructor['status'] == 'PASS_ORIGINAL_OBSERVATION_FILE_CONSTRUCTOR_NOT_LIVE'
            and constructor['core_binding_keys'] == sorted(CORE_KEYS)
            and constructor['startup_path'] == binding['startup_path']
            and constructor['startup_sha256'] == binding['startup_sha256']
            and constructor['source_files'] == len(binding['source_files'])
            and constructor['capture_receipts'] == 4
            and constructor['resource_guard_identity'] == proof
            and constructor['identity_HTTP_GPU_called'] is False,
            'actual original constructor-before-publication gate absent/different')
    return proof


def group_row(stored, raw, folder, remote_folder):
    extra = {'finished_epoch', 'process_exit_path', 'process_exit_sha256'}
    require(set(stored) == set(raw) | extra
            and {k: stored[k] for k in raw} == raw,
            'GROUP original raw fields / exactly three native parent fields')
    require(type(stored['finished_epoch']) in (float, int)
            and math.isfinite(stored['finished_epoch']) and stored['finished_epoch'] > 0
            and stored['process_exit_path'] == str(remote_folder / 'process-exit.json')
            and stored['process_exit_sha256'] == sha(folder / 'process-exit.json'),
            'GROUP actual reaped parent/child exit binding')


def group_audit(raw, partial, prep, window, remote, arm, index, pin):
    raw.pin(prep / pin['path'], pin)
    frozen = read(prep / pin['path'])
    folder = window / 'control/attempt1' / arm / 'http' / f'{index:03d}-{pin["row_id"]}-repeat{pin["ordinal"]}'
    report = {'arm': arm, 'index': index, 'row_id': pin['row_id'], 'ordinal': pin['ordinal'],
              'input_tokens': frozen['input_tokens'], 'concurrency': frozen['concurrency'],
              'candidate_route_expected': frozen['candidate_route_expected'],
              'dynamic_candidate_activation': frozen['dynamic_candidate_activation'],
              'GROUP_closed': False, 'streams': [], 'errors': [], 'failures': []}
    if not folder.exists():
        report['status'] = 'NOT_EXECUTED_OR_NOT_MIRRORED_NO_INFERENCE'
        return report, None
    require(read(folder / 'frozen-group.json') == frozen, 'actual frozen group/body/order/salt changed')
    main, primes = [], []
    actual = {}
    for kind, packets in (('prime', frozen['primes']), ('request', frozen['requests'])):
        for i, packet in enumerate(packets):
            child = folder / f'{kind}-{i:02d}'
            entry = {'kind': kind, 'index': i, 'request_id': packet['body']['request_id']}
            try:
                row = partial.actual_request(child, packet)
                close = read(child / 'native-close.json')
                require(close['worker_path'] == str(remote.parent /
                        'context-qsa-transport-prep1/transport_worker.py')
                        and close['worker_sha256'] == TRANSPORT_PINS['transport_worker.py']
                        and close['LINE_CAP'] == 2 * 1024**2
                        and close['ORIGINAL_LINE_CAP'] == 512 * 1024,
                        'actual completed HTTP child worker source/line budget proof')
                require(row['raw_response']['path'] == str(remote / child.relative_to(window) / 'raw-lines.jsonl'),
                        'raw response canonical arm/request path')
                actual[kind, i] = row
                (main if kind == 'request' else primes).append(row)
                entry.update(status='PASS_RAW_STREAM_CHILD_ONLY', outputs=len(row['output_token_ids']),
                             TTFT_s=row['ttft_s'], raw_sha256=sha(child / 'raw-lines.jsonl'))
            except Exception as error:
                entry.update(status='RAW_INCOMPLETE_OR_FAILED', error=repr(error))
            report['streams'].append(entry)
    computed = raw.common_metrics(main) if len(main) == len(frozen['requests']) else None
    if computed is not None:
        report['raw_common_metrics'] = computed
    for name in ('FAILURE.json', 'CLEANUP-FAILURE.json'):
        if (folder / name).exists():
            report['failures'].append({'path': str(folder / name), 'sha256': sha(folder / name),
                                       'value': read(folder / name)})
    if (folder / 'GROUP.json').exists():
        try:
            group = read(folder / 'GROUP.json')
            completion = read(folder.parent / f'group-complete-{index:03d}.json')
            require(completion['path'] == str(remote / folder.relative_to(window) / 'GROUP.json')
                    and completion['sha256'] == sha(folder / 'GROUP.json')
                    and (completion['row_id'], completion['ordinal']) == (pin['row_id'], pin['ordinal']),
                    'GROUP complete receipt SHA/path/order')
            require(group['status'] == 'COMPLETE_HTTP_GROUP_NOT_PERFORMANCE_ADMISSION'
                    and (group['arm'], group['row_id'], group['ordinal']) == (arm, pin['row_id'], pin['ordinal'])
                    and group['candidate_route_expected'] == frozen['candidate_route_expected']
                    and group['dynamic_candidate_activation'] == frozen['dynamic_candidate_activation']
                    and set(group['checks']) == set(frozen['required_checks'])
                    and all(v is True for v in group['checks'].values()), 'GROUP identity/strict checks/route')
            require(len(group['request_results']) == len(frozen['requests'])
                    and len(group['priming_results']) == len(frozen['primes'])
                    and len(actual) == len(frozen['requests']) + len(frozen['primes']), 'GROUP/raw cardinality')
            for i, stored in enumerate(group['request_results']):
                child = folder / f'request-{i:02d}'
                group_row(stored, actual['request', i], child, remote / child.relative_to(window))
            for i, stored in enumerate(group['priming_results']):
                require(set(stored['checks']) == raw.PRIME_CHECKS and all(v is True for v in stored['checks'].values()),
                        'exact prime independent native check inventory')
                child = folder / f'prime-{i:02d}'
                group_row(stored['request_result'], actual['prime', i], child, remote / child.relative_to(window))
            for key, value in computed.items():
                if key not in ('counts', 'durations', 'pairs'):
                    raw.close_value(group['metrics'][key], value, 'common metric: ' + key)
            require(len(group['metrics']['per_request']) == len(main), 'common per-request extent')
            for i, stored in enumerate(group['metrics']['per_request']):
                require(stored['common_tokens'] == computed['counts'][i]
                        and stored['common_interval_ordinal_pairs'] == computed['pairs'][i], 'common interval membership')
                raw.close_value(stored['common_covered_interval_s'], computed['durations'][i], 'covered interval time')
            require(not report['failures'], 'closed GROUP also has failure evidence')
            report['GROUP_closed'] = True
        except Exception as error:
            report['errors'].append(repr(error))
    report['status'] = 'CLOSED_GROUP_SUBGATE_ONLY' if report['GROUP_closed'] else 'UNCOMPLETED_GROUP_OR_RESOURCE_STOP'
    compact = None if not report['GROUP_closed'] else {
        'tokens': [r['output_token_ids'] for r in main], 'metrics': computed,
        'prime_cold_c1_TTFT_s': [r['ttft_s'] for r in primes]}
    return report, compact


def paired(values):
    if any(v is None or not math.isfinite(v) or v <= 0 for t in values for v in t):
        return {'status': 'INCOMPLETE_METRIC_NO_FILTERING', 'paired_values': values}
    gains = [100 * (1 - b / ((a0 + a2) / 2)) for a0, b, a2 in values]
    drifts = [100 * abs(a2 - a0) / ((a0 + a2) / 2) for a0, b, a2 in values]
    return {'status': 'DESCRIPTIVE_ONLY_N2', 'paired_values': values,
            'improvement_pct_per_pair': gains, 'AA_drift_pct_per_pair': drifts,
            'median_paired_improvement_pct': statistics.median(gains), 'CI': None}


def descriptions(matrix, results):
    reports, exits = [], []
    for row in matrix['rows']:
        triples = [[results.get((a, row['row_id'], n)) for a in ARMS] for n in (0, 1)]
        value = {**row, 'n_per_row': 2, 'CI': None, 'performance_admission': 'NONE'}
        if any(g is None for t in triples for g in t):
            value.update(status='INCOMPLETE_6_GROUP_COMPARISON_NO_FILTERING',
                         closed_presence=[[g is not None for g in t] for t in triples])
            reports.append(value)
            continue
        comparisons = []
        for ordinal, triple in enumerate(triples):
            tokens = [g['tokens'] for g in triple]
            aa, b0, b2 = tokens[0] == tokens[2], tokens[1] == tokens[0], tokens[1] == tokens[2]
            status = 'PASS_OUTPUT_TOKENS_ONLY' if aa and b0 else 'FAIL_B_ONLY_OUTPUT_CHANGE' if aa else 'INCONCLUSIVE_AA_SELF_VARIATION'
            differences = []
            for i, streams in enumerate(zip(*tokens)):
                for name, a, b in (('A0_A2', streams[0], streams[2]), ('A0_B', streams[0], streams[1]), ('A2_B', streams[2], streams[1])):
                    if a != b:
                        differences.append({'request_index': i, 'pair': name,
                                            'first_different_token_index': next(j for j, (x, y) in enumerate(zip(a, b)) if x != y)})
            comparisons.append({'ordinal': ordinal, 'status': status, 'A0_A2_equal': aa,
                                'B_A0_equal': b0, 'B_A2_equal': b2, 'first_differences': differences})
            exits.append(2 if aa and not b0 else 3 if not aa else 0)
        valid = all(g['metrics']['common_status'] == 'COMMON_WINDOW_OBSERVED' for t in triples for g in t)
        common = [[g['metrics']['pooled_complete_interval_ms_per_output_token'] for g in t] for t in triples]
        value.update(status='DESCRIPTIVE_FIXED_2_TRIPLES_ONLY', token_equality=comparisons,
            common_statuses=[[g['metrics']['common_status'] for g in t] for t in triples],
            common_decode=paired(common) if valid else {'status': 'NO_COMPLETE_COMMON_COMPARISON_NO_FILTERING', 'paired_values': common},
            main_TTFT=paired([[g['metrics']['median_TTFT_s'] for g in t] for t in triples]),
            end_to_end_makespan=paired([[g['metrics']['HTTP_makespan_s'] for g in t] for t in triples]),
            end_to_end_client_tokens_s=[[g['metrics']['end_to_end_client_tokens_s'] for g in t] for t in triples],
            prime_cold_c1_TTFT_s=[[g['prime_cold_c1_TTFT_s'] for g in t] for t in triples])
        reports.append(value)
    return reports, 2 if 2 in exits else 3 if 3 in exits else 0


def run(args):
    base, window, remote = args.base, args.window, args.runtime_root
    raw, partial = helpers(base)
    prep = base / 'context-qsa-prep1'
    require(sha(prep / 'MANIFEST.json') == MATRIX_SHA
            and sha(base / 'context-qsa-control-prep4/MANIFEST.json') == CONTROL_SHA
            and sha(window / 'control/plan.json') == args.plan_sha256, 'source/approved plan SHA differs')
    require(remote.name == 'context-qsa-run4' and window.name == remote.name, 'exact new window identity')
    matrix_pin = next(p for p in read(prep / 'MANIFEST.json')['files']
                      if p['path'] == 'matrix.frozen.json')
    raw.pin(prep / 'matrix.frozen.json', matrix_pin)
    matrix = read(prep / 'matrix.frozen.json')
    selected = args.arms.split(',')
    require(len(set(selected)) == len(selected) and all(a in ARMS for a in selected)
            and (args.scope != 'full' or selected == list(ARMS)), 'explicit closed arm scope')
    arm_reports, groups, results, errors = [], [], {}, []
    for arm in selected:
        arm_root = window / 'control/attempt1' / arm
        wrapper_path = arm_root / 'http-unit.log.exit.json'
        if wrapper_path.exists():
            wrapper = read(wrapper_path)
            require(type(wrapper['exit']) is int, 'actual terminated HTTP wrapper required')
        else:
            require(args.scope == 'stopped-partial' and not (arm_root / 'http').exists(),
                    'HTTP data cannot be audited before wrapper termination')
            wrapper = {'exit': None, 'scope': 'HTTP_NOT_STARTED_OR_NOT_MIRRORED'}
        assigned = [p for p in matrix['groups'] if p['arm'] == arm]
        if (arm_root / 'model/binding.json').exists():
            proof = resource_binding(base, remote, arm, arm_root)
        else:
            require(args.scope == 'stopped-partial', 'closed arm startup binding absent')
            proof = None
        closed = 0
        for index, pin in enumerate(assigned):
            try:
                report, compact = group_audit(raw, partial, prep, window, remote, arm, index, pin)
            except Exception as error:
                report, compact = {'arm': arm, 'index': index, 'row_id': pin['row_id'],
                    'ordinal': pin['ordinal'], 'GROUP_closed': False, 'streams': [],
                    'errors': [repr(error)], 'status': 'GROUP_METADATA_INCOMPLETE_OR_INVALID'}, None
            groups.append(report)
            if compact is not None:
                results[arm, pin['row_id'], pin['ordinal']] = compact
                closed += 1
            if report['errors'] or any(s['status'] == 'RAW_INCOMPLETE_OR_FAILED' for s in report['streams']):
                errors.append({'arm': arm, 'index': index, 'scope': 'raw/GROUP evidence error or incomplete stream'})
        value = {'arm': arm, 'HTTP_wrapper_exit': wrapper['exit'], 'GROUP_closed': closed,
                 'resource_guard_identity': proof, 'arm_completion_verified': False}
        if args.scope != 'stopped-partial':
            final = read(arm_root / 'http/RESULT.json')
            require(wrapper['exit'] == 0 and closed == 38
                    and final['status'] == 'COMPLETE_HTTP_MATRIX_NOT_PERFORMANCE_ADMISSION'
                    and final['arm'] == arm and final['completed_groups'] == 38
                    and final['HTTP_requests'] == 266 and final['outputs'] == 37496
                    and final['resource_guard_identity'] == proof, 'full closed arm totals/resource proof')
            require([(r['row_id'], r['ordinal']) for r in final['group_records']]
                    == [(p['row_id'], p['ordinal']) for p in assigned], 'final group ordering')
            for index, (pin, record) in enumerate(zip(assigned, final['group_records'])):
                folder = arm_root / 'http' / f'{index:03d}-{pin["row_id"]}-repeat{pin["ordinal"]}'
                require(record == read(arm_root / 'http' / f'group-complete-{index:03d}.json')
                        and record['path'] == str(remote / folder.relative_to(window) / 'GROUP.json')
                        and record['sha256'] == sha(folder / 'GROUP.json'), 'final completion receipt bijection')
            gate = read(arm_root / 'final-guard.json')
            require(gate['status'] == 'PASS_COMPLETE_HTTP_REAL_QUEUE_IDENTITY'
                    and gate['arm'] == arm and gate['resource_guard_identity'] == proof
                    and gate['HTTP_result_sha256'] == sha(arm_root / 'http/RESULT.json')
                    and read(arm_root / 'worker-stop.json')['status'] == 'PASS_OWNED_ARM_STOPPED_GPU_EMPTY',
                    'actual final guard/owned arm stop closure')
            value['arm_completion_verified'] = True
        for name in ('FAILURE.json', 'client-resource-FAILURE.json'):
            if (arm_root / 'http' / name).exists():
                value.setdefault('failures', []).append({'path': str(arm_root / 'http' / name),
                    'sha256': sha(arm_root / 'http' / name), 'value': read(arm_root / 'http' / name)})
        arm_reports.append(value)
    rows, compare_exit = descriptions(matrix, results)
    streams = [s for g in groups for s in g['streams'] if s['status'] == 'PASS_RAW_STREAM_CHILD_ONLY']
    total, outputs = len(streams), sum(s['outputs'] for s in streams)
    if args.scope == 'full':
        require(total == 798 and outputs == 112488 and sum(g['GROUP_closed'] for g in groups) == 114,
                'actual fixed full matrix totals')
    return {'status': 'INDEPENDENT_RAW_AUDIT_NOT_WINDOW_OR_PERFORMANCE_PASS',
        'scope': args.scope, 'arms': arm_reports, 'HTTP_verified': total, 'outputs_verified': outputs,
        'GROUP_closed': sum(g['GROUP_closed'] for g in groups), 'groups': groups, 'rows': rows,
        'evidence_errors': errors, 'output_comparison_exit_for_complete_rows': compare_exit,
        'n_per_row': 2, 'CI': None, 'old_NO_GO': 'UNCHANGED', 'no_missing_groups_replaced_or_filtered': True,
        'not_claimed': ['GPU timing/steps', 'logits validation', 'dynamic candidate activation',
                        'continuous GPU peak', 'NVML B-minus-A comparison', 'original restoration'],
        'plan_sha256': args.plan_sha256}


if __name__ == '__main__':
    os.umask(0o077)
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--window', type=Path, required=True)
    p.add_argument('--runtime-root', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--scope', choices=('closed-arm', 'full', 'stopped-partial'), required=True)
    p.add_argument('--arms', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    value = {'runtime': sys.version, 'source_sha256': sha(Path(__file__)),
             'old_raw_helper_sha256': LEGACY_SHA, 'old_request_helper_sha256': PARTIAL_SHA}
    try:
        value['actual'] = run(args)
        code = 1 if value['actual']['evidence_errors'] else 0
    except Exception as error:
        value.update(error=repr(error), traceback=traceback.format_exc(), status='AUDIT_FAILURE_RETAINED_NO_INFERENCE')
        code = 1
    value['checker_exit'] = code
    value['checker_exit_scope'] = 'auditor only; never experimental/window PASS'
    with args.output.open('x') as f:
        json.dump(value, f, allow_nan=False, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'checker_exit': code, 'sha256': sha(args.output)}))
    raise SystemExit(code)
