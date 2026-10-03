"""Offline sampled-memory review; never imports a model or issues live queries."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
REMOTE = Path('/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930')
WINDOW = 'context-qsa-run4'

PACKAGE_MANIFEST_SHA = '31efd893097bf3768f9d6b958143f82b59e59cf1e8efa251189875d6fb15a079'
MATRIX_MANIFEST_SHA = '728a58ebe0269cb48e9d3dd57dd60fe640a6598c5659066185450a2b20435b59'
TRANSPORT_MANIFEST_SHA = '6bda93a24d43514e44a5697fa4d5bc92f58b59bc1ae413e7164ba9466c56af66'
TRANSPORT_SOURCE_PINS = {
    'check_wire.py': '738297f1529826dd9796d95c2f9d22f0444908d486e3c8443855b28bbbdbeff3',
    'transport_worker.py': 'a309a920992e9be021026b293cf4fd7a40349b1755103acd53dde9ea5aa2b998',
    'transport_deadline_context.py': 'a60c9910a492b2666a33e32ed705a6085644c9626065c313a6d4474954ea7c14',
    'dependencies.py': '24d0fc96214ab14208543a5e84db21496779dbbaaa04e7002e294aaf8ef46149',
    'DEPENDENCIES.json': 'c7509ace6279d3048d55c9815ac0d131c6e25fa6ed1ca597d36004a2b2ec7b04',
}
ARMS = ('A0', 'B', 'A2')
CORE_KEYS = {'unit', 'invocation_id', 'api_pid', 'api_starttime', 'workers',
             'selected_API_environment', 'telemetry_dir', 'source_files',
             'startup_path', 'startup_sha256', 'capture_receipts', 'E7_identity'}


def require(value, message):
    if value is not True:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_body(body):
    return hashlib.sha256(json.dumps({k: v for k, v in body.items() if k != 'request_id'},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Review:
    def __init__(self, base, plan_sha):
        self.base, self.pins = Path(base).resolve(), {}
        require(re.fullmatch('[0-9a-f]{64}', plan_sha) is not None,
                'root-reviewed exact run4 plan SHA required')
        self.plan_sha = plan_sha
        self.window = self.base / WINDOW
        package = self.base / 'context-qsa-control-prep4'
        require(self.pin(package / 'MANIFEST.json') == PACKAGE_MANIFEST_SHA,
                'sealed prep4 manifest differs')
        manifest = self.read(package / 'MANIFEST.json')
        files = {r['path']: r for r in manifest['files']}
        for name in ('identity_contract.py', 'guard_activation.py', 'http_runner.py',
                     'nvml_capture.py', 'CONTRACT.json', 'transport_activation.py'):
            require(self.pin(package / name) == files[name]['sha256'], 'resource source SHA differs')
        self.identity = source_module(package / 'identity_contract.py', 'memory_review_identity')
        previous = sys.modules.get('identity_contract')
        try:
            sys.modules['identity_contract'] = self.identity
            self.nvml = source_module(package / 'nvml_capture.py', 'memory_review_nvml')
        finally:
            if previous is None:
                sys.modules.pop('identity_contract', None)
            else:
                sys.modules['identity_contract'] = previous
        require(self.identity.PROCESS_NVML_MIB == self.identity.DEVICE_NVML_MIB == 32384,
                'run4 dual sampled boundary differs')
        old_guard = self.base / 'service-w2-control-prep4/original_guard.py'
        guard_sha = self.pin(old_guard)
        require(guard_sha == '66a5e2220501e9f522304628218bfe99a76bb8f7877a33571131411b679d32c9',
                'immutable original guard source differs')
        self.expected_proof = self.source_proof(files, guard_sha)
        matrix = self.base / 'context-qsa-prep1'
        require(self.pin(matrix / 'MANIFEST.json') == MATRIX_MANIFEST_SHA, 'original matrix seal differs')
        matrix_files = {r['path']: r for r in self.read(matrix / 'MANIFEST.json')['files']}
        require(self.pin(matrix / 'matrix.frozen.json') == matrix_files['matrix.frozen.json']['sha256'],
                'frozen matrix differs')
        self.matrix = self.read(matrix / 'matrix.frozen.json')
        require(self.pin(self.window / 'control/plan.json') == plan_sha,
                'actual approved run4 plan differs')
        plan = self.read(self.window / 'control/plan.json')
        contract = self.read(package / 'CONTRACT.json')
        require(contract['resources']['client_cgroup_bytes'] == 1024**3
                and contract['resources']['client_memory_stat_required'] is True
                and contract['resources']['client_events_max_required'] == 0,
                'new future-only1GiB/stat/events.max0 contract differs')
        require(plan['window'] == str(REMOTE / WINDOW) and plan['arms'] == list(ARMS)
                and plan['schema'] == 'step58-contextqsa4-pure-ABA-control'
                and plan['resource_contract'] == contract['resources']
                and plan['identity_contract_sha256'] == self.expected_proof['identity_module_sha256']
                and plan['files'][str(REMOTE / 'context-qsa-control-prep4/MANIFEST.json')]
                == PACKAGE_MANIFEST_SHA
                and plan['matrix_path'] == str(REMOTE / 'context-qsa-prep1/matrix.frozen.json'),
                'actual run4 plan resource/source/matrix contract differs')

    def source_proof(self, files, guard_sha):
        """Reconstruct exact deployed proof from pinned bytes, no runtime imports."""
        source_pins = {str(REMOTE / 'context-qsa-control-prep4' / name): files[name]['sha256']
                      for name in ('identity_contract.py', 'guard_activation.py',
                                   'nvml_capture.py', 'http_runner.py', 'CONTRACT.json')}
        source_pins[str(REMOTE / 'service-w2-control-prep4/original_guard.py')] = guard_sha
        directory = self.base / 'context-qsa-transport-prep1'
        require(self.pin(directory / 'MANIFEST.json') == TRANSPORT_MANIFEST_SHA,
                'reviewed transport source manifest differs')
        manifest = self.read(directory / 'MANIFEST.json')
        require(manifest['stable_sources'] == TRANSPORT_SOURCE_PINS,
                'five stable transport source pins differ')
        pins = {}
        for name, digest in TRANSPORT_SOURCE_PINS.items():
            require(self.pin(directory / name) == digest,
                    'actual transport source differs: ' + name)
            pins[str(REMOTE / 'context-qsa-transport-prep1' / name)] = digest
        pins[str(REMOTE / 'context-qsa-transport-prep1/MANIFEST.json')] = TRANSPORT_MANIFEST_SHA
        pins[str(REMOTE / 'context-qsa-control-prep4/transport_activation.py')] = files['transport_activation.py']['sha256']
        pump = str(REMOTE / 'context-qsa-transport-prep1/transport_deadline_context.py')
        transport = {
            'status': 'OWNED_SAME_SOURCE_PUMP_AND_2MiB_WORKER_SELECTED',
            'Pump_module_path': pump,
            'Pump_module_sha256': TRANSPORT_SOURCE_PINS['transport_deadline_context.py'],
            'Pump_class_module': '_context4_owned_transport',
            'Pump_add_code_path': pump,
            'worker_path': str(REMOTE / 'context-qsa-transport-prep1/transport_worker.py'),
            'worker_sha256': TRANSPORT_SOURCE_PINS['transport_worker.py'],
            'dependency_module_path': str(REMOTE / 'context-qsa-transport-prep1/dependencies.py'),
            'transport_manifest_sha256': TRANSPORT_MANIFEST_SHA,
            'source_pins': pins,
            'source_selection_not_actual_HTTP_or_performance': True,
        }
        source_pins.update(pins)
        return {
            'status': 'OWNED_RESOURCE_CONTRACT_ACTUALLY_BOUND',
            'identity_module_path': str(REMOTE / 'context-qsa-control-prep4/identity_contract.py'),
            'identity_module_sha256': files['identity_contract.py']['sha256'],
            'original_guard_module_path': str(REMOTE / 'service-w2-control-prep4/original_guard.py'),
            'original_guard_module_sha256': guard_sha,
            'gpu_identity_function_is_owned': True, 'listener_identity_function_is_owned': True,
            'process_limit_MiB': 32384, 'device_limit_MiB': 32384,
            'model_cgroup_bytes': 120 * 1024**3,
            'transport_identity': transport, 'source_pins': source_pins}

    def pin(self, path):
        path = Path(path)
        require(path.is_relative_to(self.base) and not path.is_symlink(), 'input path escapes mirror')
        digest = sha(path)
        self.pins[str(path.relative_to(self.base))] = digest
        return digest

    def read(self, path, cap=16 * 1024**2):
        path = Path(path)
        require(path.stat().st_size <= cap, 'bounded offline JSON input exceeded')
        self.pin(path)
        return json.loads(path.read_text())

    def from_remote(self, path):
        path = Path(path)
        require(path.is_relative_to(REMOTE) and '..' not in path.parts, 'remote pin escapes evidence root')
        return self.base / path.relative_to(REMOTE)

    def binding(self, arm, root):
        path = root / 'model/binding.json'
        if not path.exists():
            return None
        binding = self.read(path)
        require(set(binding) == CORE_KEYS and len(binding['workers']) == 4,
                'exact original12 core binding keys required')
        require(binding['unit'] == 'step58-contextqsa4-' + arm.lower() + '.service',
                'actual startup unit differs')
        startup_path = self.from_remote(binding['startup_path'])
        require(self.pin(startup_path) == binding['startup_sha256'],
                'startup receipt SHA differs')
        startup = self.read(startup_path)
        require(startup['resource_guard_identity'] == self.expected_proof,
                'SHA-bound startup receipt actual owned proof differs')
        require(startup['status'] == 'PASS_W2_PURE_SERVICE_STARTUP_NOT_PERFORMANCE'
                and startup['arm'] == arm and startup['unit'] == binding['unit']
                and startup['api_pid'] == binding['api_pid']
                and startup['invocation_id'] == binding['invocation_id']
                and startup['capture_receipts'] == binding['capture_receipts']
                and startup['worker_pids'] == {str(w['rank']): w['pid'] for w in binding['workers']}
                and startup['identity']['properties']['MainPID'] == str(binding['api_pid']),
                'original startup binding identity contract differs')
        for remote, digest in self.expected_proof['source_pins'].items():
            require(binding['source_files'].get(remote) == digest
                    and self.pin(self.from_remote(remote)) == digest,
                    'actual owned guard source binding differs')
        workers = binding['workers']
        require([w['rank'] for w in workers] == list(range(4))
                and tuple(w['physical_uuid'] for w in workers) == self.identity.UUIDS
                and len({w['pid'] for w in workers}) == 4, 'four exact rank/PID/UUID bindings')
        captures = binding['capture_receipts']
        require(len(captures) == 4 and {r['rank'] for r in captures} == set(range(4)),
                'four exact capture rank pins required')
        for capture in captures:
            require(self.pin(self.from_remote(capture['path'])) == capture['sha256'],
                    'actual startup capture SHA differs')
        constructor = self.read(root / 'binding-constructor-gate.json')
        require(constructor['status'] == 'PASS_ORIGINAL_OBSERVATION_FILE_CONSTRUCTOR_NOT_LIVE'
                and constructor['core_binding_keys'] == sorted(CORE_KEYS)
                and constructor['startup_path'] == binding['startup_path']
                and constructor['startup_sha256'] == binding['startup_sha256']
                and constructor['source_files'] == len(binding['source_files'])
                and constructor['capture_receipts'] == 4
                and constructor['resource_guard_identity'] == self.expected_proof
                and constructor['identity_HTTP_GPU_called'] is False,
                'actual producer constructor gate receipt differs')
        return binding

    def closed(self, arm, root):
        proof = root / 'worker-stop.json'
        if proof.exists():
            value = self.read(proof)
            require(value['status'] == 'PASS_OWNED_ARM_STOPPED_GPU_EMPTY' and value['arm'] == arm,
                    'complete arm stop proof differs')
            ended = self.read(root / 'http-unit-ended.json')
            require(ended['MainPID'] == '0' and ended['ActiveState'] in ('inactive', 'failed'),
                    'arm HTTP has not closed')
            return {'scope': 'ARM_STOP_AND_HTTP_TERMINAL', 'path': str(proof.relative_to(self.base))}
        for label in ('exit-final', 'outer-before-restore'):
            path = self.window / 'control/attempt1' / (label + '-cleanup.json')
            if not path.exists():
                continue
            value = self.read(path)
            units = {'step58-contextqsa4-' + arm.lower() + '.service',
                     'step58-contextqsa4-http-' + arm.lower() + '.service'}
            rows = {r['unit']: r for r in value['rows'] if r['unit'] in units}
            if set(rows) != units:
                continue
            states = [r.get('after') or r['before'] for r in rows.values()]
            if all(s.get('MainPID') == '0' and s.get('ActiveState') in ('inactive', 'failed') for s in states):
                return {'scope': 'FAILED_ARM_NATIVE_CLEANUP_TERMINAL',
                        'path': str(path.relative_to(self.base)), 'cleanup_status': value['status']}
        raise ValueError('explicit closed arm evidence absent: ' + arm)

    def client(self, arm, root):
        path = root / 'http/client-resource.json'
        if not path.exists():
            return {'status': 'UNAVAILABLE', 'actual_peak_claimed': False}
        value = self.read(path)
        files = value['files']
        expected = '/system.slice/step58-contextqsa4-http-' + arm.lower() + '.service'
        require(value['proc_self_cgroup_raw'].strip() == '0::' + expected
                and value['cgroup_path'] == '/sys/fs/cgroup' + expected
                and value['systemd_run_CLI_peak_used_as_actual_peak'] is False,
                'actual client cgroup route/CLI peak substitution differs')
        events = dict(line.split() for line in files['memory.events'].splitlines())
        stat = dict(line.split() for line in files['memory.stat'].splitlines())
        require(bool(stat) and all(int(value) >= 0 for value in stat.values()),
                'actual client memory.stat absent/invalid')
        checks = {'memory_max_1GiB': files['memory.max'] == str(1024**3),
            'current_peak_within_1GiB': all(0 <= int(files[n]) <= 1024**3
                for n in ('memory.current', 'memory.peak')),
            'events_max_zero': events.get('max') == '0',
            'swap_zero': files['memory.swap.max'] == files['memory.swap.current'] == '0',
            'oom_zero': all(events.get(n) == '0' for n in ('oom', 'oom_kill')),
            'client_cpu14_42': files['cpuset.cpus.effective'] == '14,42'}
        require(value['checks'] == checks and value['status'] == (
            'PASS_ACTUAL_CLIENT_CGROUP' if all(checks.values()) else 'FAIL'),
            'client resource receipt independent recompute differs')
        return {'status': value['status'], 'actual_cgroup_peak_bytes': int(files['memory.peak']),
                'checks': checks, 'files': files, 'actual_memory_stat': stat,
                'path': str(path.relative_to(self.base))}

    def boundary(self, arm, binding, value, source, pointer, meta):
        identity = value['identity']
        props = identity['properties']
        require(props['MainPID'] == str(binding['api_pid'])
                and props['InvocationID'] == binding['invocation_id']
                and props['ControlGroup'] == '/system.slice/' + binding['unit']
                and props['ActiveState'] == 'active' and props['SubState'] == 'running'
                and props['NRestarts'] == '0', 'boundary API/source invocation differs')
        require(int(props['MemoryMax']) == 120 * 1024**3
                and all(0 <= int(props[n]) <= 120 * 1024**3 for n in ('MemoryCurrent', 'MemoryPeak'))
                and props['MemorySwapCurrent'] == props['MemorySwapMax'] == '0'
                and all(identity['memory_events'].get(n) == '0' for n in ('oom', 'oom_kill')),
                'boundary original model cgroup gates differ')
        nvml = identity['actual_NVML']
        require(nvml['process_limit_MiB'] == nvml['device_limit_MiB'] == 32384,
                'saved boundary used wrong process/device contract')
        process_text = '\n'.join(f'{r["pid"]}, {r["uuid"]}, {r["used_MiB"]} MiB' for r in nvml['processes'])
        device_text = '\n'.join(f'{u}, {m}' for u, m in nvml['devices_used_MiB'].items())
        require(self.identity.gpu_identity(process_text, device_text, binding['workers']) == nvml,
                'saved actual NVML did not pass exact native four-rank dual gate')
        processes = {r['uuid']: r['used_MiB'] for r in nvml['processes']}
        result = []
        for kind, values in (('process', processes), ('device', nvml['devices_used_MiB'])):
            for uuid in self.identity.UUIDS:
                result.append({'arm': arm, **meta, 'kind': kind, 'uuid': uuid,
                    'used_MiB': values[uuid], 'source': str(source.relative_to(self.base)),
                    'json_pointer': pointer + '/identity/actual_NVML',
                    'mapping': 'DIRECT_SAVED_RETURN_IDENTITY', 'raw_query_mapping': 'UNMAPPED'})
        return result

    def raw(self, root, binding):
        results = []
        raw_paths = sorted(root.rglob('*.raw.json'))
        class_paths = set(root.rglob('*.classification.json'))
        require(len(raw_paths) <= 4096, 'bounded per-arm NVML receipts exceeded')
        for path in raw_paths:
            pair = path.with_name(path.name.replace('.raw.json', '.classification.json'))
            require(pair in class_paths, 'raw classification partner missing')
            class_paths.remove(pair)
            raw, classified = self.read(path), self.read(pair)
            require(self.from_remote(classified['raw_path']) == path
                    and classified['raw_sha256'] == self.pin(path)
                    and raw['status'] == 'RAW_RECORDED_BEFORE_NATIVE_VALIDATION'
                    and raw['additional_query'] is False and raw['query_calls'] == 1
                    and classified['native_gate_replaced'] is False
                    and 0 < raw['query_started_ns'] <= raw['query_returned_ns'] <= classified['classified_ns'],
                    'raw/classification SHA/order/query-count binding differs')
            command_failure = classified['kind'] == 'ORIGINAL_QUERY_COMMAND_FAILED'
            computed = self.nvml.classify(raw['argv'], raw['raw'])
            if not command_failure:
                require(all(classified.get(k) == v for k, v in computed.items()),
                        'classification differs from exact frozen pure classifier')
            issues = [{'kind': 'ORIGINAL_QUERY_COMMAND_FAILED', 'error': classified['error']}] if command_failure else computed.get('issues', [])
            issues = list(issues)
            parsed = computed.get('parsed_rows', [])
            if computed['kind'] in ('PROCESS_MEMORY', 'DEVICE_MEMORY') and not command_failure:
                if not any(r['kind'] == 'UNKNOWN_FIELD' for r in issues):
                    uuids = [r['uuid'] for r in parsed]
                    if not (len(uuids) == 4 and len(set(uuids)) == 4
                            and set(uuids) == set(self.identity.UUIDS)):
                        issues.append({'kind': 'UUID_BIJECTION_FAILURE',
                                       'observed_UUIDs': uuids})
                    if computed['kind'] == 'PROCESS_MEMORY' and binding is not None:
                        expected = {(w['pid'], w['physical_uuid']) for w in binding['workers']}
                        observed = {(r['pid'], r['uuid']) for r in parsed}
                        if observed != expected:
                            issues.append({'kind': 'PROCESS_PID_UUID_BIJECTION_FAILURE',
                                'expected': sorted(expected), 'observed': sorted(observed)})
            results.append({'source': str(path.relative_to(self.base)), 'source_sha256': self.pin(path),
                'classification_source': str(pair.relative_to(self.base)),
                'query_started_ns': raw['query_started_ns'], 'query_returned_ns': raw['query_returned_ns'],
                'kind': computed['kind'], 'issues': issues, 'parsed_rows': parsed,
                'mapping': 'UNMAPPED', 'reason': 'raw recorder has no explicit row/repeat/prime/main/boundary reference',
                'original_classification': classified})
        require(not class_paths, 'orphan classification without raw')
        return results

    def arm(self, arm):
        root = self.window / 'control/attempt1' / arm
        closure = self.closed(arm, root)
        binding = self.binding(arm, root)
        boundaries, completed_groups, failures = [], [], []
        pins = [p for p in self.matrix['groups'] if p['arm'] == arm]
        for index, pin in enumerate(pins):
            group_root = root / 'http' / f'{index:03d}-{pin["row_id"]}-repeat{pin["ordinal"]}'
            if not group_root.exists():
                continue
            require(binding is not None, 'saved group boundaries lack actual startup binding')
            original_path = self.base / 'context-qsa-prep1' / pin['path']
            require(self.pin(original_path) == pin['sha256'], 'fixed group body/order/salt source differs')
            frozen = self.read(original_path)
            require(self.read(group_root / 'frozen-group.json') == frozen, 'actual assigned frozen group differs')
            group_path = group_root / 'GROUP.json'
            group = self.read(group_path) if group_path.exists() else None
            if group is not None:
                require(group['arm'] == arm and group['row_id'] == pin['row_id']
                        and group['ordinal'] == pin['ordinal']
                        and group['status'] == 'COMPLETE_HTTP_GROUP_NOT_PERFORMANCE_ADMISSION',
                        'saved GROUP row/repeat/status differs')
                completed_groups.append({'index': index, 'row_id': pin['row_id'], 'repeat': pin['ordinal']})
                meta = {'row_id': pin['row_id'], 'repeat': pin['ordinal'], 'phase': 'main',
                    'prime_index': None, 'body_signature': [canonical_body(p['body']) for p in frozen['requests']]}
                for tag in ('before', 'after'):
                    boundaries.extend(self.boundary(arm, binding, group[tag], group_path,
                        '/' + tag, dict(meta, boundary=tag)))
            for prime in range(pin['primes']):
                path = group_root / f'prime-{prime:02d}.json'
                if not path.exists():
                    continue
                value = self.read(path)
                require(value['index'] == prime, 'prime index differs')
                if group is not None:
                    require(group['priming_results'][prime] == value, 'GROUP/standalone prime duplicate differs')
                meta = {'row_id': pin['row_id'], 'repeat': pin['ordinal'], 'phase': 'prime',
                    'prime_index': prime, 'body_signature': [canonical_body(frozen['primes'][prime]['body'])]}
                for tag in ('before', 'after'):
                    boundaries.extend(self.boundary(arm, binding, value[tag], path,
                        '/' + tag, dict(meta, boundary=tag)))
            failure = group_root / 'FAILURE.json'
            if failure.exists():
                failures.append({'source': str(failure.relative_to(self.base)), 'receipt': self.read(failure)})
        raw = self.raw(root, binding)
        issues = sorted((r for r in raw if r['issues']), key=lambda r: r['query_returned_ns'])
        maxima = {}
        for kind in ('PROCESS_MEMORY', 'DEVICE_MEMORY'):
            maxima[kind] = {u: max((p['used_MiB'] for r in raw if r['kind'] == kind
                for p in r['parsed_rows'] if p['uuid'] == u), default=None) for u in self.identity.UUIDS}
        client = self.client(arm, root)
        result_path = root / 'http/RESULT.json'
        result = self.read(result_path) if result_path.exists() else None
        if result is not None:
            require(result['arm'] == arm and result['resource_guard_identity'] == self.expected_proof
                    and result['status'] == 'COMPLETE_HTTP_MATRIX_NOT_PERFORMANCE_ADMISSION'
                    and result['completed_groups'] == 38
                    and result['measured_requests'] == 146 and result['prime_requests'] == 120
                    and result['HTTP_requests'] == 266 and result['outputs'] == 37496
                    and len(result['group_records']) == 38,
                    'HTTP result actual resource identity proof differs')
            for record in result['group_records']:
                require(self.pin(self.from_remote(record['path'])) == record['sha256'],
                        'completed GROUP differs from native arm result SHA')
        arm_failures = []
        for name in ('FAILURE.json', 'client-resource-FAILURE.json'):
            path = root / 'http' / name
            if path.exists():
                arm_failures.append({'source': str(path.relative_to(self.base)),
                                     'receipt': self.read(path)})
        return {'arm': arm, 'closure': closure, 'actual_owned_identity_proof':
            self.expected_proof if binding else None,
            'identity_proof_status': 'VERIFIED_CORE12_STARTUP_SHA_CONSTRUCTOR_RECEIPT_CHAIN'
                if binding else 'UNAVAILABLE',
            'completed_groups': completed_groups, 'assigned_groups': 38,
            'boundaries': boundaries, 'raw_samples': raw, 'raw_pair_count': len(raw),
            'raw_unmapped_count': len(raw), 'sampled_max_MiB_by_UUID': maxima,
            'first_recorded_resource_issue': issues[0] if issues else None,
            'client_resource': client, 'HTTP_result': result, 'group_failures': failures,
            'arm_failures': arm_failures,
            'recorded_resource_condition': ('FAILURE_RECORDED' if issues or client['status'] == 'FAIL'
                else 'PASS_ALL_AVAILABLE_SAMPLES' if binding and client['status'] == 'PASS_ACTUAL_CLIENT_CGROUP'
                else 'UNAVAILABLE_NOT_PASS'),
            'scope_complete': len(completed_groups) == 38 and result is not None
                and len(boundaries) == 2 * (38 + 120) * 2 * 4,
            'raw_max_scope': 'all parsed returned PID/UUID rows; ownership issues separately '
                'preserved, no claim that invalid/foreign process rows are owned allocation',
            'complete_boundary_inventory_is_overall_run_PASS': False,
            'continuous_GPU_peak_claimed': False, 'candidate_causal_memory_attribution': False}


def differences(reports):
    def key(row):
        return tuple(row[n] for n in ('row_id', 'repeat', 'phase', 'prime_index', 'boundary', 'kind', 'uuid'))
    indexed = {}
    for arm, report in reports.items():
        rows = report['boundaries']
        values = {key(row): row for row in rows}
        require(len(values) == len(rows), 'duplicate direct boundary assignment')
        indexed[arm] = values
    output, unavailable = [], []
    for reference in ('A0', 'A2'):
        b, a = indexed.get('B', {}), indexed.get(reference, {})
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                unavailable.append({'comparison': 'B-minus-' + reference,
                    'key': list(k), 'status': 'MISSING_MATCHED_BOUNDARY'})
                continue
            require(a[k]['body_signature'] == b[k]['body_signature'],
                    'matched boundary body/salt differs beyond request_id')
            output.append({'comparison': 'B-minus-' + reference,
                **{n: b[k][n] for n in ('row_id', 'repeat', 'phase', 'prime_index', 'boundary', 'kind', 'uuid')},
                'B_used_MiB': b[k]['used_MiB'], 'A_used_MiB': a[k]['used_MiB'],
                'delta_MiB': b[k]['used_MiB'] - a[k]['used_MiB'],
                'B_source': b[k]['source'], 'B_pointer': b[k]['json_pointer'],
                'A_source': a[k]['source'], 'A_pointer': a[k]['json_pointer'],
                'scope': 'matched saved sampled identity; neither continuous peak nor causal candidate allocation'})
    return output, unavailable


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='/private/tmp/flash-next-step58-20260930')
    parser.add_argument('--arms', nargs='+', choices=ARMS, required=True)
    parser.add_argument('--plan-sha256', required=True,
                        help='exact new run4 plan SHA reviewed by root')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    require(destination.is_relative_to(HERE) and destination != HERE,
            'all outputs must be inside the owned new review directory')
    require(len(set(args.arms)) == len(args.arms), 'duplicate arm input')
    destination.mkdir(mode=0o700)
    review = None
    try:
        review = Review(args.base, args.plan_sha256)
        arms = {arm: review.arm(arm) for arm in ARMS if arm in args.arms}
        native_raw = review.raw(review.window / 'control/attempt1/nvml-native', None)
        delta, unavailable = differences(arms)
        complete = set(arms) == set(ARMS) and all(a['scope_complete'] for a in arms.values())
        result = {'status': 'COMPLETE_SAMPLED_MEMORY_REVIEW_ONLY' if complete else 'PARTIAL_EXISTING_SAMPLES_ONLY',
            'window': WINDOW, 'approved_plan_sha256': args.plan_sha256, 'arms': arms,
            'paired_boundary_differences': delta, 'unavailable_matches': unavailable,
            'native_control_raw_unmapped': native_raw,
            'continuous_GPU_peak_claimed': False, 'candidate_causal_memory_attribution': False,
            'not_performance_admission': True, 'not_restoration_acceptance': True,
            'native_control_raw_status': 'preserved outside per-arm matching; no phase inferred',
            'pins': review.pins, 'script_sha256': sha(__file__)}
        save(destination / 'MEMORY-REVIEW.json', result)
        fields = ['comparison', 'row_id', 'repeat', 'phase', 'prime_index', 'boundary',
                  'kind', 'uuid', 'B_used_MiB', 'A_used_MiB', 'delta_MiB',
                  'B_source', 'B_pointer', 'A_source', 'A_pointer', 'scope']
        with (destination / 'BOUNDARY-DIFFERENCES.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(delta)
        print(json.dumps({'status': result['status'], 'arms': {
            a: {'completed_groups': len(v['completed_groups']), 'raw_pairs': v['raw_pair_count'],
                'raw_unmapped': v['raw_unmapped_count'], 'sampled_max_MiB_by_UUID': v['sampled_max_MiB_by_UUID'],
                'client_resource': v['client_resource'],
                'first_issue_source': v['first_recorded_resource_issue']['source']
                    if v['first_recorded_resource_issue'] else None} for a, v in arms.items()},
            'matched_differences': len(delta), 'unavailable': len(unavailable)}))
    except BaseException as error:
        save(destination / 'PARSER-FAILURE.json', {'status': 'INVALID_OR_INCOMPLETE_EVIDENCE',
            'error': repr(error), 'pins': review.pins if review else {},
            'original_inputs_unmodified': True, 'no_live_operations': True})
        raise


if __name__ == '__main__':
    main()
