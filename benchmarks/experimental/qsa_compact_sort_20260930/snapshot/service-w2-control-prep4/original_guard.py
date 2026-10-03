"""Real group-external unit/queue/cache/E7 evidence; missing proof is STOP.

This module is not a model observer. It reads existing metrics and the original
E7 exporter only before release/after all request processes terminate. A future
controller must independently pin the startup receipt and construct its complete
binding; synthetic parser tests do not establish that live startup gate.
"""
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

from frozen import BASE
from io_tools import save, sha
from identity_contract import gpu_identity, listener_identity
from e7_counters import e7_deltas

LABELS = {'engine': '0', 'model_name': 'flash-next'}
METRICS = {'vllm:num_requests_running', 'vllm:num_requests_waiting',
           'vllm:prefix_cache_queries_total', 'vllm:prefix_cache_hits_total',
           'vllm:request_success_total'}
FENCE = BASE / 'service-numeric-candidate-B-control-prep1/restoration/original-check/exporter_fence.py'
FENCE_SHA = '9c0f2c81720a1d49f1c49b932409160c750a99c11cbd38bcb990949dc2ce4166'


def metric_rows(text):
    rows = {}
    for line in text.splitlines():
        if not any(line.startswith(name + '{') for name in METRICS):
            continue
        match = re.fullmatch(r'(vllm:[a-z_]+)\{(.*)\} ([^ ]+)', line)
        if not match or match[1] not in METRICS:
            raise ValueError('native metric schema changed')
        labels, end = {}, 0
        for item in re.finditer(r'([A-Za-z_]\w*)="((?:[^"\\]|\\.)*)"(?:,|$)', match[2]):
            if item.start() != end or item[1] in labels:
                raise ValueError('metric labels duplicate/unparsed')
            labels[item[1]] = json.loads('"' + item[2] + '"')
            end = item.end()
        if end != len(match[2]) or any(labels.get(k) != v for k, v in LABELS.items()):
            raise ValueError('foreign engine/model or malformed labels')
        if set(labels) != set(LABELS) | ({'finished_reason'}
                if match[1] == 'vllm:request_success_total' else set()):
            raise ValueError('unexpected native metric dimensions')
        number = float(match[3])
        if not math.isfinite(number) or number < 0 or not number.is_integer():
            raise ValueError('native counter/gauge must be finite nonnegative integer')
        key = (match[1], labels.get('finished_reason'))
        if key in rows:
            raise ValueError('duplicate metric series')
        rows[key] = int(number)
    required = {(key, None) for key in METRICS if key != 'vllm:request_success_total'}
    required |= {('vllm:request_success_total', why)
                 for why in ('stop', 'length', 'abort', 'error', 'repetition')}
    if rows.keys() != required:
        raise ValueError('required native queue/cache/terminal metrics absent')
    return rows


def counter_delta(before, after):
    if before.keys() != after.keys():
        raise ValueError('native counter series changed')
    delta = {key: after[key] - value for key, value in before.items()}
    if any(value < 0 for value in delta.values()):
        raise ValueError('native counter regressed')
    return delta


def terminal_delta(metrics, outcomes):
    expected_abort = outcomes.count('EXPECTED_CANCELLED_NOT_COMPLETED')
    expected_length = outcomes.count('COMPLETE')
    if expected_abort + expected_length != len(outcomes):
        raise ValueError('unknown frozen terminal outcome')
    # AsyncLLM.abort removes output processor state before sending ABORT. It
    # does not update finished-request stats and late output is then ignored.
    # A client cut can race a normal finish, so preserve the actual 6..8 length
    # delta for the frozen six-complete/two-cut group. This is not an engine ACK.
    length = metrics[('vllm:request_success_total', 'length')]
    return (expected_length <= length <= expected_length + expected_abort
            and all(metrics[('vllm:request_success_total', reason)] == 0
                    for reason in ('abort', 'stop', 'error', 'repetition')))


def exact_cancel_cut(packet, row):
    if packet['cancel_contract'] is None:
        return row['status'] == 'COMPLETE'
    cut = row['actual_cut']
    return (packet['cancel_contract'] == row['cancel_contract'] == {
                'positive_chunks': 2, 'cut': 'after-positive-chunk-count'}
            and row['status'] == 'EXPECTED_CANCELLED_NOT_COMPLETED'
            and row['completed'] is False and row['done'] is False
            and row['finishes'] == row['usages'] == []
            and len(row['positive_chunks']) == cut['positive_chunks'] == 2
            and cut['event_ordinal'] == row['positive_chunks'][-1]['ordinal']
            and cut['output_tokens'] == len(row['output_token_ids']) > 0)


def output(argv):
    return subprocess.check_output(argv, text=True, timeout=10).strip()


class Observation:
    def __init__(self, binding):
        # This binding is produced only by the independent startup gate, never
        # inferred from request concurrency or a source configuration label.
        required = {'unit', 'invocation_id', 'api_pid', 'api_starttime', 'workers',
                    'selected_API_environment', 'telemetry_dir', 'source_files',
                    'startup_path', 'startup_sha256', 'capture_receipts', 'E7_identity'}
        if set(binding) != required or len(binding['workers']) != 4:
            raise ValueError('complete independent startup binding required')
        if sha(binding['startup_path']) != binding['startup_sha256']:
            raise ValueError('startup receipt bytes changed')
        startup = json.loads(Path(binding['startup_path']).read_text())
        if (startup['status'] != 'PASS_W2_PURE_SERVICE_STARTUP_NOT_PERFORMANCE'
                or startup['unit'] != binding['unit']
                or startup['api_pid'] != binding['api_pid']
                or startup['invocation_id'] != binding['invocation_id']
                or startup['capture_receipts'] != binding['capture_receipts']
                or startup['worker_pids'] != {str(w['rank']): w['pid']
                                             for w in binding['workers']}
                or startup['identity']['properties']['MainPID'] != str(binding['api_pid'])):
            raise ValueError('pure-service startup gate absent')
        self.binding = binding
        self.descriptors = {}
        for path, expected in binding['source_files'].items():
            if sha(path) != expected:
                raise ValueError('startup source/library pin mismatch')
            self.descriptors[path] = self.file_identity(path)
        if not self.descriptors or len(binding['capture_receipts']) != 4:
            raise ValueError('source and four rank capture receipts absent')
        for row in binding['capture_receipts']:
            if sha(row['path']) != row['sha256']:
                raise ValueError('capture receipt changed')

    @staticmethod
    def file_identity(path):
        stat = Path(path).stat()
        return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]

    @staticmethod
    def pid_start(pid):
        return int((Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19])

    def identity(self):
        return self.actual_identity(self.binding, self.descriptors)

    @classmethod
    def actual_identity(cls, binding, descriptors):
        # The independent startup collector uses the same real process/resource
        # validator before publishing its first PASS receipt.
        b = binding
        keys = ('MainPID', 'InvocationID', 'NRestarts', 'ActiveState', 'SubState',
                'ControlGroup', 'MemoryCurrent', 'MemoryPeak', 'MemoryMax',
                'MemorySwapCurrent', 'MemorySwapMax')
        props = dict(line.split('=', 1) for line in output(['systemctl', 'show', b['unit'],
                     *['-p' + key for key in keys]]).splitlines())
        if (props['MainPID'] != str(b['api_pid']) or props['InvocationID'] != b['invocation_id']
                or props['NRestarts'] != '0' or props['ActiveState'] != 'active'
                or props['SubState'] != 'running'
                or props['ControlGroup'] != '/system.slice/' + b['unit']
                or cls.pid_start(b['api_pid']) != b['api_starttime']):
            raise ValueError('actual API PID/starttime/invocation/restart changed')
        if (int(props['MemoryMax']) != 120 * 1024**3
                or any(int(props[k]) != 0 for k in ('MemorySwapCurrent', 'MemorySwapMax'))
                or any(int(props[k]) > 120 * 1024**3 for k in ('MemoryCurrent', 'MemoryPeak'))):
            raise ValueError('actual cgroup memory/swap cap differs')
        cg = Path('/sys/fs/cgroup') / props['ControlGroup'].lstrip('/')
        events = dict(line.split() for line in (cg / 'memory.events').read_text().splitlines())
        if any(int(events[key]) != 0 for key in ('oom', 'oom_kill')):
            raise ValueError('owned model cgroup OOM')
        environment = dict(row.decode().split('=', 1) for row in
            (Path('/proc') / str(b['api_pid']) / 'environ').read_bytes().split(b'\0') if b'=' in row)
        if any(environment.get(key) != value for key, value in b['selected_API_environment'].items()):
            raise ValueError('actual selected API environment differs')
        if any(key.startswith(('STEP58_NUMERIC_', 'STEP58_ADMISSION_', 'PHYSICAL_FIXTURE_'))
               for key in environment):
            raise ValueError('diagnostic activation leaked into timed service')
        for path, before in descriptors.items():
            if cls.file_identity(path) != before:
                raise ValueError('pinned source/library changed during arm')
        for rank, worker in enumerate(b['workers']):
            pid = worker['pid']
            if (worker['rank'] != rank or cls.pid_start(pid) != worker['starttime_ticks']
                    or (Path('/proc') / str(pid) / 'cgroup').read_text().strip()
                       != '0::' + props['ControlGroup']):
                raise ValueError('actual worker PID/rank/starttime/cgroup differs')
        nvml = gpu_identity(output(['nvidia-smi',
            '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader']),
            output(['nvidia-smi', '--query-gpu=uuid,memory.used',
                    '--format=csv,noheader,nounits']), b['workers'])
        listener = listener_identity(output(['ss', '-H', '-ltnp', 'sport = :8201']),
                                     b['api_pid'])
        return {'properties': props, 'memory_events': events,
                'actual_NVML': nvml, 'listener': listener}

    def metrics(self):
        with urllib.request.urlopen('http://127.0.0.1:8201/health', timeout=5) as response:
            if response.status != 200:
                raise ValueError('owned API health failed')
        with urllib.request.urlopen('http://127.0.0.1:8201/metrics', timeout=5) as response:
            raw = response.read(2 * 1024**2 + 1)
        if len(raw) > 2 * 1024**2:
            raise ValueError('metrics byte bound')
        return raw.decode(), metric_rows(raw.decode())

    def idle(self):
        _, rows = self.metrics()
        if any(rows[(name, None)] for name in ('vllm:num_requests_running', 'vllm:num_requests_waiting')):
            raise ValueError('owned API actual queue not empty')
        return time.time()

    def e7(self):
        rows = [json.loads((Path(self.binding['telemetry_dir']) /
                f'e7-{worker["pid"]}.json').read_text()) for worker in self.binding['workers']]
        if any(row['pid'] != worker['pid'] or row['rank'] != rank
               for rank, (row, worker) in enumerate(zip(rows, self.binding['workers']))):
            raise ValueError('exact existing exporter worker files differ')
        for row, expected in zip(rows, self.binding['E7_identity']):
            if {key: row[key] for key in ('pid', 'rank', 'ranks', 'mode',
                    'module_path', 'package_hashes')} != expected:
                raise ValueError('exact prebound original E7 source/PID identity differs')
        return rows


class OriginalGuard:
    def __init__(self, observation, output_dir):
        if type(observation) is not Observation:
            raise TypeError('production guard requires real Observation; no injected True adapter')
        if sha(FENCE) != FENCE_SHA:
            raise ValueError('frozen exporter fence source changed')
        spec = importlib.util.spec_from_file_location('w2_frozen_exporter_fence', FENCE)
        self.fence = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.fence)
        self.obs, self.root = observation, Path(output_dir)
        self.root.mkdir(mode=0o700)
        self.expected = observation.e7()
        self.ordinal = 0

    def stable(self, label, completed):
        traces = []
        def trace(row):
            traces.append(row)
            if len(traces) > 64:
                raise ValueError('passive publication trace budget')
        result = self.fence.wait_fence(label, self.expected, self.obs.e7, self.obs.idle,
            lambda name, value: save(self.root / name, value, 2 * 1024**2),
            completed, trace)
        save(self.root / (label + '-trace.json'), traces, 4 * 1024**2)
        return result

    def before(self, group, deadline):
        if time.perf_counter() >= deadline:
            raise TimeoutError('matrix deadline before group-external guard')
        label = f'group-{self.ordinal:03d}'
        self.ordinal += 1
        identity = self.obs.identity()
        if time.perf_counter() >= deadline:
            raise TimeoutError('matrix deadline during group-external before guard')
        e7 = self.stable(label + '-before', 0)
        raw, metrics = self.obs.metrics()
        if any(metrics[(name, None)] for name in ('vllm:num_requests_running', 'vllm:num_requests_waiting')):
            raise ValueError('queue changed at before boundary')
        with (self.root / (label + '-before-metrics.txt')).open('x') as handle:
            handle.write(raw)
        return {'label': label, 'identity': identity, 'E7': e7,
                'metrics': [{'name': k[0], 'reason': k[1], 'value': v} for k, v in metrics.items()]}

    def after(self, group, before, results, deadline, *, prime=None):
        if time.perf_counter() >= deadline:
            raise TimeoutError('matrix deadline after request close/reap')
        label = before['label']
        old = {(row['name'], row['reason']): row['value'] for row in before['metrics']}
        publication_deadline = min(deadline, time.perf_counter() + 15)
        delta = None
        observations = []
        raw = None
        try:
            while time.perf_counter() < publication_deadline:
                raw, metrics = self.obs.metrics()
                observation = {'wall_epoch': time.time(), 'native_values': [
                    {'name': key[0], 'reason': key[1], 'value': value}
                    for key, value in metrics.items()]}
                observations.append(observation)
                delta = counter_delta(old, metrics)
                idle = not any(metrics[(name, None)] for name in (
                    'vllm:num_requests_running', 'vllm:num_requests_waiting'))
                observation.update({'queue_drained': idle,
                    'native_deltas': [{'name': key[0], 'reason': key[1], 'value': value}
                                      for key, value in delta.items()]})
                if len(observations) > 64:
                    raise ValueError('group-external native publication trace cap')
                if idle and terminal_delta(delta, group['terminal_states']):
                    break
                time.sleep(0.25)
            else:
                raise TimeoutError('native completed-count publication/queue drain missing')
        finally:
            save(self.root / (label + '-native-publication.json'), observations,
                 256 * 1024)
            if raw is not None:
                with (self.root / (label + '-after-metrics.txt')).open('x') as handle:
                    handle.write(raw)
        completed_epoch = max(row['finished_epoch'] for row in results)
        e7 = self.stable(label + '-after', completed_epoch)
        identity = self.obs.identity()
        if time.perf_counter() >= deadline:
            raise TimeoutError('matrix deadline during group-external after guard')
        if identity['properties']['InvocationID'] != before['identity']['properties']['InvocationID']:
            raise ValueError('group actual invocation changed')
        vectors = e7_deltas(before['E7'], e7)
        statuses = [row['status'] for row in results]
        checks = {
            'requests_ok': all(row['status'] in ('COMPLETE', 'EXPECTED_CANCELLED_NOT_COMPLETED') for row in results),
            'input_tokens': all(len(packet['body']['prompt']) == row['prompt_tokens']
                                for packet, row in zip(group['requests'], results)),
            'output_tokens': all(row['status'] != 'COMPLETE' or len(row['output_token_ids']) == packet['body']['max_tokens']
                                 for packet, row in zip(group['requests'], results)),
            'terminal_states': statuses == group['terminal_states'],
            'queue_drained': True,  # Actual idle() above; never a missing adapter default.
            'no_restart': identity['properties']['NRestarts'] == '0',
            'original_route_counters_valid': bool(vectors
                and vectors[0]['hook_steps'] > 0 and terminal_delta(delta, statuses)),
        }
        if 'actual_zero_prefix_cache_hits' in group['required_checks']:
            checks['actual_zero_prefix_cache_hits'] = delta[('vllm:prefix_cache_hits_total', None)] == 0
        if 'prime_ok' in group['required_checks']:
            checks['prime_ok'] = prime is not None and prime['status'] == 'COMPLETE'
        if 'actual_shared_prefix_cache_hits' in group['required_checks']:
            checks['actual_shared_prefix_cache_hits'] = delta[('vllm:prefix_cache_hits_total', None)] > 0
        if 'exact_cancel_cut' in group['required_checks']:
            checks['exact_cancel_cut'] = all(exact_cancel_cut(packet, row)
                for packet, row in zip(group['requests'], results))
        if checks.keys() != set(group['required_checks']) or not all(checks.values()):
            raise ValueError('frozen group check missing/failed')
        value = {'identity': identity, 'E7': e7, 'full_E7_deltas': vectors,
            'metrics': [{'name': k[0], 'reason': k[1], 'value': v} for k, v in metrics.items()],
            'metric_deltas': [{'name': k[0], 'reason': k[1], 'value': v} for k, v in delta.items()],
            'checks': checks, 'scope': 'actual original telemetry, only outside timed request groups',
            'engine_cancel_ack': 'UNVERIFIED' if 'exact_cancel_cut' in checks else 'NOT_APPLICABLE',
            'engine_cancelled_count': None,
            'cancel_scope': 'Frozen client two-positive cut+native close/reap+queue drain; no scheduler ACK or cancellation benefit claim.'}
        save(self.root / (label + '-after.json'), value)
        return value, checks
