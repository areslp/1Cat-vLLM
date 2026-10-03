"""Real loopback/socket/child checks; synthetic SSE, no model/8201 request.

For native cgroup observation start --server-only outside the client unit and
pass its port. Auto-server local mode is explicitly not client cgroup proof.
"""
import argparse
import hashlib
import http.server
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
import transport_deadline_context as transport
from io_tools import save, sha, tree_bytes


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == 'darwin' else 1024)


def runtime():
    import requests
    import urllib3
    if (requests.__version__ != '2.34.2' or urllib3.__version__ != '2.7.0'
            or not sys.version.startswith('3.12.13 ') or 'torch' in sys.modules
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise ValueError('exact CPU HTTP library/profile required')
    return {'python': sys.version, 'executable': sys.executable,
            'requests': requests.__version__, 'urllib3': urllib3.__version__,
            'requests_source': sha(requests.__file__), 'urllib3_source': sha(urllib3.__file__)}


def child(port, packet, output, mode):
    import requests
    spec = importlib.util.spec_from_file_location('CPU_owned_worker', HERE / 'transport_worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    runtime()
    if not 1024 < port < 65536 or port in (8200, 8201):
        raise ValueError('dedicated loopback fixture port required')
    original = requests.post

    def routed(url, **kwargs):
        if url != 'http://127.0.0.1:8201/v1/completions':
            raise ValueError('unexpected production URL before CPU route')
        kwargs['headers'] = dict(kwargs['headers'], **{'X-CPU-Mode': mode})
        response = original(f'http://127.0.0.1:{port}/v1/completions', **kwargs)
        socket = response.raw._fp.fp.raw._sock
        if socket.getpeername() != ('127.0.0.1', port):
            raise ValueError('actual socket is not dedicated loopback fixture')
        save(Path(output) / 'socket-route.json', {
            'source8201_contacted': False, 'peer': list(socket.getpeername()),
            'source_url': url, 'mode': mode, 'scope': 'CPU fixture endpoint/header only'})
        return response

    requests.post = routed
    sys.argv = ['transport_worker.py', '--packet', packet, '--output', output]
    worker.main()


def serve(output, seconds):
    output.mkdir(mode=0o700)
    runtime()
    errors, lock = [], threading.Lock()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'

        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                count = int(self.headers['Content-Length'])
                if not 0 < count <= 4 * 1024**2 or self.path != '/v1/completions':
                    raise ValueError('finite synthetic HTTP input')
                body = json.loads(self.rfile.read(count))
                mode = self.headers['X-CPU-Mode']
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                if mode in ('line-over', 'received-over', 'count-over'):
                    if mode == 'line-over':
                        self.wfile.write(b':' + b'x' * (2 * 1024**2) + b'\n')
                    elif mode == 'received-over':
                        for _ in range(9):
                            self.wfile.write(b':' + b'x' * (2 * 1024**2 - 1) + b'\n')
                            self.wfile.flush()
                    else:
                        self.wfile.write(b':\n' * 4097)
                    self.wfile.flush()
                    return
                if mode != 'normal':
                    raise ValueError('unregistered synthetic response mode')
                response_id = 'cmpl-' + body['request_id']
                tokens = body['max_tokens']
                for index in range(tokens):
                    choice = {'index': 0, 'text': 'x', 'logprobs': None,
                        'finish_reason': 'length' if index == tokens - 1 else None,
                        'stop_reason': None,
                        'prompt_token_ids': body['prompt'] if index == 0 else None,
                        'token_ids': [10000 + index]}
                    data = {'id': response_id, 'object': 'text_completion',
                        'created': 1790964841, 'model': 'flash-next', 'choices': [choice]}
                    self.wfile.write(b'data: ' + json.dumps(data, ensure_ascii=False,
                        separators=(',', ':')).encode() + b'\n\n')
                    self.wfile.flush()
                    time.sleep(0.0005)
                data = {'id': response_id, 'choices': [], 'usage': {
                    'prompt_tokens': len(body['prompt']), 'completion_tokens': tokens,
                    'total_tokens': len(body['prompt']) + tokens}}
                self.wfile.write(b'data: ' + json.dumps(data).encode()
                                  + b'\n\ndata: [DONE]\n\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected negative client closure is not model cancellation.
            except BaseException as error:
                with lock:
                    errors.append(repr(error))

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = False
    server.block_on_close = True
    port = server.server_address[1]
    if port in (8200, 8201):
        raise ValueError('forbidden selected server port; no retry')
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.05})
    thread.start()
    save(output / 'SERVER.json', {'port': port, 'pid': os.getpid(), 'seconds': seconds,
        'synthetic_only': True, 'outside_client_unit_required_for_native_resource_proof': True})
    print(json.dumps({'event': 'SERVER_READY', 'port': port}), flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    def owner_input():
        if sys.stdin.readline(32) == 'STOP\n':
            stop.set()

    threading.Thread(target=owner_input, daemon=True).start()
    try:
        stop.wait(seconds)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        save(output / 'CLOSED.json', {'thread_ended': not thread.is_alive(),
            'errors': errors, 'peak_RSS_bytes': rss()})
    if errors or thread.is_alive():
        raise ValueError('CPU server error/cleanup failure')


def cgroup(output):
    if sys.platform != 'linux':
        value = {'status': 'UNAVAILABLE_NON_LINUX_NOT_NATIVE_CGROUP_PROOF',
                 'platform': sys.platform}
    else:
        raw = Path('/proc/self/cgroup').read_text()
        lines = raw.strip().splitlines()
        if len(lines) != 1 or not lines[0].startswith('0::/'):
            raise ValueError('unified actual CPU client cgroup required')
        path = Path('/sys/fs/cgroup') / lines[0][4:]
        names = ('memory.max', 'memory.current', 'memory.peak', 'memory.events',
                 'memory.stat', 'memory.swap.current', 'memory.swap.max',
                 'cpuset.cpus.effective')
        files = {name: (path / name).read_text().strip() for name in names}
        value = {'status': 'ACTUAL_NATIVE_CPU_CLIENT_CGROUP_READBACK',
                 'proc_self_cgroup': raw, 'path': str(path), 'files': files,
                 '512MiB_peak_pass': int(files['memory.peak']) <= 512 * 1024**2,
                 'not_run3_peak_composition': True}
    save(output / 'client-cgroup.json', value, 256 * 1024)
    return value


def client(output, port, full, precharge=None):
    output.mkdir(mode=0o700)
    version = runtime()
    matrix_dir = BASE / 'context-qsa-prep1'
    matrix = json.loads((matrix_dir / 'matrix.frozen.json').read_text())
    selected = {'context-65536-greedy-c8', 'context-131072-greedy-c1',
                'context-131072-greedy-c4', 'context-261632-greedy-c2'}
    pins = [p for p in matrix['groups'] if p['arm'] == 'A0'
            and (full or p['ordinal'] == 0 and p['row_id'] in selected)]
    if len(pins) != (38 if full else 4):
        raise ValueError('fixed CPU inventory differs')
    if not full:
        pins.sort(key=lambda p: (-int(p['row_id'].split('-')[1]), -p['requests']))
    cases, primary = [], None
    began = time.perf_counter()

    def jobs(packets, destination, mode='normal'):
        pump = transport.Pump()
        entries = []
        try:
            for index, packet in enumerate(packets):
                path = destination / f'packet-{index:02d}.json'
                target = destination / f'child-{index:02d}'
                save(path, packet, 4 * 1024**2)
                job = pump.add(path, target, argv=[sys.executable, '-I', '-B', __file__,
                    '--child-port', str(port), '--packet', str(path), '--output', str(target),
                    '--mode', mode])
                entries.append((job, packet))
            pump.wait_ready([j for j, _ in entries], time.perf_counter() + 30)
            for job, _ in entries:
                job.release(time.perf_counter() + 60)
            failure = None
            try:
                pump.wait_finished([j for j, _ in entries], time.perf_counter() + 65)
            except RuntimeError as error:
                if mode == 'normal':
                    raise
                failure = repr(error)
            rows = []
            for job, packet in entries:
                closed = json.loads((job.output / 'native-close.json').read_text())
                if (not job.reaped or closed['worker_path'] != str(HERE / 'transport_worker.py')
                        or closed['worker_sha256'] != sha(HERE / 'transport_worker.py')
                        or closed['LINE_CAP'] != 2 * 1024**2):
                    raise ValueError('actual owned worker/limit/reap differs')
                row = {'path': str(job.output), 'peak_RSS_bytes': closed['peak_RSS_bytes'],
                       'tree_bytes': tree_bytes(job.output), 'actual_child_exit': job.process.returncode}
                if mode == 'normal':
                    value = json.loads((job.output / 'result.json').read_text())
                    if (len(value['output_token_ids']) != packet['body']['max_tokens']
                            or value['prompt_tokens'] != len(packet['body']['prompt'])):
                        raise ValueError('real frozen parser response/usage extent differs')
                    row.update(received_bytes=value['raw_received_bytes'],
                               raw_record_bytes=value['raw_response']['bytes'])
                else:
                    value = json.loads((job.output / 'FAILURE.json').read_text())
                    rejected = json.loads((job.output / 'rejected-line.json').read_text())
                    expected = {'line-over': 'line_bytes', 'received-over': 'received_bytes',
                                'count-over': 'line_count'}[mode]
                    if (not failure or value['stream_limit_failure']['violated'] != [expected]
                            or rejected['stream_limits']['violated'] != [expected]
                            or rejected['prefix_bytes'] > 65536
                            or len(rejected['prefix_hex']) != rejected['prefix_bytes'] * 2
                            or row['tree_bytes'] > 34 * 1024**2
                            or job.process.returncode == 0 or (job.output / 'result.json').exists()):
                        raise ValueError('strict rejection gate/bounded raw prefix not proven')
                    row['rejection'] = rejected
                rows.append(row)
            return rows
        finally:
            pump.close()

    try:
        if precharge is not None:
            source = Path(precharge)
            size = tree_bytes(source)
            if not source.is_dir() or size > 256 * 1024**2:
                raise ValueError('finite owned prefix precharge max256MiB')
            shutil.copytree(source, output / 'capacity-prefix')
            save(output / 'PRECHARGE.json', {'source': str(source), 'bytes': size,
                'copied_bytes': tree_bytes(output / 'capacity-prefix'),
                'capacity_stress_only_not_performance_or_run3_peak_composition': True})
        for number, pin in enumerate(pins):
            path = matrix_dir / pin['path']
            if sha(path) != pin['sha256']:
                raise ValueError('original frozen body/salt bytes differ')
            group = json.loads(path.read_text())
            target = output / f'{number:02d}-{pin["row_id"]}-repeat{pin["ordinal"]}'
            target.mkdir()
            save(target / 'frozen-group.json', group, 8 * 1024**2)
            prime_rows = []
            for index, packet in enumerate(group['primes']):
                directory = target / f'prime-{index:02d}'
                directory.mkdir()
                prime_rows += jobs([packet], directory)
            rows = jobs(group['requests'], target)
            case = {'row_id': pin['row_id'], 'repeat': pin['ordinal'],
                'input_tokens': group['input_tokens'], 'concurrency': pin['requests'],
                'primes': prime_rows, 'main': rows, 'parent_peak_RSS_bytes': rss(),
                'sum_parent_plus_individual_worker_peaks': rss() + sum(r['peak_RSS_bytes'] for r in rows)}
            save(target / 'CPU-GROUP.json', case)
            cases.append(case)
            if time.perf_counter() - began > 300:
                raise TimeoutError('bounded CPU suite deadline')
        packet = group['requests'][0]
        for mode in ('line-over', 'received-over', 'count-over'):
            target = output / mode
            target.mkdir()
            rows = jobs([packet], target, mode)
            cases.append({'case': mode, 'rejection': rows})
    except BaseException as error:
        primary = error
        save(output / 'FAILURE.json', {'error': repr(error), 'closed_cases': len(cases),
             'status': 'CPU_WIRE_FAILED_NO_RETRY_NOT_MODEL_RESULT'})
        raise
    finally:
        # Query/readback only, after every owned job's close/reap attempt.
        cgroup(output)
    value = {'status': 'PASS_REAL_LOOPBACK_TRANSPORT_NOT_MODEL_OR_512_GUARANTEE',
        'runtime': version, 'cases': cases, 'full_A0_transport_inventory': full,
        'groups': len(pins), 'parent_peak_RSS_bytes': rss(), 'tree_bytes': tree_bytes(output),
        'elapsed_s': time.perf_counter() - began, 'original_worker_sha': sha(matrix_dir / 'transport_worker.py'),
        'owned_worker_sha': sha(HERE / 'transport_worker.py'), 'Pump_SHA': sha(HERE / 'transport_deadline_context.py'),
        'worker_default_path': str(transport.WORKER), 'frozen_parser_changed': False,
        'endpoint_only_CPU_adapter': True, 'source8201_contacted': False,
        'server_excluded_from_client_RSS_sum': True, 'not_actual_run3_peak_composition': True,
        'not_GPU_or_candidate_or_performance_admission': True}
    save(output / 'RESULT.json', value, 4 * 1024**2)
    print(json.dumps({k: v for k, v in value.items() if k != 'cases'}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--server-only', action='store_true')
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--server-port', type=int)
    parser.add_argument('--child-port', type=int)
    parser.add_argument('--packet')
    parser.add_argument('--mode', default='normal')
    parser.add_argument('--full-A0', action='store_true')
    parser.add_argument('--precharge-tree', type=Path)
    args = parser.parse_args()
    if args.child_port is not None:
        child(args.child_port, args.packet, str(args.output), args.mode)
    elif args.server_only:
        serve(args.output, args.seconds)
    elif args.server_port is not None:
        client(args.output, args.server_port, args.full_A0, args.precharge_tree)
    else:
        raise ValueError('explicit independent loopback server port required')


if __name__ == '__main__':
    main()
