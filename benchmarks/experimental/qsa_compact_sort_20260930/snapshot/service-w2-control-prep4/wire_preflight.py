"""Three real requests/urllib3 cases on a dedicated synthetic loopback port.

The only URL adaptation is explicit and recorded. The model endpoint 8201 is
never contacted. This uses the current worker, unchanged frozen SSE parser and
actual process deadline/close/reap path. It is not model or W2 timing evidence.
"""
import argparse
import hashlib
import http.server
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frozen import module, require, PINS
from io_tools import save, sha, tree_bytes
from transport_deadline import Pump, PY, worker_environment

ROOT = Path(__file__).resolve().parent


def runtime():
    expected = json.loads((ROOT / 'runtime-pins.json').read_text())
    import requests
    import urllib3
    if (requests.__version__ != expected['requests_version']
            or urllib3.__version__ != expected['urllib3_version']
            or not sys.version.startswith('3.12.13 ')
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'torch' in sys.modules):
        raise ValueError('actual deployed CPU-only HTTP runtime differs')
    checked = []
    for row in expected['files']:
        item = importlib.import_module(row['module'])
        path = Path(item.__file__).resolve()
        if str(path) != row['path'] or sha(path) != row['sha256']:
            raise ValueError('deployed native HTTP source pin changed: ' + row['module'])
        checked.append({'module': row['module'], 'path': str(path), 'sha256': sha(path)})
    for name in PINS:
        require(name)
    return {'requests': requests.__version__, 'urllib3': urllib3.__version__,
            'python': sys.version, 'executable': sys.executable, 'source_pins': checked,
            'CUDA_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
            'Torch_imported': 'torch' in sys.modules}


def packet(name):
    body = {'request_id': 'W2-wire-' + name, 'prompt': [1, 2, 3],
            'stream': True, 'n': 1, 'best_of': 1, 'return_token_ids': True,
            'stream_options': {'include_usage': True}, 'max_tokens': 4,
            'ignore_eos': True}
    return {'body': body, 'header': {'X-Request-Id': body['request_id']},
        'body_sha256': hashlib.sha256(json.dumps(body, sort_keys=True,
                                   separators=(',', ':')).encode()).hexdigest(),
        'identity': module('stream_ids').ids(body, body['request_id']),
        'cancel_contract': {'positive_chunks': 2, 'cut': 'after-positive-chunk-count'}
                           if name == 'cancel' else None,
        'expected_terminal': 'EXPECTED_CANCELLED_NOT_COMPLETED'
                             if name == 'cancel' else 'COMPLETE'}


def child(port, packet_path, destination):
    import requests
    import transport_worker
    runtime()
    if not 1024 < port < 65536 or port in (8200, 8201):
        raise ValueError('synthetic dedicated ephemeral port required')
    original = requests.post
    def adapted(url, **kwargs):
        if url != 'http://127.0.0.1:8201/v1/completions':
            raise ValueError('unexpected worker endpoint before explicit adaptation')
        target = f'http://127.0.0.1:{port}/v1/completions'
        response = original(target, **kwargs)
        sock = response.raw._fp.fp.raw._sock
        if sock.getpeername() != ('127.0.0.1', port):
            response.close()
            raise ValueError('actual socket peer is not this synthetic server')
        save(Path(destination) / 'actual-socket-adapter.json', {
            'source_endpoint': url, 'actual_endpoint': target,
            'peername': list(sock.getpeername()), 'sockname': list(sock.getsockname()),
            'socket_fileno': sock.fileno(),
            'actual_socket_class': type(sock).__module__ + '.' + type(sock).__name__,
            'requests': requests.__version__, 'Torch_imported': 'torch' in sys.modules,
            'scope': 'Explicit synthetic URL redirect only; no request to source endpoint'})
        return response
    requests.post = adapted
    sys.argv = ['transport_worker.py', '--packet', str(packet_path),
                '--output', str(destination)]
    transport_worker.main()


def run(destination):
    root = Path(destination)
    root.mkdir(mode=0o700)
    identity = runtime()
    save(root / 'runtime.json', identity, 1024**2)
    stop = threading.Event()
    server_rows = []
    rows_lock = threading.Lock()
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'
        def log_message(self, *args):
            pass
        def do_POST(self):
            row = {'path': self.path, 'started_monotonic': time.perf_counter(),
                   'scope': 'synthetic server; no native engine/model'}
            try:
                length = int(self.headers['Content-Length'])
                if not 0 < length <= 16384 or self.path != '/v1/completions':
                    raise ValueError('synthetic bounded input')
                body = json.loads(self.rfile.read(length))
                name = body['request_id'].removeprefix('W2-wire-')
                if name not in ('normal', 'cancel', 'trickle-no-newline'):
                    raise ValueError('unregistered synthetic case')
                row.update(case=name, request_id=body['request_id'],
                           header_id=self.headers['X-Request-Id'])
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                if name == 'trickle-no-newline':
                    # Continuous recv keeps native read timeout from acting as
                    # an absolute deadline. No line is ever returned here.
                    while not stop.is_set():
                        self.wfile.write(b'x' * 64)
                        self.wfile.flush()
                        time.sleep(0.03)
                else:
                    response_id = 'cmpl-' + body['request_id']
                    for index in range(4):
                        self.wfile.write(b':' + b'c' * 4096 + b'\n')
                        self.wfile.write(('data: ' + json.dumps({'id': response_id,
                            'choices': [{'index': 0, 'token_ids': [10 + index],
                            'finish_reason': 'length' if index == 3 else None}]})
                            + '\n\n').encode())
                        self.wfile.flush()
                        time.sleep(0.03)
                    self.wfile.write(('data: ' + json.dumps({'id': response_id,
                        'choices': [], 'usage': {'prompt_tokens': 3,
                        'completion_tokens': 4, 'total_tokens': 7}})
                        + '\n\ndata: [DONE]\n\n').encode())
                    self.wfile.flush()
                row['terminal'] = 'synthetic-server-return'
            except (BrokenPipeError, ConnectionResetError) as error:
                row['terminal'] = type(error).__name__
            except BaseException as error:
                row.update(terminal='SYNTHETIC_SERVER_ERROR', error=repr(error))
            finally:
                row['finished_monotonic'] = time.perf_counter()
                with rows_lock:
                    server_rows.append(row)
    class CPUHTTPServer(http.server.ThreadingHTTPServer):
        daemon_threads = False
        block_on_close = True
    server = CPUHTTPServer(('127.0.0.1', 0), Handler)
    port = server.server_address[1]
    if port in (8200, 8201):
        server.server_close()
        raise ValueError('forbidden selected ephemeral port; no retry')
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={'poll_interval': 0.02})
    thread.start()
    outcomes = []
    try:
        for name in ('normal', 'cancel', 'trickle-no-newline'):
            path = root / (name + '.packet.json')
            save(path, packet(name))
            target = root / name
            pump = Pump()
            job = pump.add(path, target, argv=[PY, '-I', '-B', str(Path(__file__).resolve()),
                '--child-port', str(port), '--packet', str(path), '--output', str(target)],
                env=worker_environment())
            try:
                pump.wait_ready([job], time.perf_counter() + 5)
                limit = 0.6 if name == 'trickle-no-newline' else 5
                job.release(time.perf_counter() + limit)
                expected_error = None
                try:
                    pump.wait_finished([job], job.started + limit + 4)
                except RuntimeError as error:
                    expected_error = repr(error)
                    if name != 'trickle-no-newline':
                        raise
                if name == 'trickle-no-newline':
                    if (not expected_error or not job.reaped
                            or job.abort_reason != 'absolute request/group deadline'
                            or job.process.returncode not in (-15, -9)
                            or (target / 'result.json').exists()
                            or job.finished - job.started > 3):
                        raise ValueError('real continuous recv deadline/termination/reap not proven')
                else:
                    value = json.loads((target / 'result.json').read_text())
                    if (value['status'] != packet(name)['expected_terminal']
                            or len(value['output_token_ids']) != (2 if name == 'cancel' else 4)
                            or value['done'] is not (name == 'normal')):
                        raise ValueError('real wire frozen parser outcome differs')
                    observed = value['native_socket_descriptor_observations']
                    if name == 'normal' and (
                            observed['live_socket_timeout_updates'] <= 0
                            or observed['confirmed_native_closed_buffer_iterations'] <= 0
                            or len(value['finishes']) != 1 or len(value['usages']) != 1
                            or value['usages'][0]['usage'] != {
                                'prompt_tokens': 3, 'completion_tokens': 4,
                                'total_tokens': 7}):
                        raise ValueError('native live-FD and closed-buffer EOF branches not both observed')
                socket = json.loads((target / 'actual-socket-adapter.json').read_text())
                if socket['peername'] != ['127.0.0.1', port]:
                    raise ValueError('actual native socket receipt missing')
                outcomes.append({'case': name, 'actual_child_exit': job.process.returncode,
                    'expected_deadline_failure': expected_error, 'reaped': job.reaped,
                    'elapsed_release_to_reap_s': job.finished - job.started,
                    'source_endpoint_contacted': False,
                    'process_exit_sha256': sha(target / 'process-exit.json'),
                    'actual_socket_sha256': sha(target / 'actual-socket-adapter.json')})
                if name != 'trickle-no-newline':
                    outcomes[-1]['native_socket_descriptor_observations'] = observed
            finally:
                pump.close()
    except BaseException as error:
        save(root / 'FAILURE.json', {'status': 'FAIL_SYNTHETIC_WIRE_NO_AUTO_RETRY',
                                   'error': repr(error)[:4096], 'completed_cases': outcomes})
        raise
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
        deadline = time.perf_counter() + 1
        while len(server_rows) < len(outcomes) and time.perf_counter() < deadline:
            time.sleep(0.01)
        save(root / 'server.json', {'address': ['127.0.0.1', port],
            'server_thread_ended': not thread.is_alive(), 'rows': server_rows,
            'source_endpoint_8201_never_contacted': True})
    if (thread.is_alive() or len(server_rows) != 3
            or any(row['terminal'] == 'SYNTHETIC_SERVER_ERROR' for row in server_rows)
            or tree_bytes(root) > 1024**2 or 'torch' in sys.modules):
        raise ValueError('synthetic cleanup/artifact/runtime gate failed')
    value = {'status': 'PASS_REAL_REQUESTS_SOCKET_SYNTHETIC_NOT_MODEL_W2',
        'cases': outcomes, 'synthetic_HTTP_requests': 3, 'model_HTTP_requests': 0,
        'native_engine_cancel_ack': 'UNVERIFIED', 'GPU_operations': 0,
        'artifact_bytes_before_receipt': tree_bytes(root),
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'server_and_request_processes_ended': True,
        'returned_line_bytes_cap_is_wire_hard_cap': False,
        'scope': 'Actual native socket access/EOF/two-positive client cut and absolute parent termination; not engine cancellation or model timing'}
    save(root / 'RESULT.json', value)
    print(json.dumps(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--child-port', type=int)
    parser.add_argument('--packet')
    args = parser.parse_args()
    if args.child_port is not None:
        child(args.child_port, args.packet, args.output)
    else:
        run(args.output)


if __name__ == '__main__':
    main()
