"""One bounded requests/urllib3 stream, owned by an external hard-deadline job.

No server, model, Torch or CUDA imports. READY precedes all network activity.
The parent must terminate/kill/reap the process at the absolute deadline; socket
timeout alone deliberately is not claimed to bound iter_lines' multiple recv.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

# Python -I avoids inherited experiment/PYTHONPATH shims. Only this owned package
# is added, then frozen imports separately check their complete source identity.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frozen import module
from io_tools import (LINE_CAP, LINE_COUNT_CAP, REQUEST_RECEIVED_CAP,
                      REQUEST_TREE_CAP, save, sha, tree_bytes)


def event(kind, **values):
    print(json.dumps({'event': kind, **values}, sort_keys=True), flush=True)


def refresh_timeout(response, socket, initial_fd, remaining, observations):
    """Only native confirmed EOF may leave buffered iter_lines lines to consume.

    requests.models.iter_lines yields all split lines from one already-read
    chunk. urllib3 may close its HTTPResponse/socket before yielding that final
    chunk. A negative FD alone never establishes EOF; other settimeout errors
    remain failures. The independent parent deadline still bounds next().
    """
    descriptor = socket.fileno()
    if type(initial_fd) is not int or initial_fd < 0 or type(descriptor) is not int:
        raise ValueError('native original socket descriptor invalid')
    if descriptor == initial_fd:
        socket.settimeout(remaining)  # No catch or permissive EBADF fallback.
        observations['live_socket_timeout_updates'] += 1
    elif descriptor == -1 and response.raw.closed is True:
        observations['confirmed_native_closed_buffer_iterations'] += 1
    else:
        raise ValueError('socket changed or closed without confirmed native EOF')


def execute(packet, output, deadline):
    import requests
    from urllib3.util import Timeout
    parser, identities = module('sse_parser'), module('stream_ids')
    body = json.dumps(packet['body'], sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(body).hexdigest() != packet['body_sha256']:
        raise ValueError('frozen request body changed')
    started = time.perf_counter()
    expires = min(deadline, started + 120)
    response, frames, received, lines, positives = None, [], 0, 0, 0
    terminal, close_ok = 'eof', False
    descriptor_observations = {'live_socket_timeout_updates': 0,
                               'confirmed_native_closed_buffer_iterations': 0}
    raw_path = output / 'raw-lines.jsonl'
    try:
        remaining = expires - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError('expired before HTTP submit')
        response = requests.post('http://127.0.0.1:8201/v1/completions', data=body,
            headers={**packet['header'], 'Content-Type': 'application/json'},
            stream=True, timeout=Timeout(total=remaining,
                connect=min(10, remaining), read=remaining))
        if response.status_code != 200:
            payload = response.raw.read(65537)
            save(output / 'HTTP-error.json', {'status': response.status_code,
                'body_hex': payload[:65536].hex(), 'truncated': len(payload) > 65536})
            raise RuntimeError('unexpected HTTP status ' + str(response.status_code))
        socket = response.raw._fp.fp.raw._sock  # Exact pinned native adapter.
        initial_fd = socket.fileno()
        iterator = iter(response.iter_lines(chunk_size=4096))
        with raw_path.open('x') as raw:
            while True:
                remaining = expires - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError('absolute per-request/global deadline')
                refresh_timeout(response, socket, initial_fd, remaining,
                                descriptor_observations)
                try:
                    line = next(iterator)
                except StopIteration:
                    break
                elapsed = time.perf_counter() - started
                if not isinstance(line, bytes):
                    raise TypeError('requests iter_lines changed type')
                received += len(line)
                lines += 1
                if (elapsed > expires - started or len(line) > LINE_CAP
                        or received > REQUEST_RECEIVED_CAP or lines > LINE_COUNT_CAP):
                    raise ValueError('stream time/line/received/line-count cap')
                raw.write(json.dumps({'elapsed_s': elapsed, 'raw_hex': line.hex()}) + '\n')
                raw.flush()  # Retain partial bytes before possible process kill.
                if not line.startswith(b'data:'):
                    continue
                text = line[5:].strip().decode('utf-8', errors='strict')
                frames.append({'elapsed_s': elapsed, 'data': text})
                if len(frames) > packet['body']['max_tokens'] + 16:
                    raise ValueError('frozen frame extent exceeded')
                if text == '[DONE]':
                    continue
                chunk = json.loads(text)
                identities.check_chunk(chunk, packet['identity'])
                if any(choice.get('token_ids') for choice in chunk.get('choices', [])):
                    positives += 1
                    if positives == 1:
                        event('FIRST_POSITIVE', observed_monotonic=time.perf_counter())
                    cancel = packet['cancel_contract']
                    if cancel and positives == cancel['positive_chunks']:
                        terminal = 'client_cancelled'
                        break
        result = parser.parse(packet['body'], packet['header']['X-Request-Id'], frames,
            terminal=terminal, elapsed_s=time.perf_counter() - started,
            expected_cancel=packet['cancel_contract'] is not None,
            cancel_contract=packet['cancel_contract'])
        if result['status'] != packet['expected_terminal']:
            raise ValueError('unexpected terminal outcome')
        # Every original frame remains in raw-lines. Avoid embedding it again in
        # result and again in group-result; parser/metrics/tokens stay unchanged.
        result.pop('raw_events')
        result.update(started_monotonic=started, finished_monotonic=time.perf_counter(),
            prompt_tokens=len(packet['body']['prompt']),
            raw_received_bytes=received, body_sha256=packet['body_sha256'],
            raw_response={'path': str(raw_path), 'sha256': sha(raw_path),
                          'bytes': raw_path.stat().st_size},
            timing_definition='unchanged frozen sse_parser/perf38')
        result['native_socket_descriptor_observations'] = descriptor_observations
        save(output / 'result.json', result, 256 * 1024)
        if tree_bytes(output) > REQUEST_TREE_CAP:
            raise ValueError('request tree byte budget')
    except BaseException as error:
        save(output / 'FAILURE.json', {'status': 'FAILED_PARTIAL_NOT_COMPLETED',
            'error': repr(error)[:2048], 'elapsed_s': time.perf_counter() - started,
            'received_data_frames': len(frames), 'raw_received_bytes': received,
            'body_sha256': packet['body_sha256'],
            'native_socket_descriptor_observations': descriptor_observations,
            'raw_response': {'path': str(raw_path), 'exists': raw_path.exists(),
                             'sha256': sha(raw_path) if raw_path.exists() else None}}, 65536)
        raise
    finally:
        if response is not None:
            response.close()
        close_ok = True
        save(output / 'native-close.json', {'response_close_returned': close_ok,
            'response_existed': response is not None, 'pid': os.getpid(),
            'closed_monotonic': time.perf_counter(),
            'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                (1 if sys.platform == 'darwin' else 1024),
            'torch_imported': 'torch' in sys.modules,
            'CVD': os.environ.get('CUDA_VISIBLE_DEVICES')}, 65536)


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--packet', required=True)
    args.add_argument('--output', required=True)
    options = args.parse_args()
    path, output = Path(options.packet), Path(options.output)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('packet byte cap')
    packet = json.loads(path.read_text())
    output.mkdir(mode=0o700)
    save(output / 'request.json', packet)
    # Dependency imports are outside timed release and contain no model modules.
    import requests
    import urllib3
    module('sse_parser')
    event('READY', pid=os.getpid(), requests=requests.__version__,
          urllib3=urllib3.__version__, ready_monotonic=time.perf_counter())
    command = json.loads(sys.stdin.readline(1025))
    if set(command) != {'event', 'deadline_monotonic'} or command['event'] != 'START':
        raise ValueError('one exact START required')
    deadline = command['deadline_monotonic']
    if type(deadline) not in (int, float) or deadline <= time.perf_counter():
        raise ValueError('invalid absolute deadline')
    execute(packet, output, deadline)


if __name__ == '__main__':
    main()
