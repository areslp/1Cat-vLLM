"""Run the unchanged artifact scanner outside the request event loop.

Source preparation for a separate diagnostic, never modifies frozen run4.
The worker keeps the same 8GiB/count/trace caps and scan interval. Parent ticks
only consume small pipe receipts; lost, stale or failing monitors stop the run.
"""
import argparse
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parent.parent / 'service-w2-control-prep4'
RECEIPT_CAP = 65536


def emit(**value):
    print(json.dumps(value, sort_keys=True), flush=True)


def worker(root, trace, cpu):
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    sys.path.insert(0, str(SOURCE))
    from artifact_budget import BudgetMonitor, INTERVAL
    monitor = BudgetMonitor(root, trace)
    command = selectors.DefaultSelector()
    command.register(sys.stdin, selectors.EVENT_READ)
    try:
        while True:
            emit(event='SCAN_BEGIN', at=time.perf_counter(), pid=os.getpid())
            size = monitor.tick(force=True)
            emit(event='SCAN_END', at=monitor.last, bytes=size,
                 pid=os.getpid(), scan_s=monitor.max_scan_s)
            if command.select(INTERVAL):
                data = sys.stdin.readline(64)
                if data != 'STOP\n':
                    raise ValueError('owner disconnected or invalid command')
                break
    finally:
        monitor.close()
        command.close()


class AsyncBudgetMonitor:
    def __init__(self, root, trace, *, cpu=None, stale_s=5.0):
        self.trace = Path(trace)
        self.latest = self.scan_begin = None
        self.buffer = bytearray()
        self.stale_s = stale_s
        self.closed = False
        self.events = []
        argv = [sys.executable, '-I', '-B', str(Path(__file__).resolve()),
                '--worker', '--root', str(root), '--trace', str(trace)]
        if cpu is not None:
            argv += ['--cpu', str(cpu)]
        self.stderr_path = self.trace.with_suffix('.stderr')
        self.stderr = self.stderr_path.open('xb')
        try:
            self.process = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.stderr, start_new_session=True,
                env={'PATH': '/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES': '',
                     'PYTHONDONTWRITEBYTECODE': '1'})
        except BaseException:
            self.stderr.close()
            raise
        os.set_blocking(self.process.stdout.fileno(), False)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def _read(self, timeout=0):
        consumed = 0
        while self.selector.select(timeout):
            timeout = 0
            key = self.selector.get_key(self.process.stdout)
            data = os.read(key.fd, 4096)
            if not data:
                raise RuntimeError('artifact monitor ended unexpectedly')
            self.buffer.extend(data)
            consumed += len(data)
            if consumed > RECEIPT_CAP:
                raise ValueError('artifact monitor receipt drain cap')
            if len(self.buffer) > RECEIPT_CAP:
                raise ValueError('artifact monitor receipt cap')
            while b'\n' in self.buffer:
                raw, _, rest = self.buffer.partition(b'\n')
                self.buffer[:] = rest
                event = json.loads(raw)
                if event['pid'] != self.process.pid:
                    raise ValueError('artifact monitor PID mismatch')
                if event['event'] == 'SCAN_BEGIN':
                    self.scan_begin = event
                elif event['event'] == 'SCAN_END':
                    if not 0 <= event['bytes'] <= 8 * 1024**3:
                        raise ValueError('artifact budget exceeded')
                    self.latest = event
                else:
                    raise ValueError('unexpected artifact monitor event')
                self.events.append(event)
                if len(self.events) > 32768:
                    raise ValueError('artifact monitor event count cap')

    def tick(self, force=False):
        if self.closed:
            raise RuntimeError('closed artifact monitor')
        if force:
            # Outside request timing: require a scan completed after this call.
            # Ordinary ticks drain all queued receipts before checking freshness.
            requested = time.perf_counter()
            deadline = time.perf_counter() + self.stale_s
            while self.latest is None or self.latest['at'] < requested:
                if time.perf_counter() >= deadline:
                    raise TimeoutError('artifact monitor initial scan timeout')
                if self.process.poll() is not None:
                    raise RuntimeError('artifact monitor failed before admission')
                self._read(max(0, min(.05, deadline - time.perf_counter())))
        else:
            self._read()
        if self.process.poll() is not None:
            raise RuntimeError('artifact monitor failed')
        if self.latest is None or time.perf_counter()-self.latest['at'] > self.stale_s:
            raise TimeoutError('artifact monitor receipt stale')
        if self.stderr_path.stat().st_size > RECEIPT_CAP:
            raise ValueError('artifact monitor stderr cap')
        return self.latest['bytes']

    def close(self):
        if self.closed:
            return
        outcome = {'pid': self.process.pid, 'signal_sent': None}
        code = None
        try:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write(b'STOP\n')
                    self.process.stdin.flush()
                except BrokenPipeError:
                    outcome['stop_pipe_already_closed'] = True
                finally:
                    try:
                        self.process.stdin.close()
                    except BrokenPipeError:
                        outcome['stop_pipe_already_closed'] = True
            try:
                code = self.process.wait(timeout=self.stale_s)
            except subprocess.TimeoutExpired:
                outcome['signal_sent'] = 'TERM'
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    code = self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    outcome['signal_sent'] = 'KILL'
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    code = self.process.wait(timeout=1)
            outcome.update(actual_child_exit=code, reaped=True)
        finally:
            if code is None:
                # Even an interrupted wait or vanished process-group race must
                # end with an actual bounded reap attempt before closing pipes.
                try:
                    if self.process.poll() is None:
                        outcome['signal_sent'] = 'KILL'
                        try:
                            os.killpg(self.process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    code = self.process.wait(timeout=1)
                    outcome.update(actual_child_exit=code, reaped=True)
                except BaseException as error:
                    outcome.update(reaped=False, cleanup_error=repr(error))
            self.closed = outcome.get('reaped', False)
            self.selector.close()
            self.process.stdout.close()
            self.stderr.close()
            with self.trace.with_suffix('.exit.json').open('x') as stream:
                json.dump(outcome, stream)
        if code != 0 or outcome['signal_sent'] is not None:
            raise RuntimeError('artifact monitor did not close successfully')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--trace', required=True)
    parser.add_argument('--cpu', type=int)
    args = parser.parse_args()
    worker(args.root, args.trace, args.cpu)
