"""Finite owner of request processes; timeout means kill, reap and STOP.

No future timeout is mistaken for thread/socket termination. All network streams
stay in children with isolated Python startup. Process readiness is before group
release; the external parent continues supervising multi-recv iter_lines calls.
"""
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

from dependencies import activate
activate()
from io_tools import save

PY = '/home/l/work/1Cat-vLLM/.venv/bin/python'
WORKER = Path(__file__).resolve().parent / 'transport_worker.py'
STDERR_CAP = 65536
EVENT_CAP = 4096


def worker_environment():
    # Neither numeric/admission configs nor production PYTHONPATH are inherited.
    return {'PATH': '/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES': '',
            'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1', 'LANG': 'C.UTF-8'}


class Job:
    def __init__(self, argv, output, env):
        self.output = Path(output)
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, close_fds=True, env=env)
        self.pid = self.process.pid
        self.created = time.perf_counter()
        self.ready = self.started = self.first_positive = None
        self.ready_observed = None
        self.deadline = self.finished = self.abort_time = None
        self.abort_reason = None
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.events = []
        self.eof = set()
        self.reaped = False

    def release(self, deadline):
        if self.ready is None or self.started is not None or self.process.poll() is not None:
            raise ValueError('only one live READY process can release')
        self.started = time.perf_counter()
        self.deadline = min(deadline, self.started + 600)
        if self.deadline <= self.started:
            raise TimeoutError('group deadline expired before release')
        self.process.stdin.write((json.dumps({'event': 'START',
            'deadline_monotonic': self.deadline}) + '\n').encode())
        self.process.stdin.flush()
        self.process.stdin.close()

    def terminate(self, reason):
        if self.abort_time is None:
            self.abort_time = time.perf_counter()
            self.abort_reason = reason
        self.signal_owned(signal.SIGTERM)

    def signal_owned(self, sig):
        if self.process.poll() is not None:
            return
        # start_new_session owns this exact PGID. Only this Popen's newly
        # confirmed terminal state can excuse a getpgid/killpg exit race.
        try:
            if os.getpgid(self.pid) != self.pid:
                raise RuntimeError('owned process group changed')
            os.killpg(self.pid, sig)
        except ProcessLookupError as error:
            if self.process.poll() is None:
                # ESRCH can precede waitpid visibility. Only this exact child
                # returning from a bounded native wait confirms termination.
                try:
                    self.process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    raise error


class Pump:
    def __init__(self, *, budget_check=None):
        self.selector = selectors.DefaultSelector()
        self.jobs = []
        self.budget_check = budget_check

    def add(self, packet_path, output, *, argv=None, env=None):
        if len([j for j in self.jobs if not j.reaped]) >= 8:
            raise ValueError('more than eight in-flight request processes')
        argv = argv or [PY, '-I', '-B', str(WORKER),
                        '--packet', str(packet_path), '--output', str(output)]
        job = Job(argv, output, worker_environment() if env is None else env)
        for name in ('stdout', 'stderr'):
            pipe = getattr(job.process, name)
            os.set_blocking(pipe.fileno(), False)
            self.selector.register(pipe, selectors.EVENT_READ, (job, name))
        self.jobs.append(job)
        return job

    def read_event(self, job, raw):
        if len(raw) > EVENT_CAP:
            raise ValueError('worker event byte cap')
        row = json.loads(raw)
        if row['event'] == 'READY':
            if job.ready is not None or job.started is not None or row['pid'] != job.pid:
                raise ValueError('duplicate/stale/wrong-PID READY')
            job.ready = row
            job.ready_observed = time.perf_counter()
        elif row['event'] == 'FIRST_POSITIVE':
            if job.started is None or job.first_positive is not None:
                raise ValueError('early/duplicate FIRST_POSITIVE')
            observed = row['observed_monotonic']
            if type(observed) not in (int, float) or not job.started <= observed <= time.perf_counter():
                raise ValueError('positive event time outside actual submission')
            job.first_positive = row
        else:
            raise ValueError('unexpected worker event')
        job.events.append(row)
        if len(job.events) > 2:
            raise ValueError('worker event-count cap')

    def tick(self, seconds=0.1):
        if self.budget_check is not None:
            self.budget_check()
        for key, _ in self.selector.select(min(seconds, 0.1)):
            job, name = key.data
            data = os.read(key.fileobj.fileno(), 4096)
            if not data:
                self.selector.unregister(key.fileobj)
                key.fileobj.close()
                job.eof.add(name)
                continue
            buffer = getattr(job, name)
            buffer.extend(data)
            if name == 'stderr':
                if len(buffer) > STDERR_CAP:
                    raise ValueError('stderr cap; preserve first bytes and abort')
            else:
                while b'\n' in buffer:
                    line, _, rest = buffer.partition(b'\n')
                    buffer[:] = rest
                    self.read_event(job, line)
                if len(buffer) > EVENT_CAP:
                    raise ValueError('partial stdout event byte cap')
        now = time.perf_counter()
        for job in self.jobs:
            code = job.process.poll()
            if job.deadline is not None and now >= job.deadline and code is None:
                job.terminate('absolute request/group deadline')
            if job.abort_time is not None and code is None:
                if now - job.abort_time >= 1:
                    job.signal_owned(signal.SIGKILL)
                if now - job.abort_time > 3:
                    raise TimeoutError('owned child did not exit after TERM/KILL; cgroup cleanup required')
            if code is not None and job.eof == {'stdout', 'stderr'} and not job.reaped:
                if job.stdout:
                    raise ValueError('worker stdout incomplete JSON line')
                job.process.wait(timeout=0)
                if not job.process.stdin.closed:
                    job.process.stdin.close()
                job.reaped = True
                job.finished = now
                # Process exit is the close proof; no result file alone can pass.
                if not job.output.is_dir():
                    job.output.mkdir(mode=0o700)
                with (job.output / 'stderr.log').open('xb') as handle:
                    handle.write(job.stderr[:STDERR_CAP])
                save(job.output / 'process-exit.json', {
                    'pid': job.pid, 'actual_child_exit': code,
                    'ready': job.ready, 'release_monotonic': job.started,
                    'ready_parent_observed_monotonic': job.ready_observed,
                    'finished_monotonic': now, 'created_monotonic': job.created,
                    'deadline_monotonic': job.deadline,
                    'abort_reason': job.abort_reason, 'events': job.events,
                    'process_reaped': True, 'stderr_bytes': len(job.stderr),
                    'no_thread_or_future_completion_assumption': True})

    def wait_ready(self, jobs, deadline):
        while not all(job.ready is not None for job in jobs):
            if time.perf_counter() >= deadline:
                raise TimeoutError('untimed preparation READY deadline')
            self.tick()
            if any(job.reaped for job in jobs):
                raise RuntimeError('worker exited before READY/release')

    def wait_finished(self, jobs, deadline):
        while not all(job.reaped for job in jobs):
            if time.perf_counter() >= deadline:
                raise TimeoutError('supervisor global deadline including cleanup')
            self.tick()
        for job in jobs:
            if job.abort_reason or job.process.returncode != 0:
                raise RuntimeError('actual request process failed or was terminated')
            if not (job.output / 'native-close.json').is_file():
                raise ValueError('successful response.close receipt missing')
            row = json.loads((job.output / 'native-close.json').read_text())
            if (row['response_close_returned'] is not True or row['pid'] != job.pid
                    or type(row['closed_monotonic']) not in (int, float)
                    or not job.started <= row['closed_monotonic'] <= job.deadline):
                raise ValueError('response close not proven within absolute deadline')

    def close(self):
        # Called on every group exit. Cleanup can fail, but never silently leave
        # live children while reporting an ordinary completed group.
        # A budget STOP must not prevent best-effort process reaping. The
        # caller preserves the primary budget failure before reaching here.
        self.budget_check = None
        errors = []
        for job in self.jobs:
            try:
                if not job.reaped:
                    job.terminate('owner cleanup after group failure')
            except BaseException as error:
                errors.append(repr(error))
        deadline = time.perf_counter() + 4
        try:
            while not all(j.reaped for j in self.jobs) and time.perf_counter() < deadline:
                self.tick()
        except BaseException as error:
            errors.append(repr(error))
        self.selector.close()
        if errors or any(not j.reaped for j in self.jobs):
            raise RuntimeError('request process cleanup failed: ' + repr(errors))
