"""Finite initial experiment; no automatic confirmation, retries or arm changes."""
ARMS = ('A0', 'B', 'A2')


class Flow:
    def __init__(self):
        self.started = []
        self.stopped = []
        self.active = None
        self.failure = None

    def begin(self, arm):
        if (self.failure is not None or self.active is not None
                or self.started != self.stopped
                or len(self.started) >= 3 or arm != ARMS[len(self.started)]):
            raise ValueError('only one ordered A0/B/A2, stopped before next')
        self.active = arm
        self.started.append(arm)

    def stop(self, arm, proof):
        if (arm != self.active or proof['status'] != 'PASS_OWNED_ARM_STOPPED_GPU_EMPTY'
                or proof['arm'] != arm):
            raise ValueError('exact actual owned stopped arm required')
        self.stopped.append(arm)
        self.active = None

    def fail(self, error):
        self.failure = repr(error)[:4096]

    def analyzed(self):
        if self.failure is not None or tuple(self.stopped) != ARMS or self.active is not None:
            raise ValueError('complete stopped initial three arms required before CI')
