"""Measured 8GiB aggregate STOP; not a filesystem quota or wire byte limiter."""
import json
from pathlib import Path
import time

from io_tools import WINDOW_CAP

INTERVAL = 1.0
MAX_FILES = 50000
MAX_TRACE_BYTES = 8 * 1024**2
MAX_TRACE_ROWS = 16384
DISK_RESERVE = 16 * 1024**3


class BudgetMonitor:
    def __init__(self, window, trace):
        self.root = Path(window)
        self.trace = Path(trace).open('xb')
        self.bytes = self.rows = 0
        self.last = None
        self.max_scan_s = 0

    def tick(self, force=False):
        now = time.perf_counter()
        if not force and self.last is not None and now - self.last < INTERVAL:
            return
        total = count = 0
        for path in self.root.rglob('*'):
            if path.is_symlink():
                raise ValueError('aggregate artifact symlink')
            if path.is_file():
                count += 1
                total += path.stat().st_size
                if count > MAX_FILES:
                    raise ValueError('bounded aggregate inventory count exhausted')
        ended = time.perf_counter()
        self.max_scan_s = max(self.max_scan_s, ended - now)
        row = (json.dumps({'observed_monotonic': ended, 'observed_epoch': time.time(),
            'bytes': total, 'files': count, 'scan_s': ended - now,
            'previous_scan_end_monotonic': self.last}) + '\n').encode()
        if self.bytes + len(row) > MAX_TRACE_BYTES or self.rows >= MAX_TRACE_ROWS:
            raise ValueError('aggregate monitor evidence budget')
        self.trace.write(row)
        self.trace.flush()
        self.bytes += len(row)
        self.rows += 1
        self.last = ended
        if total > WINDOW_CAP:
            raise ValueError('aggregate measured window exceeded 8GiB; preserve and restore')
        return total

    def close(self):
        self.trace.close()


def free_space_preflight(window):
    import shutil
    free = shutil.disk_usage(Path(window)).free
    # Headroom before measured cap, eight bounded request trees and preserved
    # restoration evidence. This is admission space, not a quota guarantee.
    required = WINDOW_CAP + 8 * 34 * 1024**2 + DISK_RESERVE
    if free < required:
        raise ValueError('disk free below 8GiB cap + in-flight requests +16GiB reserve')
    return {'free_bytes': free, 'required_free_bytes': required,
            'window_measured_cap_bytes': WINDOW_CAP,
            'request_tree_inflight_bytes': 8 * 34 * 1024**2,
            'preserved_evidence_reserve_bytes': DISK_RESERVE,
            'filesystem_quota': False}
