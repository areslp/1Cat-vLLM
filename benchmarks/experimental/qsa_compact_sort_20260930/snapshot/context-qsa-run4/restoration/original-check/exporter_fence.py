"""Passive full-counter publication fence; no requests or exporter mutation."""
import json
import time

IDENTITY = ('pid', 'rank', 'ranks', 'mode', 'module_path', 'package_hashes')


def same_counters(rows):
    return len({json.dumps(r['counters'], sort_keys=True) for r in rows}) == 1


def wait_fence(name, expected, read_rows, idle, save, last_completed,
               trace, clock=time.monotonic, sleep=time.sleep, epoch=time.time):
    """Select two equal fresh vectors separated by the 1s publish period."""
    boundary = idle()
    assert boundary >= last_completed, 'queue boundary predates completion'
    assert [r['rank'] for r in expected] == list(range(4))
    deadline = clock() + 15
    first = None
    while clock() < deadline:
        rows = sorted(read_rows(), key=lambda r: r['rank'])
        assert [r['rank'] for r in rows] == list(range(4)), 'rank identity changed'
        for row, old in zip(rows, expected):
            assert all(row[key] == old[key] for key in IDENTITY), (
                name + ' source/PID identity changed')
            assert type(row['time']) in (int, float)
            assert isinstance(row['counters'], dict)
            assert all(type(v) in (int, float) and v >= 0
                       for v in row['counters'].values())
        trace({'observed_epoch': epoch(), 'rows': rows})
        if all(r['time'] > boundary for r in rows) and same_counters(rows):
            if first is None or rows[0]['counters'] != first[0]['counters']:
                first = rows
            elif all(r['time'] - z['time'] >= 1.0
                     for r, z in zip(rows, first)):
                assert idle() >= boundary
                receipt = {'status': 'FRESH_STABLE_FULL_COUNTERS_ALL_RANKS',
                    'boundary_epoch': boundary,
                    'last_completed_epoch': last_completed,
                    'export_period_s': 1.0, 'passive_budget_s': 15,
                    'first': first, 'second': rows}
                save(name + '-fence.json', receipt)
                return rows
        else:
            first = None
        sleep(0.25)
    raise TimeoutError(name + ' exporter fence timed out within 15s')
