"""CPU faults for the newly added passive sampling boundary only."""
import copy
import unittest

from exporter_fence import wait_fence


def rows(published=101.0, counters=12):
    return [{'pid': 1000 + rank, 'rank': rank, 'ranks': 4, 'mode': 'original',
             'module_path': '/pinned/e7.py', 'package_hashes': {'e7': 'pin'},
             'time': published, 'counters': {'hook_calls': counters,
                                          'compressed_steps': 2}}
            for rank in range(4)]


class FenceTest(unittest.TestCase):
    def run_case(self, frames):
        elapsed = [0.0]
        saved, trace = {}, []

        def read():
            index = min(int(elapsed[0] * 4), len(frames) - 1)
            return copy.deepcopy(frames[index])

        def sleep(seconds):
            elapsed[0] += seconds

        value = wait_fence('before', rows(), read, lambda: 100.0,
                          lambda n, v: saved.update({n: v}), 99.0, trace.append,
                          clock=lambda: elapsed[0], sleep=sleep,
                          epoch=lambda: 100.0 + elapsed[0])
        return value, saved, trace, elapsed[0]

    def test_stale_initial_then_stable_success(self):
        selected, saved, trace, elapsed = self.run_case(
            [rows(99), rows(100), rows(101), rows(101.25), rows(102)])
        receipt = saved['before-fence.json']
        self.assertEqual(selected, rows(102))
        self.assertEqual(receipt['first'], rows(101))
        self.assertEqual(receipt['second'], rows(102))
        self.assertEqual(len(trace), 5)
        self.assertLessEqual(elapsed, 15)

    def test_stale_never_fresh_times_out(self):
        with self.assertRaisesRegex(TimeoutError, 'within 15s'):
            self.run_case([rows(99)])

    def test_rank_counter_mismatch_never_passes(self):
        mismatched = rows()
        mismatched[3]['counters']['hook_calls'] -= 1
        with self.assertRaisesRegex(TimeoutError, 'within 15s'):
            self.run_case([mismatched])

    def test_source_or_pid_change_fails(self):
        for field, value in (('pid', 9999), ('module_path', '/foreign.py'),
                             ('package_hashes', {'e7': 'changed'})):
            with self.subTest(field=field):
                changed = rows()
                changed[0][field] = value
                with self.assertRaisesRegex(AssertionError, 'source/PID'):
                    self.run_case([changed])


if __name__ == '__main__':
    unittest.main()
