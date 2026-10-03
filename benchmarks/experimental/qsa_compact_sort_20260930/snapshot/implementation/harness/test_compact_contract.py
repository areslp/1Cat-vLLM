"""Bounded CPU contracts. No compile, CUB execution, CUDA or float validation."""

from pathlib import Path
import random
import unittest

from compact_contract import CAPACITY, EMPTY, bucket, ordered, scatter
from compact_contract import stable_compact


class CompactChecks(unittest.TestCase):
    def table(self, count):
        generator = random.Random(58 + count)
        slots, owners = [EMPTY] * CAPACITY, [EMPTY] * CAPACITY
        locations = generator.sample(range(CAPACITY), count)
        masks = [0xF, 0xF000, 0xF00F, 0xF000000, 0xF00000F,
                 0xF00F000, 0xF00F00F]
        for index, slot in enumerate(locations):
            slots[slot] = (masks[index % len(masks)] << 32) | (index + 1)
            owners[slot] = index
        return slots, owners

    def test_bucket_edges_and_full_fallback(self):
        counts = (0, 1, 1024, 1025, 2048, 2049, 4096, 4097, 8192)
        expected = (1024, 1024, 1024, 2048, 2048, 4096, 4096, 8192, 8192)
        for count, size in zip(counts, expected):
            self.assertEqual(bucket(count), size)
            slots, owners = self.table(count)
            compact = stable_compact(slots, owners)
            self.assertEqual(len(compact), count)
            self.assertEqual(compact, [(entry, owner) for entry, owner in
                zip(slots, owners) if entry & EMPTY != EMPTY])
            padded = compact + [(EMPTY, EMPTY)] * (size - count)
            self.assertEqual(scatter(ordered(zip(slots, owners))),
                             scatter(ordered(padded)))
        with self.assertRaises(ValueError):
            bucket(8193)

    def test_empty_suffix_and_missing_category_padding(self):
        pages, masks, length = scatter([])
        self.assertEqual(length, 0)
        self.assertEqual(pages, [-777] * 4160)
        self.assertEqual(masks, [0xA5A5A5A5] * 4160)
        entries = [(0xF << 32 | 91, 0), (0xF000000 << 32 | 92, 1)]
        pages, masks, length = scatter(ordered(entries))
        self.assertEqual(length, 64)
        self.assertEqual(pages[:16], [91] * 8 + [92] + [91] * 7)
        self.assertEqual(masks[1:8] + masks[9:16], [0] * 14)
        self.assertEqual(pages[16:], [-777] * 4144)

    def test_same_owner_implies_same_physical_microblock(self):
        for page_size in (4, 8, 32, 1632):
            for logical_group in range(8192):
                physical = set()
                for token in range(logical_group * 4, logical_group * 4 + 4):
                    page, offset = divmod(token, page_size)
                    physical_page = (page * 17) % 37
                    physical.add(physical_page * (page_size // 4) + offset // 4)
                self.assertEqual(len(physical), 1)

    def test_stable_compaction_retains_original_tie_order(self):
        slots, owners = [EMPTY] * CAPACITY, [EMPTY] * CAPACITY
        for slot, page in [(0, 7), (15, 6), (16, 5), (8191, 4)]:
            slots[slot], owners[slot] = 0xF << 32 | page, 0
        self.assertEqual([entry & EMPTY for entry, _ in
                          stable_compact(slots, owners)], [7, 6, 5, 4])
        # Artificial ties are bookkeeping-only; distinct producer entries
        # cannot share owner under the page_size%4 input contract above.


def source_checks(sources):
    old = sources['reference54/source/planner54.cu']
    new = sources['source/planner58.cu']
    host = 'at::Tensor flash_attention_grouped_sparse_page4_plan('
    assert old[old.index(host):] == new[new.index(host):]
    hash_start = '__device__ __forceinline__ void grouped_sparse_hash_insert('
    category_end = '  return active_m_tiles;\n}'
    for source in (old, new):
        assert 'page_size > 0 && page_size % 4 == 0' in source
    assert old[old.index(hash_start):old.index(category_end)+len(category_end)] == (
        new[new.index(hash_start):new.index(category_end)+len(category_end)])
    kernel = '__launch_bounds__(kGroupedSparsePlannerThreads, 1) void '
    load = '  unsigned long long entries[kGroupedSparseItemsPerThread];'
    assert old[old.index(kernel):old.index(load)] == new[
        new.index(kernel):new.index(load)]
    assert 'int last = sort_capacity;' in new
    assert 'sorted_index < sort_capacity' in new
    assert 'if (slot < valid_count)' in new
    assert 'valid_count > 4096' in new
    assert '__shared__ int' not in new
    assert 'Sort(sort_keys, entries, 0, 36)' in new
    assert 'Sort(keys, entries, 0, 36)' in new
    harness = sources['harness/probe58.py']
    start = harness.index('def graph_bucket_transition():')
    end = harness.index('def bucket_boundary(', start)
    transition = harness[start:end]
    assert '(0,1,1024,1025,2048,2049,4096,4097,8192,0,1)' in transition
    assert 'destination.copy_(source)' in transition
    assert "plan[0].fill_(-777);plan[1].fill_(0xA5A5A5A5)" in transition
    assert "plan[2].fill_(-777)" in transition
    assert 'call(\'reference\',inp,ro);graph.replay()' in transition
    assert 'assert addresses==' in transition
