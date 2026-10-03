"""Bounded CPU adapter/bookkeeping tests; not CUDA/service fallback proof."""
import argparse
import ast
import io
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from headers import HeaderUnpickler, Storage, rebuild, rebuild_v3
from production import load_namespace, FUNCTIONS


class FakeTensor:
    def __init__(self, shape, storage=None):
        self.shape = shape
        self.device = SimpleNamespace(index=0)
        self.storage = storage or object()

    def __getitem__(self, index):
        length = len(range(self.shape[0])[index])
        return FakeTensor((length, *self.shape[1:]), self.storage)


class FakeTorch:
    int32 = 'i32'
    uint32 = 'u32'
    float32 = 'f32'

    def __init__(self):
        self.stream = 11
        self.cuda = SimpleNamespace(current_stream=lambda *_: SimpleNamespace(
            cuda_stream=self.stream))

    def empty(self, shape, **kwargs):
        return FakeTensor(tuple(shape))


def extension(abi=2, split=True):
    events = []
    mod = SimpleNamespace(
        grouped_sparse_page4_abi_version=lambda: abi,
        grouped_sparse_page4_plan_fwd=lambda *args: None,
        grouped_sparse_page4_fwd=lambda *args: events.append('packed'),
        decode_paged_xqa_fwd=lambda *args: None)
    if split:
        mod.grouped_sparse_page4_split_fwd = lambda *args: events.append('split')
    return mod, events


class ContractTests(unittest.TestCase):
    def test_all_owned_python_parses(self):
        for path in Path(__file__).parent.glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))

    def test_source_bodies_exact(self):
        source = Path(__file__).parent / 'source/qsa.production.py'
        originals = {n.name: ast.dump(n) for n in ast.parse(source.read_text()).body
                     if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS}
        ns = load_namespace(None)
        self.assertEqual(len(ns['source_extraction']), len(originals))
        for row in ns['source_extraction']:
            self.assertEqual(ns[row['name']].__code__.co_firstlineno, row['line'])

    def test_forward_dispatch_actual_body(self):
        for rows, mode, abi, split, mapping, expected in (
                (64, 'split', 2, True, object(), 'split'),
                (65, 'split', 2, True, object(), 'packed'),
                (512, 'split', 2, True, object(), 'packed'),
                (8, '', 2, True, object(), 'packed'),
                (8, 'split', 2, False, object(), 'packed'),
                (8, 'split', 2, True, None, 'packed'),
                (8, 'split', 1, True, object(), 'packed')):
            ns = load_namespace(None, split=mode)
            mod, events = extension(abi, split)
            ns['_qsa_grouped_page4_forward'](
                mod, FakeTensor((rows, 6, 256)), *([None] * 7),
                .1, 'float16' if abi == 1 else 'fp8_e4m3', 1., 1., mapping)
            self.assertEqual(events, [expected])

    def test_quantized_abi1_ineligible(self):
        ns = load_namespace(None)
        mod, _ = extension(1)
        self.assertFalse(ns['_qsa_grouped_page4_supported'](mod, 'fp8_e4m3'))
        self.assertTrue(ns['_qsa_grouped_page4_supported'](mod, 'float16'))

    def test_unknown_abi_fail_closed(self):
        ns = load_namespace(None)
        mod, _ = extension(0)
        self.assertFalse(ns['_qsa_grouped_page4_supported'](mod, 'fp8_e4m3'))

    def test_workspace_capacity_stream(self):
        torch = FakeTorch()
        ns = load_namespace(torch)
        previous = None
        for groups, capacity in ((1, 1), (3, 4), (8, 8), (2, 8)):
            views = ns['_qsa_grouped_page4_workspace'](FakeTensor((groups * 8, 6, 256)))
            self.assertEqual(ns['_SM70_QSA_GROUPED_PAGE4_WORKSPACES'][(0, 11)][0], capacity)
            self.assertEqual(views[0].shape, (groups, 4160))
            if groups == 2:
                self.assertIs(views[0].storage, previous)
            previous = views[0].storage
        torch.stream = 12
        other = ns['_qsa_grouped_page4_workspace'](FakeTensor((16, 6, 256)))
        self.assertIsNot(other[0].storage, previous)

    def test_outer_routing_actual_body_host_mocks(self):
        for rows, grouped, abi, expected in (
                (7, True, 2, [('xqa', 7)]),
                (8, True, 2, [('grouped', 8)]),
                (9, True, 2, [('grouped', 8), ('xqa', 1)]),
                (65, True, 2, [('grouped', 64), ('xqa', 1)]),
                (33, False, 2, [('xqa', 16), ('xqa', 16), ('xqa', 1)]),
                (17, True, 1, [('xqa', 16), ('xqa', 1)])):
            ns = load_namespace(None, grouped=grouped)
            mod, _ = extension(abi)
            events = []
            ns['_qsa_sparse_paged_attention_sm70_grouped_page4'] = (
                lambda q, *args: events.append(('grouped', q.shape[0])))
            ns['_qsa_sparse_paged_attention_sm70_xqa_page4_batch'] = (
                lambda q, *args: events.append(('xqa', q.shape[0])))
            name = 'flash_attn_v100.flash_attn_interface'
            old = sys.modules.get(name)
            sys.modules[name] = SimpleNamespace(flash_attn_v100_cuda=mod)
            try:
                tensor = FakeTensor((rows, 6, 256))
                ns['_qsa_sparse_paged_attention_sm70_xqa_page4'](
                    tensor, None, None, tensor, None, tensor, tensor,
                    None, tensor, 'fp8_e4m3', 1., 1.)
            finally:
                if old is None:
                    del sys.modules[name]
                else:
                    sys.modules[name] = old
            self.assertEqual(events, expected)

    def test_headers_byte_bounds(self):
        tensor = rebuild(Storage('0', 20, 2), 0, (4, 5), (5, 1))
        self.assertEqual(tensor.bytes, 40)
        self.assertEqual(rebuild_v3(Storage('0', 80, 1), 0,
                                   (4, 5), (5, 1), False, None, 'uint32').bytes, 80)
        with self.assertRaises(AssertionError):
            rebuild_v3(Storage('0', 79, 1), 0, (4, 5), (5, 1),
                       False, None, 'uint32')

    def test_untrusted_header_globals_rejected(self):
        with self.assertRaises(ValueError):
            HeaderUnpickler(io.BytesIO(pickle.dumps(Path('/tmp/no-action')))).load()


class ActualCPUTensorTests(unittest.TestCase):
    def test_null_poison_and_signed_zero_inputs(self):
        import torch
        from fixtures import witness
        case = witness(torch, 'null_padding')
        self.assertTrue(bool((case.pk[:408] == 127).all()))
        self.assertTrue(bool((case.pv[:408] == 255).all()))
        self.assertFalse(bool(((case.pk[408:] & 127) == 127).any()))
        self.assertFalse(bool(((case.pv[408:] & 127) == 127).any()))
        self.assertTrue(bool(torch.isfinite(case.q).all()))
        case = witness(torch, 'signed_zero_read')
        self.assertTrue(bool((case.pv[816] == 128).all()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_aba_actual_tensors_and_fixed_addresses(self):
        import torch
        from fixtures import witness
        from runtime import State
        self.assertFalse(torch.cuda.is_initialized())
        case = witness(torch, 'slot_reuse_aba')
        pristine = [t.clone() for t in case.inputs[:5] + [case.q, case.pk, case.pv, case.ids]]
        state = State(torch, case, torch.device('cpu'))
        self.assertEqual(state.update('A'), [])
        changed = state.update('B')
        self.assertTrue(set(case.witness['must_change_fields']) <= set(changed))
        state.check_inputs_unchanged()
        self.assertEqual(state.update('A'), [])
        state.check_inputs_unchanged()
        self.assertEqual(state.pointer_list(), state.addresses)
        for before, after in zip(pristine, case.inputs[:5] + [case.q, case.pk, case.pv, case.ids]):
            self.assertTrue(torch.equal(before, after), 'retained fixture mutated')
        self.assertFalse(torch.cuda.is_initialized())

    def test_witness_tail_masks_independent_input_walk(self):
        import torch
        from fixtures import witness
        case = witness(torch, 'causal_tail_table_end')
        li, bt, mapping, pos, lengths = case.inputs[:5]
        result = {}
        for row in range(8):
            visible = min(int(pos[row]) + 1, int(lengths[int(mapping[row])]))
            complete = min(visible // 4, 2051 // 4)
            for index in range(complete):
                tokens = li[row, index * 4:index * 4 + 4].tolist()
                if tokens[0] < 0:
                    continue
                for token in tokens:
                    if token >= 0:
                        physical = int(bt[int(mapping[row]), token // 1632]) * 408 + token % 1632 // 4
                        result[physical] = result.get(physical, 0) | 1 << (row * 4 + token % 4)
            if visible % 4:
                token = int(li[row, complete * 4])
                self.assertEqual(token, visible // 4 * 4)
                physical = int(bt[int(mapping[row]), token // 1632]) * 408 + token % 1632 // 4
                result[physical] = result.get(physical, 0) | ((1 << (visible % 4)) - 1) << (row * 4)
        for physical, expected in case.witness['exact_masks'].items():
            self.assertEqual(result[int(physical)], expected)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--torch-cpu', action='store_true')
    args = p.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests)
    if args.torch_cpu:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ActualCPUTensorTests))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
