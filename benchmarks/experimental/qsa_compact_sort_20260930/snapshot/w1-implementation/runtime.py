"""CUDA helpers imported only by the explicitly authorized offline runner."""
import hashlib
import importlib.util
import json
from pathlib import Path
import resource

from production import load_namespace

CPU_LIMIT = 3 * 1024 ** 3
GPU_ALLOC_LIMIT = 2 * 1024 ** 3
GPU_RESERVED_LIMIT = 3 * 1024 ** 3
ARTIFACT_LIMIT = 256 * 1024 ** 2


def load_extension(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def budget(torch, destination):
    stats = {'gpu_allocated': torch.cuda.max_memory_allocated(),
             'gpu_reserved': torch.cuda.max_memory_reserved(),
             'cpu_peak_rss': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
             'artifact_bytes': sum(p.stat().st_size for p in destination.rglob('*')
                                   if p.is_file())}
    assert stats['gpu_allocated'] <= GPU_ALLOC_LIMIT, stats
    assert stats['gpu_reserved'] <= GPU_RESERVED_LIMIT, stats
    assert stats['cpu_peak_rss'] <= CPU_LIMIT, stats
    assert stats['artifact_bytes'] <= ARTIFACT_LIMIT, stats
    return stats


class GuardBank:
    def __init__(self, torch):
        self.torch = torch
        self.allocations = []

    def allocate(self, shape, dtype, device):
        torch = self.torch
        shape = tuple(shape)
        base = torch.empty((shape[0] + 2, *shape[1:]),
                           dtype=dtype, device=device)
        value = base[1:-1]
        if dtype in (torch.int32, torch.int64):
            raw, sentinel = base, -777
        elif dtype == torch.uint32:
            raw, sentinel = base.view(torch.int32), -1515870811
        elif dtype == torch.float16:
            raw, sentinel = base.view(torch.int16), 0x7E35
        elif dtype == torch.float32:
            raw, sentinel = base.view(torch.int32), 0x7FC01234
        else:
            raise ValueError(dtype)
        self.allocations.append((base, raw, sentinel))
        return value

    def reset(self):
        for base, raw, sentinel in self.allocations:
            raw.fill_(sentinel)

    def check(self):
        for base, raw, sentinel in self.allocations:
            assert bool((raw[:1] == sentinel).all()), 'prefix guard overwritten'
            assert bool((raw[-1:] == sentinel).all()), 'suffix guard overwritten'

    def addresses(self):
        return [base.data_ptr() for base, _, _ in self.allocations]


class GuardTorch:
    def __init__(self, torch, bank):
        self.torch, self.bank = torch, bank

    def __getattr__(self, key):
        return getattr(self.torch, key)

    def empty(self, shape, *, dtype, device):
        return self.bank.allocate(shape, dtype, device)


class ForwardProxy:
    def __init__(self, extension):
        self.extension = extension
        self.events = []

    def __getattr__(self, name):
        function = getattr(self.extension, name)
        if name in ('grouped_sparse_page4_fwd', 'grouped_sparse_page4_split_fwd'):
            def call(*args):
                self.events.append('split' if 'split' in name else 'packed')
                return function(*args)
            return call
        return function


class State:
    def __init__(self, torch, case, device):
        self.torch, self.case = torch, case
        self.inputs = [t.to(device, copy=True).contiguous() for t in case.inputs[:5]]
        self.scalars = case.inputs[5:]
        self.q = case.q.to(device, copy=True).contiguous()
        self.pk = case.pk.to(device, copy=True).contiguous()
        self.pv = case.pv.to(device, copy=True).contiguous()
        self.ids = case.ids.to(device, copy=True).contiguous()
        assert self.ids.numel() > 0 and self.pk.shape[0] == self.ids.numel()
        assert bool((self.ids[1:] > self.ids[:-1]).all())
        self.addresses = self.pointer_list()
        self.expected = case.inputs[:5] + [case.q, case.pk, case.pv]

    def pointer_list(self):
        return [x.data_ptr() for x in self.inputs +
                [self.q, self.pk, self.pv, self.ids]]

    def update(self, phase):
        torch, case = self.torch, self.case
        source = [t.clone() for t in case.inputs[:5]]
        q, pk, pv = case.q, case.pk, case.pv
        if phase == 'B':
            # Same physical input universe; reorder request slots plus queries.
            # All source tensors come from retained CPU fixture bytes.
            source[0] = source[0].flip([0])
            source[1] = source[1].flip([0])
            old_map = source[2].flip([0])
            source[2] = torch.where(old_map >= 0,
                                    source[1].shape[0] - 1 - old_map, old_map)
            source[3] = source[3].flip([0])
            source[4] = source[4].flip([0])
            q = -q.flip([0])
            pk, pv = pk ^ 128, pv ^ 128
            if case.witness and case.witness.get('force_physical_table_update'):
                source[1] = source[1].remainder(case.inputs[7] - 1) + 1
        changed = []
        for key, target, value in zip(
                ('logical_indices', 'block_table', 'token_to_req',
                 'query_positions', 'sequence_lengths', 'q', 'k_blocks', 'v_blocks'),
                self.inputs + [self.q, self.pk, self.pv],
                source + [q, pk, pv]):
            target.copy_(value)
            original = (dict(zip(('logical_indices', 'block_table', 'token_to_req',
                                  'query_positions', 'sequence_lengths'),
                                 case.inputs[:5])) | {'q': case.q,
                                 'k_blocks': case.pk, 'v_blocks': case.pv})[key]
            if not torch.equal(value, original):
                changed.append(key)
        self.expected = source + [q, pk, pv]
        assert self.pointer_list() == self.addresses
        if phase == 'B' and case.witness:
            assert set(case.witness.get('must_change_fields', ())) <= set(changed)
        return changed

    def check_inputs_unchanged(self):
        torch = self.torch
        for target, expected in zip(self.inputs + [self.q, self.pk, self.pv],
                                    self.expected):
            actual = target.cpu()
            if expected.dtype == torch.float16:
                actual, expected = actual.view(torch.int16), expected.view(torch.int16)
            assert torch.equal(actual, expected), 'read-only input changed'
        assert torch.equal(self.ids.cpu(), self.case.ids)


class Branch:
    def __init__(self, torch, state, reference, planner, graph=False, name='branch'):
        self.torch, self.state, self.planner = torch, state, planner
        self.name = name
        self.bank = GuardBank(torch)
        self.ns = load_namespace(GuardTorch(torch, self.bank), split='split')
        self.forward = ForwardProxy(reference)
        self.plan = self.ns['_qsa_grouped_page4_workspace'](state.q)[:3]
        self.lse = self.ns['_qsa_grouped_page4_workspace'](state.q)[3]
        self.out = self.bank.allocate(state.q.shape, torch.float16, state.q.device)
        self.columns = torch.arange(4160, device=state.q.device)[None, :]
        self.graph = None
        self.reset()
        if graph:
            stream = torch.cuda.Stream(device=state.q.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self.call()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.reset()
            before = len(self.forward.events)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.call()
            self.capture_routes = self.forward.events[before:]
            assert len(self.capture_routes) == 1
        self.addresses = self.bank.addresses()

    def reset(self):
        self.bank.reset()

    def call(self):
        torch, state = self.torch, self.state
        self.planner(*state.inputs, *self.plan, *state.scalars)
        pages, masks, lengths = self.plan
        self.valid = self.columns < lengths[:, None] // 4
        self.lookup = torch.searchsorted(state.ids, pages.long().clamp(min=0))
        self.lookup = self.lookup.clamp(max=state.ids.numel() - 1)
        self.mapped_pages = torch.where(self.valid, self.lookup,
                                        torch.zeros_like(self.lookup)).int()
        meta = state.case.meta
        self.ns['_qsa_grouped_page4_forward'](
            self.forward, state.q, state.pk, state.pv, self.out,
            self.mapped_pages, masks, lengths, self.lse,
            state.q.shape[2] ** -.5, meta['kv_cache_dtype'],
            meta['k_scale'], meta['v_scale'], state.inputs[2])

    def execute(self):
        self.reset()
        if self.graph is None:
            self.call()
        else:
            self.graph.replay()

    def validate(self):
        torch, state = self.torch, self.state
        self.bank.check()
        assert self.bank.addresses() == self.addresses
        assert torch.equal(state.ids[self.lookup][self.valid],
                           self.plan[0].long()[self.valid]), 'missing physical KV'
        expected = 'split' if state.q.shape[0] <= 64 else 'packed'
        events = (self.capture_routes if self.graph is not None
                  else self.forward.events[-1:])
        assert events == [expected], f'wrong original dispatcher route {events}'
        return expected


def equal(torch, x, y, label, destination, case_name):
    xx = x.contiguous()
    yy = y.contiguous()
    if xx.dtype == torch.float16:
        xx, yy = xx.view(torch.int16), yy.view(torch.int16)
    if xx.dtype == torch.float32:
        xx, yy = xx.view(torch.int32), yy.view(torch.int32)
    if not torch.equal(xx, yy):
        differing = (xx != yy).flatten()
        first = int(differing.to(torch.int8).argmax().item())
        lo, hi = max(0, first - 16), first + 17
        path = destination / 'first-divergence.pt'
        assert not path.exists()
        torch.save({'case': case_name, 'field': label, 'first_flat': first,
                    'shape': list(x.shape),
                    'reference_window': xx.flatten()[lo:hi].cpu(),
                    'candidate_window': yy.flatten()[lo:hi].cpu()}, path)
        assert path.stat().st_size <= 64 * 1024 ** 2
        raise AssertionError(f'{case_name} {label} first_flat={first}')


def compare_branches(torch, branches, destination, case_name):
    first = branches[0]
    for branch in branches:
        for index, (left, right) in enumerate(zip(first.bank.allocations,
                                                 branch.bank.allocations)):
            equal(torch, left[1], right[1], f'{first.name}_vs_{branch.name}/full_storage_{index}',
                  destination, case_name)
        equal(torch, first.mapped_pages, branch.mapped_pages,
              f'{first.name}_vs_{branch.name}/own_remap',
              destination, case_name)


def witness_check(torch, case, branch):
    if not case.witness:
        return {}
    pages, masks, lengths = [t.cpu() for t in branch.plan]
    effective = int(lengths[0]) // 4
    real = {}
    padding = []
    for page, mask in zip(pages[0, :effective].tolist(),
                          masks[0, :effective].tolist()):
        if mask:
            assert page not in real, 'duplicate real physical entry'
            real[page] = mask
        else:
            padding.append(page)
    assert set(case.witness.get('must_select', ())) <= set(real)
    for page, expected in case.witness.get('exact_masks', {}).items():
        assert real[int(page)] == expected, (page, real.get(int(page)), expected)
    if case.witness.get('zero_mask_padding_nonnull'):
        assert len(padding) > 0 and set(padding) <= set(real)
        assert 0 not in padding
        assert bool((case.pk[:408] == 127).all())
        assert bool((case.pv[:408] == 255).all())
        assert bool(torch.isfinite(case.q).all())
        assert bool(torch.isfinite(branch.out).all()), 'null NaN leaked into valid output'
    if case.witness.get('selected_negative_zero_v'):
        selected = case.ids.tolist().index(816)
        assert bool((case.pv[selected] == 128).all())
    for row in case.witness.get('padding_rows', ()):
        assert all(((mask >> (row * 4)) & 15) == 0 for mask in real.values())
    return {'real_entry_count': len(real), 'zero_mask_padding_entries': len(padding),
            'real_pages_and_masks': real, 'witness_assertions': True}
