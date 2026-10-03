"""One first-divergence bank per rank, separate from private workspace.

Snapshots input/metadata/output values through integer views, never float math.
No KV values: exact planner replay is possible; attention replay is not complete.
"""


def bit_view(torch, tensor):
    if tensor.dtype == torch.float16:
        return tensor.view(torch.int16)
    if tensor.dtype in (torch.float32, torch.uint32):
        return tensor.view(torch.int32)
    return tensor


def values(arguments, reference, private):
    rp, rm, rl, r_lse = reference
    return {
        'q': arguments['q'], 'logical_indices': arguments['logical_indices'],
        'block_table': arguments['block_table'],
        'token_to_req': arguments['token_to_req'],
        'query_positions': arguments['query_positions'],
        'sequence_lengths': arguments['sequence_lengths'],
        'reference_pages': rp, 'reference_masks': rm,
        'reference_lengths': rl, 'reference_out': arguments['out'],
        'reference_lse': r_lse,
        'candidate_pages': private.pages.value,
        'candidate_masks': private.masks.value,
        'candidate_lengths': private.lengths.value,
        'candidate_out': private.out.value,
        'candidate_lse': private.lse.value,
    }


class FirstFailure:
    def __init__(self, torch, example, cap):
        self.torch = torch
        estimated = 32 + sum(v.numel() * v.element_size() for v in example.values())
        assert estimated <= cap - 1024 * 1024, 'first snapshot preallocation cap'
        # All banks allocated during descriptor-bound preparation at FULL40.
        self.bank = {name: torch.empty_like(value, memory_format=torch.contiguous_format)
                     for name, value in example.items()}
        self.flag = torch.zeros(4, dtype=torch.int64,
                                device=next(iter(example.values())).device)
        self.bytes = 32 + sum(v.numel() * v.element_size() for v in self.bank.values())
        assert self.bytes <= cap - 1024 * 1024, 'first snapshot device cap'
        self.constants, self.schemas = {}, {}

    def prepare(self, index, example):
        for key, value in example.items():
            assert value.dtype == self.bank[key].dtype
            assert len(value.shape) == len(self.bank[key].shape)
            assert all(a <= b for a, b in zip(value.shape, self.bank[key].shape))
        self.schemas[index] = {key: list(value.shape) for key, value in example.items()}
        self.constants[index] = self.torch.tensor(
            [1, index, example['q'].shape[0], example['reference_pages'].shape[0]],
            dtype=self.torch.int64, device=self.flag.device)
        self.bytes += self.constants[index].numel() * 8

    def capture(self, index, example, bad):
        take = bad & (self.flag[0] == 0)
        for name, source in example.items():
            target = self.bank[name][tuple(slice(0, size) for size in source.shape)]
            target_bits = bit_view(self.torch, target)
            source_bits = bit_view(self.torch, source)
            target_bits.copy_(self.torch.where(take, source_bits, target_bits))
        # Commit after all values, same stream as comparison/captured node.
        self.flag.copy_(self.torch.where(take, self.constants[index], self.flag))

    def reset(self):
        self.flag.zero_()

    def export(self, path, cap, identity):
        flag = self.flag.cpu().tolist()
        if not flag[0]:
            return None
        index = flag[1]
        payload = {key: value[tuple(slice(0, n) for n in self.schemas[index][key])].cpu()
                   for key, value in self.bank.items()}
        retained = sum(v.numel() * v.element_size() for v in payload.values())
        assert retained <= cap - 1024 * 1024
        with path.open('xb') as stream:
            self.torch.save({'flag': flag, 'identity': identity, 'values': payload,
                             'scope': 'exact planner inputs; KV values not retained'}, stream)
        assert path.stat().st_size <= cap
        return {'path': str(path), 'bytes': path.stat().st_size,
                'node_index': index, 'scope': 'INTEGER_REPRODUCIBLE_FLOAT_KV_MISSING'}
