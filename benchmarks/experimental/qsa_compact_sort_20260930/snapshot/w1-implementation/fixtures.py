"""Original 72 seeds/9 bucket inputs plus bounded deterministic witnesses."""
from dataclasses import dataclass

VARIANTS = ('continuous', 'duplicates', 'reverse', 'shared_pages',
            'all_empty', 'slot_reuse', 'table_end', 'null_page', 'signed_zero')
BOUNDARIES = (0, 1, 1024, 1025, 2048, 2049, 4096, 4097, 8192)
TRANSITIONS = (*BOUNDARIES, 0, 1)
WITNESSES = ('causal_tail_table_end', 'padding_shared',
             'null_padding', 'signed_zero_read', 'slot_reuse_aba')


@dataclass
class Case:
    name: str
    inputs: list
    q: object
    pk: object
    pv: object
    ids: object
    meta: dict
    saved: dict | None = None
    witness: dict | None = None


def kv_and_q(torch, generator, rows, ncache, stride):
    pk = torch.randint(0, 255, (ncache * stride, 4, 1, 256),
                       generator=generator, dtype=torch.uint8)
    pk[(pk & 127) == 127] = 126
    pv = pk.roll(1, dims=0)
    pk[:stride] = 127
    pv[:stride] = 255
    q = (torch.randn(rows, 6, 256, generator=generator) * .1).half()
    return pk, pv, q


def synthetic(torch, seed, groups, variant):
    generator = torch.Generator().manual_seed(seed)
    nr, page, ncache, stride, width = 8, 1632, 64, 408, 21
    li = torch.full((groups * 8, 2051), -1, dtype=torch.int32)
    bt = torch.randint(1, ncache, (nr, width), generator=generator,
                       dtype=torch.int32)
    lengths = torch.tensor([0, 1, 1631, 1632, 1633, 8192, 32767, 32768],
                           dtype=torch.int32)
    mapping = torch.tensor([(i * 3 + seed) % nr for i in range(groups * 8)],
                           dtype=torch.int32)
    pos = torch.empty(groups * 8, dtype=torch.int64)
    if variant == 'shared_pages':
        bt[:] = bt[0].clone()
    if variant == 'all_empty':
        mapping.fill_(-1)
    if variant == 'slot_reuse':
        mapping = torch.flip(mapping, dims=[0])
    if variant == 'null_page':
        bt[0].zero_()
        lengths[0] = 1632
    if variant == 'table_end':
        bt[:, -1] = -1
    for row, request in enumerate(mapping.tolist()):
        if request < 0:
            pos[row] = -1
            continue
        length = int(lengths[request])
        visible = max(0, length - row % 4)
        pos[row] = visible - 1
        count = min(visible // 4, 512)
        if count:
            pages = torch.randperm(max(1, length // 4),
                                   generator=generator)[:count]
            if variant == 'continuous':
                pages = torch.arange(count)
            if variant == 'duplicates':
                pages = pages.remainder(max(1, count // 3))
            if variant == 'reverse':
                pages = pages.sort(descending=True).values
            li[row, :count * 4] = (
                pages[:, None] * 4 + torch.arange(4)).reshape(-1).int()
        if visible % 4:
            li[row, count * 4] = visible // 4 * 4
    # Same generator order as frozen synthetic(): KV first, then Q.
    pk, pv, q = kv_and_q(torch, generator, groups * 8, ncache, stride)
    if variant == 'signed_zero':
        pk[2 * stride:3 * stride] = 0
        pv[2 * stride:3 * stride] = 128
    return Case(f'seed{seed}-g{groups}-{variant}',
                [li, bt, mapping, pos, lengths, page, stride, ncache],
                q, pk, pv, torch.arange(ncache * stride, dtype=torch.int64),
                {'kind': 'synthetic', 'kv_cache_dtype': 'fp8_e4m3',
                 'k_scale': 1., 'v_scale': 1., 'variant': variant,
                 'seed': seed, 'groups': groups})


def boundary(torch, count):
    nr, page, ncache, stride, width = 8, 1632, 24, 408, 21
    li = torch.full((8, 2051), -1, dtype=torch.int32)
    bt = torch.arange(1, width + 1, dtype=torch.int32).expand(nr, -1).clone()
    mapping = torch.arange(nr, dtype=torch.int32)
    lengths = torch.full((nr,), 32768, dtype=torch.int32)
    pos = lengths.long() - 1
    remaining = count
    for row in range(nr):
        take = min(1024, remaining)
        remaining -= take
        li[row, :take] = 4 * (row * 1024 + torch.arange(take))
    assert remaining == 0 and 0 <= count <= 8192
    generator = torch.Generator().manual_seed(58000 + count)
    pk, pv, q = kv_and_q(torch, generator, 8, ncache, stride)
    return Case(f'bucket{count}',
                [li, bt, mapping, pos, lengths, page, stride, ncache],
                q, pk, pv, torch.arange(ncache * stride, dtype=torch.int64),
                {'kind': 'binding_boundary', 'kv_cache_dtype': 'fp8_e4m3',
                 'k_scale': 1., 'v_scale': 1., 'expected_hash_entries': count})


def witness(torch, name):
    case = boundary(torch, 1)
    case.name = 'witness-' + name
    li, bt, mapping, pos, lengths = case.inputs[:5]
    li.fill_(-1)
    mapping.zero_()
    if name == 'causal_tail_table_end':
        # Last legal table column, full pages across1632, and 1/2/3-token
        # causal tails are explicit, not a random sampling probability.
        lengths.fill_(32768)
        pos[:] = torch.tensor([1630, 1631, 1632, 1633,
                               32764, 32765, 32766, 32767])
        for row in range(8):
            selected = [token for token in (0, 1628, 1632, 32764)
                        if token + 3 <= int(pos[row])]
            for index, token in enumerate(selected):
                li[row, index * 4:index * 4 + 4] = torch.arange(token, token + 4)
            visible = int(pos[row]) + 1
            if visible % 4:
                tail_index = min(visible // 4, 512) * 4
                li[row, tail_index] = visible // 4 * 4
        case.witness = {'must_select': [815, 816, 8599],
                        'exact_masks': {'815': 0xFFFFFFF7,
                                        '816': 0xFFFF3100,
                                        '8599': 0xF7310000}}
    elif name == 'padding_shared':
        mapping[:] = torch.tensor([0, 0, -1, 1, 1, -1, 0, 1])
        lengths.fill_(1633)
        pos.fill_(1632)
        li[:, :4] = torch.arange(4)
        bt[:] = bt[0].clone()
        case.witness = {'must_select': [408], 'padding_rows': [2, 5]}
    elif name == 'null_padding':
        lengths.fill_(4)
        pos.fill_(3)
        li[:, :4] = torch.arange(4)
        case.witness = {'must_select': [408],
                        'zero_mask_padding_nonnull': True,
                        'null_nan_codes': [127, 255]}
    elif name == 'signed_zero_read':
        bt.fill_(2)
        lengths.fill_(4)
        pos.fill_(3)
        li[:, :4] = torch.arange(4)
        case.pk[2 * 408:3 * 408] = 0
        case.pv[2 * 408:3 * 408] = 128
        case.witness = {'must_select': [816], 'selected_negative_zero_v': True}
    elif name == 'slot_reuse_aba':
        mapping[:] = torch.tensor([0, 2, 1, 3, 4, 5, 6, 7])
        lengths.fill_(1633)
        pos.fill_(1632)
        li[:, :4] = torch.arange(4)
        bt[:, 0] = torch.arange(1, 9)
        case.witness = {'force_physical_table_update': True,
                        'must_change_fields': ['block_table', 'token_to_req',
                                               'q', 'k_blocks', 'v_blocks']}
    else:
        raise ValueError(name)
    case.meta['kind'] = 'deterministic_witness'
    return case
