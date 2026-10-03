"""Closed, identity-bound graph domain; no model or tensor imports."""
from dataclasses import dataclass

TARGET_NAMES = tuple(
    f"language_model.model.layers.{i}.self_attn.attn" for i in range(3, 48, 4)
)
DESCRIPTORS = {(20, 4, 5): 16, (30, 6, 5): 24, (40, 8, 5): 40}
CONSUMERS = {4: (20, 4, 5), 5: (30, 6, 5), 6: (30, 6, 5),
             7: (40, 8, 5), 8: (40, 8, 5)}


def descriptor_key(desc):
    mode = getattr(desc.cg_mode, 'name', str(desc.cg_mode))
    if mode != 'FULL' or desc.attention_context_bucket is not None:
        return None
    key = (desc.num_tokens, desc.num_reqs, desc.uniform_token_count)
    return key if key in DESCRIPTORS else None


@dataclass(frozen=True)
class Ticket:
    manager_id: int
    key: tuple
    capturing: bool


def bind_owners(target, draft, owner_type, static_context):
    def collect(model):
        return {id(obj): obj for _, obj in model.named_modules()
                if type(obj) is owner_type}
    target_set, draft_set = collect(target), collect(draft)
    actual_module_names = {id(obj): name for name, obj in target.named_modules()
                           if type(obj) is owner_type}
    for identity, obj in target_set.items():
        assert actual_module_names[identity] + '.attn' == obj.layer_name
    names = {obj.layer_name for obj in target_set.values()}
    assert names == set(TARGET_NAMES), f'target owner mismatch: {sorted(names)}'
    assert len(target_set) == 12, 'duplicate target owner/name'
    assert len(draft_set) == 1, 'expected exactly one separate MTP QSA owner'
    assert not (target_set.keys() & draft_set.keys()), 'target/draft object alias'
    assert not (names & {obj.layer_name for obj in draft_set.values()})
    for obj in [*target_set.values(), *draft_set.values()]:
        assert static_context.get(obj.layer_name) is obj, 'opaque owner mismatch'
    assert next(iter(draft_set.values())).layer_name == (
        'mtp.layers.48.self_attn.attn'), 'unexpected MTP owner'
    return target_set, draft_set


def audit_dispatch(dispatch):
    """Call the actual pinned dispatcher after capture; no scheduler rewrite."""
    rows = []
    for count in range(1, 9):
        desc = dispatch(count, count * 5, 5)
        expected = CONSUMERS.get(count)
        key = descriptor_key(desc)
        assert key == expected, (count, key, expected)
        rows.append({'actual_requests': count, 'actual_tokens': count * 5,
                     'uniform': 5, 'candidate_key': key})
    # Source guard demotes shape-only prefill to None before this dispatcher.
    for count in (4, 5, 6, 7, 8):
        assert descriptor_key(dispatch(count, count * 5, None)) is None
    return rows


class Proxy:
    """Planner-only substitution; original ABI/forward attributes unchanged."""
    def __init__(self, original, planner):
        self.original, self.planner = original, planner

    def __getattr__(self, name):
        if name == 'grouped_sparse_page4_plan_fwd':
            return self.planner
        return getattr(self.original, name)
