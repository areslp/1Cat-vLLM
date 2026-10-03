"""Pinned completion request identities; no model imports or HTTP side effects."""
import re


def completion_ids(body, header_id):
    """The body/header are frozen equal; one prompt, one completion only."""
    base = body['request_id']
    assert isinstance(base, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,120}', base)
    assert header_id == base, 'header must equal frozen body request_id'
    assert body.get('n', 1) == 1 and body.get('best_of', 1) == 1
    assert body.get('use_beam_search', False) is False
    assert body.get('stream', False) is False
    prompt = body['prompt']
    assert isinstance(prompt, str) or (
        isinstance(prompt, list) and prompt
        and all(type(token) is int and token >= 0 for token in prompt)
    ), 'exactly one text or token-ID prompt; no multi-prompt batching'
    response = f'cmpl-{base}'
    return {'client_request_id': base, 'response_id': response,
            'external_request_id': f'{response}-0'}


def bind_scheduler_ids(actual, allowed_external, bindings):
    """Bind exact external IDs to one randomised full scheduler ID each."""
    assert len(allowed_external) == len(set(allowed_external))
    assert len(actual) == len(set(actual)) and actual
    updated = dict(bindings)
    assert set(updated) <= set(allowed_external)
    assert len(updated.values()) == len(set(updated.values()))
    seen = set()
    for scheduler_id in actual:
        assert isinstance(scheduler_id, str)
        matches = [external for external in allowed_external if re.fullmatch(
            re.escape(external) + r'-[0-9a-f]{8}', scheduler_id)]
        assert len(matches) == 1, 'foreign ID or unexpected random suffix'
        external = matches[0]
        assert external not in seen, 'duplicate external request in one step'
        seen.add(external)
        assert updated.get(external, scheduler_id) == scheduler_id, (
            'external request remapped to another scheduler ID')
        assert scheduler_id not in [value for key, value in updated.items()
                                    if key != external]
        updated[external] = scheduler_id
    return updated


def verify_rank_bindings(rank_rows, expected_external, sentinel_external):
    assert sorted(row['rank'] for row in rank_rows) == [0, 1, 2, 3]
    expected = set(expected_external) | {sentinel_external}
    rows = []
    for row in rank_rows:
        mapping = row['request_id_bindings']
        assert set(mapping) == expected, 'missing cohort or sentinel mapping'
        checked = bind_scheduler_ids(list(mapping.values()), sorted(expected), {})
        assert checked == mapping
        rows.append(mapping)
    assert all(row == rows[0] for row in rows), 'cross-rank mapping mismatch'
    return rows[0]
