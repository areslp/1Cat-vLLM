"""Lossless, field-specific host events. No tensor read or permissive encoder."""

INTEGER_MAX = 2**31 - 1


def integer(value, *, optional=False, minimum=0, maximum=INTEGER_MAX):
    if value is None:
        assert optional, 'None in required integer field'
        return None
    if type(value) is not int:
        # NumPy is already part of the deployed runner. The exact declared
        # scalar family is accepted; bool/float/array/Tensor/__int__ are not.
        import numpy as np
        assert isinstance(value, np.integer), 'not Python int/NumPy integer'
        value = int(value)  # Lossless host scalar conversion, never GPU item().
    assert minimum <= value <= maximum, 'integer field out of range'
    return value


def text(value):
    assert type(value) is str and value
    return value


def fields(value, expected):
    assert type(value) is dict and set(value) == set(expected), 'event field schema'


def normalize_event(record):
    """Audit all four actual event producers; preserve original output types."""
    assert type(record) is dict and type(record.get('event')) is str
    kind = record['event']
    if kind == 'dispatch':
        fields(record, ('event', 'manager_role', 'actual_requests', 'actual_tokens',
                        'uniform', 'selected'))
        assert text(record['manager_role']) in ('target', 'other')
        selected = record['selected']
        fields(selected, ('mode', 'tokens', 'requests', 'uniform', 'bucket'))
        assert type(selected['mode']) is str
        assert selected['mode'] in ('NONE', 'PIECEWISE', 'FULL')
        return {'event': kind, 'manager_role': record['manager_role'],
            'actual_requests': integer(record['actual_requests'], minimum=1, maximum=8),
            'actual_tokens': integer(record['actual_tokens'], minimum=1),
            'uniform': integer(record['uniform'], optional=True, minimum=1),
            'selected': {'mode': selected['mode'],
                'tokens': integer(selected['tokens'], minimum=1),
                'requests': integer(selected['requests'], optional=True,
                                    minimum=1, maximum=8),
                'uniform': integer(selected['uniform'], optional=True, minimum=1),
                'bucket': integer(selected['bucket'], optional=True)}}
    if kind == 'real_scheduler':
        fields(record, ('event', 'request_ids', 'sentinel', 'scheduled_tokens',
                        'draft_counts'))
        ids = record['request_ids']
        assert type(ids) is list and len(ids) == len(set(ids)) and ids
        ids = [text(identity) for identity in ids]
        assert type(record['sentinel']) is bool
        assert type(record['scheduled_tokens']) is dict
        assert set(record['scheduled_tokens']) == set(ids)
        assert type(record['draft_counts']) is dict
        assert set(record['draft_counts']) <= set(ids)
        return {'event': kind, 'request_ids': ids, 'sentinel': record['sentinel'],
            'scheduled_tokens': {text(k): integer(v, minimum=1)
                                 for k, v in record['scheduled_tokens'].items()},
            'draft_counts': {text(k): integer(v, maximum=4)
                             for k, v in record['draft_counts'].items()}}
    if kind == 'original_owner_call':
        fields(record, ('event', 'owner', 'role', 'route'))
        assert text(record['role']) in ('target', 'draft')
        assert text(record['route']) == 'original_no_target_capture_ticket'
        return {'event': kind, 'owner': text(record['owner']),
                'role': record['role'], 'route': record['route']}
    if kind == 'unsupported_original':
        fields(record, ('event', 'owner', 'descriptor', 'q_shape'))
        descriptor, shape = record['descriptor'], record['q_shape']
        assert type(descriptor) in (tuple, list) and len(descriptor) == 3
        assert type(shape) is list and len(shape) == 3
        return {'event': kind, 'owner': text(record['owner']),
                'descriptor': tuple(integer(v, minimum=1) for v in descriptor),
                'q_shape': [integer(v, minimum=1) for v in shape]}
    raise AssertionError('unknown host event producer')
