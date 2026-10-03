"""Finite sparse Counter contract for the pinned on-mode E7 exporter only."""
import json

RUNTIME_SHA256 = '6c901559b998c3777fbcde77f94f373d8ad13327d50b4b2f352ea94266bb901d'
IDENTITY = ('pid', 'rank', 'ranks', 'mode', 'module_path', 'package_hashes')

# runtime.py:57-80 finite parameter_reason returns; model_runner.py:1154-1162
# contributes exactly the two unsupported reasons. There is no external free
# string allowance. Counter() starts empty, += is the sole update operation,
# and dict(_COUNTS) exports only keys that have actually been touched.
PARAMETER_REASONS = frozenset({
    'feature:min_p', 'feature:presence_penalty', 'feature:frequency_penalty',
    'feature:repetition_penalty', 'feature:min_tokens', 'feature:logprobs',
    'feature:prompt_logprobs', 'feature:logit_bias', 'feature:allowed_token_ids',
    'feature:structured_outputs', 'feature:thinking_token_budget',
    'feature:extra_args', 'feature:logprob_token_ids', 'feature:bad_words',
    'feature:_bad_words_token_ids', 'feature:logits_processors',
    'greedy_or_temperature', 'top_k', 'top_p',
})
UNSUPPORTED_REASONS = frozenset({'multimodal_unvalidated', 'lora_unvalidated'})
BATCH_REASONS = frozenset({
    'request_count', 'not_pure_mtp4', 'logits_shape', 'prefill', 'draft_shape',
    'logits_mapping',
})
ON_KEYS = (frozenset({
    'requests_seen', 'requests_supported', 'hook_steps', 'fallback_capture',
    'fallback_static_layout', 'fallback_vocab_layout', 'fallback_features',
    'eligible_steps', 'fallback_dtype', 'fallback_structure',
    'baseline_local_reused', 'fallback_certificate', 'compressed_steps',
}) | {'request_' + reason for reason in PARAMETER_REASONS | UNSUPPORTED_REASONS}
   | {'fallback_' + reason for reason in BATCH_REASONS})


def sparse_delta(before, after):
    """Missing-before is 0 only for source-enumerated E7 keys; never delete."""
    if type(before) is not dict or type(after) is not dict:
        raise TypeError('E7 Counter dictionaries required')
    for row in (before, after):
        if any(type(k) is not str or k not in ON_KEYS for k in row):
            raise ValueError('unknown or non-on-mode E7 Counter key')
        if any(type(v) is not int or v < 0 for v in row.values()):
            raise ValueError('E7 Counter must be nonnegative Python integer')
    if before.keys() - after.keys():
        raise ValueError('E7 Counter key removed')
    delta = {key: after[key] - before.get(key, 0) for key in sorted(after)}
    if any(value < 0 for value in delta.values()):
        raise ValueError('E7 Counter regressed')
    return delta


def e7_deltas(before, after):
    """Preserve raw rows, exact exporter identity and full four-rank deltas."""
    if len(before) != 4 or len(after) != 4:
        raise ValueError('four E7 rank rows required')
    for rows in (before, after):
        if len({json.dumps(row['counters'], sort_keys=True) for row in rows}) != 1:
            raise ValueError('full E7 Counter values differ across ranks')
    vectors = []
    for rank, (old, new) in enumerate(zip(before, after)):
        if old['rank'] != rank or new['rank'] != rank:
            raise ValueError('E7 rank order changed')
        if any(old[key] != new[key] for key in IDENTITY):
            raise ValueError('E7 source/PID identity changed')
        if (old['mode'] != 'on' or old['ranks'] != before[0]['ranks']
                or len(set(old['ranks'])) != 4
                or old['package_hashes'].get('runtime.py') != RUNTIME_SHA256):
            raise ValueError('pinned on-mode E7 runtime identity required')
        vectors.append(sparse_delta(old['counters'], new['counters']))
    if len({json.dumps(vector, sort_keys=True) for vector in vectors}) != 1:
        raise ValueError('full E7 delta differs across ranks')
    return vectors
