"""CPU-only frozen-event parser; no transport, retries or inferred completion."""
import json
import math
import statistics

from stream_ids import check_chunk, ids


def parse(body, header, events, *, terminal, elapsed_s, expected_cancel=False,
          cancel_contract=None):
    identity = ids(body, header)
    assert type(elapsed_s) in (int, float)
    assert math.isfinite(elapsed_s) and elapsed_s > 0
    assert isinstance(events, list)
    tokens, positive, finishes, usages = [], [], [], []
    done, previous = False, -1.0
    for ordinal, event in enumerate(events):
        timestamp, raw = event['elapsed_s'], event['data']
        assert type(timestamp) in (int, float)
        assert math.isfinite(timestamp) and 0 <= timestamp <= elapsed_s
        assert timestamp >= previous and not done
        previous = timestamp
        assert isinstance(raw, str)
        if raw == '[DONE]':
            done = True
            continue
        chunk = json.loads(raw)
        check_chunk(chunk, identity)
        usage = chunk.get('usage')
        if usage is not None:
            assert not chunk.get('choices'), 'usage must be a separate block'
            usages.append({'ordinal': ordinal, 'usage': usage})
            assert len(usages) == 1, 'duplicate usage'
        choices = chunk.get('choices', [])
        if not choices:
            continue
        assert not usages, 'choice after final usage'
        choice = choices[0]
        emitted = choice.get('token_ids') or []
        assert isinstance(emitted, list)
        if emitted:
            assert not finishes, 'new tokens after terminal finish'
            tokens.extend(emitted)
            positive.append({'elapsed_s': timestamp, 'tokens': len(emitted),
                             'ordinal': ordinal})
        finish = choice.get('finish_reason')
        if finish is not None:
            assert isinstance(finish, str) and finish
            finishes.append({'ordinal': ordinal, 'reason': finish})
            assert len(finishes) == 1, 'duplicate finish'
    base = {'identity': identity, 'raw_events': events, 'terminal': terminal,
            'elapsed_s': elapsed_s, 'output_token_ids': tokens,
            'positive_chunks': positive, 'finishes': finishes, 'usages': usages,
            'done': done, 'ttft_s': positive[0]['elapsed_s'] if positive else None}
    if expected_cancel:
        assert cancel_contract == {'positive_chunks': 2,
                                   'cut': 'after-positive-chunk-count'}
        assert terminal == 'client_cancelled' and not done
        assert not finishes and not usages, 'cancel was already completed'
        assert len(positive) == cancel_contract['positive_chunks']
        assert positive[-1]['ordinal'] == len(events) - 1, 'late cancellation cut'
        return {**base, 'status': 'EXPECTED_CANCELLED_NOT_COMPLETED',
                'completed': False, 'step_ms': None, 'tokens_per_step': None,
                'cancel_contract': cancel_contract,
                'actual_cut': {'event_ordinal': positive[-1]['ordinal'],
                    'positive_chunks': len(positive), 'output_tokens': len(tokens),
                    'last_event_elapsed_s': positive[-1]['elapsed_s'],
                    'closed_elapsed_s': elapsed_s}}
    assert cancel_contract is None
    assert terminal == 'eof' and done, 'missing ordinary stream completion'
    assert len(finishes) == len(usages) == 1
    assert finishes[0]['ordinal'] < usages[0]['ordinal'] < len(events) - 1
    usage = usages[0]['usage']
    expected = body['max_tokens']
    assert type(expected) is int and expected > 0
    assert len(tokens) == usage['completion_tokens'] == expected
    assert usage['prompt_tokens'] == len(body['prompt'])
    assert usage['total_tokens'] == len(body['prompt']) + expected
    if body.get('ignore_eos'):
        assert finishes[0]['reason'] == 'length'
    times = [chunk['elapsed_s'] for chunk in positive]
    # perf38: omit the first positive-to-second positive interval.
    intervals = [1000 * (b - a) for a, b in zip(times[1:], times[2:])]
    assert all(math.isfinite(value) and value > 0 for value in intervals)
    return {**base, 'status': 'COMPLETE', 'completed': True,
            'step_ms': statistics.median(intervals) if intervals else None,
            'tokens_per_step': ((expected - positive[0]['tokens']) /
                                (len(positive) - 1)
                                if len(positive) > 1 else None)}
