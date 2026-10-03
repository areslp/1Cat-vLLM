"""Separate W2 SSE identity contract; source-derived, no HTTP or model imports."""
import re


def ids(body, header):
    base = body['request_id']
    assert isinstance(base, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,120}', base)
    assert header == base, 'frozen X-Request-Id must match body'
    assert body['stream'] is True
    assert body.get('n', 1) == body.get('best_of', 1) == 1
    assert body.get('use_beam_search', False) is False
    assert body['stream_options'] == {'include_usage': True}
    assert body['return_token_ids'] is True
    prompt = body['prompt']
    assert isinstance(prompt, list) and prompt
    assert all(type(token) is int and token >= 0 for token in prompt)
    return {'client_id': base, 'response_id': 'cmpl-' + base,
            'external_engine_id': 'cmpl-' + base + '-0'}


def check_chunk(chunk, expected):
    assert 'error' not in chunk
    assert chunk['id'] == expected['response_id'], 'foreign SSE response ID'
    choices = chunk.get('choices', [])
    assert isinstance(choices, list) and len(choices) <= 1
    if choices:
        assert choices[0]['index'] == 0, 'unexpected prompt/completion index'
        tokens = choices[0].get('token_ids') or []
        assert all(type(token) is int and token >= 0 for token in tokens)
    return True
