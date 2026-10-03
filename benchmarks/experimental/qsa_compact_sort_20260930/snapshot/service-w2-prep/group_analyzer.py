"""Pure CPU matched-group analysis; synthetic inputs do not admit a service."""
import hashlib
import json
import math
import random
import statistics

DRAWS = 10000


def finite_positive(value):
    assert type(value) in (int, float)
    assert math.isfinite(value) and value > 0, 'invalid finite positive metric'
    return value


def quantile(values, probability):
    values = sorted(values)
    point = (len(values) - 1) * probability
    lower, upper = math.floor(point), math.ceil(point)
    return values[lower] * (upper - point) + values[upper] * (point - lower) \
        if lower != upper else values[lower]


def estimates(triples, *, throughput=False):
    m0, mb, m2 = (statistics.median(column) for column in zip(*triples))
    ma = (m0 + m2) / 2
    improvement = 100 * (1 - mb / ma)
    drift = 100 * abs(m2 - m0) / ma
    regression = improvement if throughput else -improvement
    return {'mA0': m0, 'mB': mb, 'mA2': m2, 'mA': ma,
            'I': improvement, 'D': drift, 'I_minus_D': improvement - drift,
            'R': regression}


def analyze(row_id, triples, *, kind, throughput=False):
    assert kind in ('primary', 'key', 'confirmation')
    assert isinstance(row_id, str) and row_id
    assert len(triples) >= 3
    assert all(len(row) == 3 for row in triples)
    triples = [[finite_positive(value) for value in row] for row in triples]
    seed = int.from_bytes(hashlib.sha256(
        b'W2-CI-v1:1729:' + row_id.encode()).digest()[:8], 'big')
    rng, draw_hash = random.Random(seed), hashlib.sha256()
    samples = {key: [] for key in ('I', 'D', 'I_minus_D', 'R')}
    for _ in range(DRAWS):
        ordinals = [rng.randrange(len(triples)) for _ in triples]
        draw_hash.update(json.dumps(ordinals, separators=(',', ':')).encode())
        draw_hash.update(b'\n')
        estimate = estimates([triples[index] for index in ordinals],
                             throughput=throughput)
        for key in samples:
            samples[key].append(estimate[key])
    point = estimates(triples, throughput=throughput)
    ci = {key: [quantile(values, 0.025), quantile(values, 0.975)]
          for key, values in samples.items()}
    if kind == 'primary':
        assert not throughput, 'primary metric is latency'
        status = ('NO_GO' if point['I'] < 0.5 or point['I_minus_D'] <= 0
                  else 'PASS_INITIAL_REQUIRES_INDEPENDENT_CONFIRMATION'
                  if ci['I'][0] >= 0.5 and ci['I_minus_D'][0] > 0
                  else 'INCONCLUSIVE')
    elif kind == 'confirmation':
        assert not throughput
        status = ('NO_GO' if point['I'] <= 0 else 'PASS_SEPARATE_CONFIRMATION'
                  if ci['I'][0] > 0 else 'INCONCLUSIVE')
    else:
        status = ('NO_GO' if point['R'] > 1 else 'PASS_KEY_ESTIMATE'
                  if ci['R'][1] <= 1 else 'INCONCLUSIVE')
    return {'status': status, 'row_id': row_id, 'kind': kind, 'point': point,
            'ci_95_percentile': ci, 'matched_group_n': len(triples),
            'direction': 'throughput' if throughput else 'latency',
            'draws': DRAWS, 'seed': seed, 'draw_ordinals_sha256': draw_hash.hexdigest(),
            'algorithm': 'joint matched-triple resample; linear quantile',
            'scope': 'finite-sample estimate; no service or SLA admission'}


def bind_groups(frozen, observed, *, arms=('A0', 'B', 'A2')):
    """Require the whole preassigned inventory; never re-pair successes."""
    assert arms in (('A0', 'B', 'A2'), ('A3', 'B3', 'A4'))
    assert frozen
    expected = {(row['row_id'], row['arm'], row['ordinal']): row for row in frozen}
    assert len(expected) == len(frozen), 'duplicate frozen group'
    actual = {}
    for row in observed:
        key = (row['row_id'], row['arm'], row['ordinal'])
        assert key in expected and key not in actual, 'unknown/duplicate group'
        contract = expected[key]
        assert row['request_ids'] == contract['request_ids'], 'group identity changed'
        required = contract['required_checks']
        assert required and set(row['checks']) == set(required)
        assert row['status'] == 'COMPLETE'
        assert all(value is True for value in row['checks'].values())
        assert row['request_terminal_states'] == contract['terminal_states']
        assert row['body_sha256s'] == contract['body_sha256s']
        assert row['arrival_rule_sha256'] == contract['arrival_rule_sha256']
        actual[key] = finite_positive(row['group_metric'])
    assert actual.keys() == expected.keys(), 'missing or partial matched group'
    triples = {}
    for row_id in sorted({key[0] for key in expected}):
        ordinals = {key[2] for key in expected if key[0] == row_id}
        assert ordinals == set(range(len(ordinals))), 'unassigned group ordinal'
        rows = []
        for ordinal in sorted(ordinals):
            keys = [(row_id, arm, ordinal) for arm in arms]
            assert all(key in actual for key in keys), 'incomplete arm membership'
            rows.append([actual[key] for key in keys])
        triples[row_id] = rows
    return triples
