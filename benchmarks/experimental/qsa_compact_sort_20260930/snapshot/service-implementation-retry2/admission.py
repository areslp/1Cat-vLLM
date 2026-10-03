"""Read-only per-cohort/off/shadow gates. No performance/service PASS inference."""
import argparse
import json
from pathlib import Path
from io_contract import save, sha256
from policy import CONSUMERS, DESCRIPTORS, TARGET_NAMES


def reuse_witness(records, required_ids, completed_client_ids):
    """Actual finished remove -> new same-slot add -> host physical binding."""
    removed, added = {}, {}
    for row in sorted(records, key=lambda r: r['sequence']):
        slot, identity = row['slot'], row['scheduler_id']
        if (row['event'] == 'slot_remove' and row['finished']
                and identity in completed_client_ids):
            removed[slot] = row
        elif row['event'] == 'slot_add':
            # Every physical occupant is retained. An intervening sentinel,
            # warmup/unattributed request, or other add invalidates old pairs.
            for key in [key for key in added if key[0] == slot]:
                del added[key]
            old = removed.get(slot)
            previous = row.get('previous_initial_binding')
            if (old and old['scheduler_id'] != identity
                    and row['previous_owner'] == old['scheduler_id']
                    and previous and previous['scheduler_id'] == old['scheduler_id']
                    and row['host_slot_generation'] >= 2):
                added[(slot, identity)] = (old, row)
        elif row['event'] == 'block_append' and row['overwrite']:
            pair = added.get((slot, identity))
            if pair and identity in required_ids:
                assert any(row['scheduler_physical_ids_by_group'])
                assert len(row['expansion_by_group']) == len(row['expanded_counts_after'])
                for ids, expansion, count in zip(row['scheduler_physical_ids_by_group'],
                        row['expansion_by_group'], row['expanded_counts_after']):
                    assert expansion >= 1 and count == len(ids) * expansion
                return {'slot': slot, 'old_scheduler_id': pair[0]['scheduler_id'],
                        'new_scheduler_id': identity,
                        'remove_sequence': pair[0]['sequence'],
                        'add_sequence': pair[1]['sequence'],
                        'append_sequence': row['sequence'],
                        'host_generation': pair[1]['host_slot_generation'],
                        'old_initial_binding': pair[1]['previous_initial_binding'],
                        'new_initial_mapping_sha256': row['physical_mapping_sha256'],
                        'identity_scope': 'actual host slot and physical table binding; no immutable KV proof'}
    raise AssertionError('no finished-owner same-slot reuse with physical binding')


def completed_primary_ids(drain):
    externals = drain['epoch']['external_request_ids']
    bindings = drain['request_id_bindings']
    cohort = drain['client_cohort']
    assert cohort['epoch'] == drain['epoch']['epoch']
    assert cohort['queue_drained'] is True
    rows = cohort['completed_requests']
    assert len(rows) == len(externals)
    assert {row['external_request_id'] for row in rows} == set(externals)
    from identity import Bindings
    checker = Bindings()
    for row in rows:
        external = row['external_request_id']
        assert row['response_id'] == external[:-2]
        assert row['n'] == row['prompt_count'] == 1
        assert row['http_status'] == 200 and row['complete'] is True
        assert checker.bind(bindings[external], externals) == external
    return {bindings[external] for external in externals}


def evaluate(ready, drain, lifecycle=(), completed_client_ids=()):
    mode = ready['mode']
    assert mode == drain['mode'] and mode in ('off', 'shadow')
    assert ready['status'] == 'CAPTURE_READY_SERVICE_UNVERIFIED'
    assert ready['rank'] == drain['rank'] and ready['pid'] == drain['pid']
    assert ready['owners'] == sorted(TARGET_NAMES)
    assert len(ready['nodes']) == 36
    epoch = drain['epoch']; spec = ready['epoch_contract'][epoch['epoch']]
    assert epoch['external_request_ids'] == spec['external_request_ids']
    assert epoch['route_expectation'] == spec['route_expectation']
    counts = drain['counts']
    if mode == 'shadow':
        assert drain['status'] == 'SHADOW_DRAIN_NOT_ADMISSION'
        assert drain['mismatches_zero'] is True
        assert drain['first_bad_node'] is None and drain['first_divergence'] is None
        assert ready['counter_shape'] == [36, 6]
        assert len(counts) == 36 and all(len(row) == 6 for row in counts)
        assert all(not any(row[1:]) for row in counts), 'shadow bit divergence'
        assert all(node['planner'] == 'candidate' for node in ready['nodes'])
    else:
        assert drain['status'] == 'OFF_HOST_DRAIN_DEVICE_UNINSTRUMENTED'
        assert ready['outer_off_host_observer'] is True
        assert counts is None and drain['mismatches_zero'] is None
        assert ready['counter_shape'] is None
        assert ready['private_bytes'] == ready['failure_bank_bytes'] == 0
        assert all(node['planner'] == 'original' for node in ready['nodes'])
    replays = {tuple(row['descriptor']): row['count'] for row in drain['host_replays']}
    assert set(replays) <= set(DESCRIPTORS)
    if mode == 'shadow':
        for index, key in enumerate(sorted(DESCRIPTORS)):
            expected = replays.get(key, 0)
            assert all(row[0] == expected for row in counts[index * 12:(index + 1) * 12]), (
                'device hits != real cached replay attribution', key)
    externals = epoch['external_request_ids']; bindings = drain['request_id_bindings']
    sentinel = drain['sentinel_external_request_id']
    assert sentinel == spec['sentinel_external_request_id']
    assert set(bindings) == set(externals) | {sentinel}
    from identity import Bindings
    checker = Bindings()
    for external, internal in bindings.items():
        assert checker.bind(internal, bindings) == external
    cohort = drain['client_cohort']
    assert cohort['epoch'] == epoch['epoch'] and cohort['queue_drained'] is True
    assert {r['external_request_id'] for r in cohort['completed_requests']} == set(externals)
    assert len(cohort['completed_requests']) == len(externals)
    assert all(r['response_id'] == r['external_request_id'][:-2]
               and r['n'] == r['prompt_count'] == 1 and r['http_status'] == 200
               and r['complete'] is True for r in cohort['completed_requests'])
    observed, consumers, fallback, target_rows = set(), set(), [], []
    scheduler = None
    sentinel_ids = {bindings[sentinel]}
    primary_ids = {bindings[x] for x in externals}
    for event in drain['events']:
        if event['event'] == 'real_scheduler':
            scheduler = event
            ids = set(event['request_ids'])
            assert ids <= set(bindings.values()) and ids
            assert event['sentinel'] == (ids == sentinel_ids)
            assert not (ids & sentinel_ids and ids & primary_ids), 'sentinel mixed with cohort'
            observed.update(ids)
        elif event['event'] == 'dispatch' and event['manager_role'] == 'target':
            assert scheduler is not None, 'dispatch missing real scheduler provenance'
            if scheduler['sentinel']:
                continue  # c1 safe drain is outside candidate/fallback coverage.
            assert set(scheduler['request_ids']) <= primary_ids
            selected = event['selected']
            key = (selected['tokens'], selected['requests'], selected['uniform'])
            eligible = (selected['mode'] == 'FULL' and selected['bucket'] is None
                        and key in DESCRIPTORS)
            row = {**event, 'eligible': eligible, 'scheduler': scheduler}
            target_rows.append(row)
            if eligible:
                c = event['actual_requests']
                assert key == CONSUMERS.get(c)
                assert event['actual_tokens'] == 5 * c and event['uniform'] == 5
                assert len(scheduler['request_ids']) == c
                assert all(n == 5 for n in scheduler['scheduled_tokens'].values())
                assert all(scheduler['draft_counts'].get(k) == 4
                           for k in scheduler['request_ids'])
                consumers.add(c)
            else:
                fallback.append(row)
    assert set(bindings.values()) <= observed, 'unobserved scheduler identity'
    assert target_rows, 'no real primary target dispatch'
    policy = spec['route_expectation']; c = spec['declared_concurrency']
    hits = sum(r[0] for r in counts) if counts is not None else None
    if policy == 'eligible_consumer':
        assert c in consumers, ('required actual consumer absent', c, sorted(consumers))
        assert replays.get(CONSUMERS[c], 0) > 0
        if mode == 'shadow': assert hits > 0, 'zero hit cannot PASS'
    elif policy == 'short_original':
        assert c <= 3 and not consumers and not replays
        assert any(r['actual_requests'] == c and r['uniform'] == 5 for r in fallback)
        if mode == 'shadow': assert hits == 0
    elif policy == 'mixed_original_witness':
        assert any(r['actual_requests'] >= 2 and r['uniform'] is None and
            any(n > 1 and r['scheduler']['draft_counts'].get(k, 0) == 0
                for k, n in r['scheduler']['scheduled_tokens'].items()) and
            any(n == 5 and r['scheduler']['draft_counts'].get(k, 0) == 4
                for k, n in r['scheduler']['scheduled_tokens'].items())
            for r in fallback), 'mixed lacks co-scheduled prefill plus MTP verify decode'
    elif policy == 'fallback_original_only':
        assert not consumers and not replays
        assert all(1 <= r['actual_requests'] <= c and r['uniform'] is None for r in fallback)
        if mode == 'shadow': assert hits == 0
    else: raise AssertionError('unknown frozen route expectation')
    reuse = None
    if spec['scenario_label_not_runtime_proof'] == 'slotreuse':
        reuse = reuse_witness([*lifecycle, *drain['lifecycle']], primary_ids,
                              set(completed_client_ids) | completed_primary_ids(drain))
    shape_only = policy == 'fallback_original_only' and any(
        r['actual_requests'] >= 4 and r['uniform'] is None
        and r['actual_tokens'] == 5 * r['actual_requests']
        and len(r['scheduler']['request_ids']) == r['actual_requests']
        and all(n == 5 for n in r['scheduler']['scheduled_tokens'].values())
        and all(r['scheduler']['draft_counts'].get(k, 0) == 0
                for k in r['scheduler']['request_ids']) for r in fallback)
    return {'actual_consumers': sorted(consumers), 'candidate_hits': hits,
            'device_counter_scope': 'instrumented' if mode == 'shadow' else 'ABSENT',
            'fallback_dispatches': len(fallback), 'bindings': bindings,
            'route_expectation': policy, 'route_gate': 'PASS', 'slot_reuse': reuse,
            'shape_only_verify_guard_witness': shape_only}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ready-dir', required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--drain-id'); group.add_argument('--all-epochs', action='store_true')
    p.add_argument('--output', required=True)
    p.add_argument('--require-hit', action='store_true')
    p.add_argument('--require-slot-reuse', action='store_true')
    a = p.parse_args(); root = Path(a.ready_dir)
    result = {'status': 'RUNNING', 'epochs': [], 'files': []}
    try:
        ready = [json.loads((root / f'capture-ready-rank{r}.json').read_text()) for r in range(4)]
        contract = ready[0]['epoch_contract']
        assert all(r['epoch_contract'] == contract for r in ready)
        assert [r['rank'] for r in ready] == [0, 1, 2, 3]
        assert len({r['pid'] for r in ready}) == 4
        assert len({r['config_sha256'] for r in ready}) == 1
        assert len({r['mode'] for r in ready}) == 1
        assert len(contract) == 20
        epochs = list(contract) if a.all_epochs else [a.drain_id]
        if a.all_epochs: assert len(epochs) == 20
        for epoch in epochs:
            assert epoch in contract
            rows, mappings = [], []
            for rank in range(4):
                path = root / f'drain-{epoch}-rank{rank}.json'
                drain = json.loads(path.read_text())
                prior, completed = [], set()
                for old in list(contract)[:list(contract).index(epoch)]:
                    old_path = root / f'drain-{old}-rank{rank}.json'
                    if old_path.exists():
                        old_drain = json.loads(old_path.read_text())
                        assert old_drain['rank'] == rank and old_drain['pid'] == ready[rank]['pid']
                        assert old_drain['epoch']['external_request_ids'] == contract[old]['external_request_ids']
                        prior.extend(old_drain['lifecycle'])
                        completed.update(completed_primary_ids(old_drain))
                row = evaluate(ready[rank], drain, prior, completed)
                if a.require_hit and ready[rank]['mode'] == 'shadow':
                    assert row['candidate_hits'] > 0
                rows.append({'rank': rank, **row}); mappings.append(row['bindings'])
                result['files'].append({'path': str(path), 'sha256': sha256(path)})
            assert all(m == mappings[0] for m in mappings), 'rank identity mismatch'
            result['epochs'].append({'epoch': epoch, 'ranks': rows})
        if not a.all_epochs:
            result['ranks'] = result['epochs'][0]['ranks']
        if a.require_slot_reuse:
            assert a.all_epochs
            assert all(any(e['ranks'][rank]['slot_reuse'] for e in result['epochs'])
                       for rank in range(4))
        if a.all_epochs:
            fallback = [e for e in result['epochs'] if e['ranks'][0][
                'route_expectation'] == 'fallback_original_only']
            assert len(fallback) == 2
            assert all(any(e['ranks'][r]['shape_only_verify_guard_witness']
                           for e in fallback) for r in range(4)), (
                'no >=4 requests shape-only verify guard witness')
        result['files'].extend({'path': str(root / f'capture-ready-rank{r}.json'),
                               'sha256': sha256(root / f'capture-ready-rank{r}.json')}
                              for r in range(4))
        result['status'] = ('SHADOW_COHORT_GATES_PASS_BENCHMARK_UNVERIFIED'
                            if ready[0]['mode'] == 'shadow' else
                            'OFF_HOST_COHORT_GATES_PASS_DEVICE_UNINSTRUMENTED')
        code = 0
    except BaseException as error:
        result.update(status='STOP_SERVICE_GATE', error=repr(error)); code = 2
    save(a.output, result)
    return code


if __name__ == '__main__': raise SystemExit(main())
