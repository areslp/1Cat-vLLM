"""Bounded create-only receipts and source/binary identity."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f'identity mismatch {path}: {actual} != {expected}')
    return actual


def save(path, value, max_bytes=8 * 1024 * 1024):
    raw = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    assert len(raw) <= max_bytes, 'receipt byte cap'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
    path.chmod(0o600)


def validate_config(cfg):
    assert cfg['diagnostic_host_audit'] is False or cfg['mode'] == 'off'
    assert cfg['mode'] in ('off', 'shadow', 'on')
    assert cfg['max_private_bytes_per_rank'] <= 32 * 1024**2
    assert cfg['max_failure_bank_bytes_per_rank'] <= 64 * 1024**2
    assert cfg['max_failure_artifact_bytes_per_rank'] <= 64 * 1024**2
    assert cfg['max_added_allocated_bytes_per_rank'] <= 256 * 1024**2
    assert cfg['max_added_reserved_bytes_per_rank'] <= 512 * 1024**2
    assert cfg['max_host_events_per_rank'] <= 4096
    assert cfg['max_lifecycle_events_per_rank'] <= 4096
    assert cfg['max_lifecycle_bytes_per_rank'] <= 4 * 1024**2
    assert cfg['max_drains_per_rank'] <= 20
    assert len(cfg['source_pins']) == 13
    assert cfg['max_shadow_seconds'] <= 900
    assert len(cfg['epochs']) <= 20
    ids = []
    for key, spec in cfg['epochs'].items():
        assert 1 <= spec['declared_concurrency'] <= 8
        assert len(spec['external_request_ids']) == spec['declared_concurrency']
        assert spec['route_expectation'] in (
            'eligible_consumer', 'short_original', 'mixed_original_witness',
            'fallback_original_only')
        ids.extend(spec['external_request_ids'])
        ids.append(spec['sentinel_external_request_id'])
    assert len(ids) == len(set(ids)), 'request/sentinel IDs cannot be reused'
    assert cfg['required_kv_tokens'] == 663816
    assert cfg['required_flags'] == {
        'ONECAT_QSA48': 'split', 'VLLM_SM70_QSA_GROUPED_PAGE4': '1',
        'VLLM_SM70_QSA_GROUPED_PAD_FIX': '0', 'HOST_INSIGHT_STAGE_MARKERS': '1'}
    return cfg
