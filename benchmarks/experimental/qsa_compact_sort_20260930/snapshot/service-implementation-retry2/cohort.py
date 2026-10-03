"""Read-only completed client-cohort receipt; actual HTTP driver owns its proof."""
import json
from pathlib import Path
from io_contract import require_hash


def validate_receipt(path, expected_sha, epoch, directory):
    path = Path(path).resolve()
    path.relative_to(Path(directory).resolve())
    assert path.stat().st_size <= 4 * 1024**2
    require_hash(path, expected_sha)
    receipt = json.loads(path.read_text())
    assert receipt['epoch'] == epoch['epoch']
    assert receipt['queue_drained'] is True
    completed = receipt['completed_requests']
    assert len(completed) == len(epoch['external_request_ids'])
    assert {r['external_request_id'] for r in completed} == set(epoch['external_request_ids'])
    assert all(r['http_status'] == 200 and r['complete'] is True for r in completed)
    assert all(r['response_id'] == r['external_request_id'][:-2]
               and r['n'] == 1 and r['prompt_count'] == 1 for r in completed)
    return receipt
