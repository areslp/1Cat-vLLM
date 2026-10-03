"""One pinned time-only amendment to the unchanged original W2 contract."""
import json
from pathlib import Path

from frozen import PINS, require, sha

ROOT = Path(__file__).resolve().parent
AMENDMENT = ROOT / 'execution-budget-revision.json'
AMENDMENT_SHA256 = '70c9ce7e7d743b219657ee11fa6428af4e8a925e3c16d567e2b94244de212c28'
ORIGINAL_BUDGETS = {
    'boot_including_maturity_per_arm': 960, 'maturity': 360,
    'matrix_per_arm': 1800, 'B_stability': 900, 'inner': 9900,
    'kill_grace': 20, 'expiry': 10020, 'restore': 2100, 'outer': 12900,
}
REPLACEMENTS = {
    'matrix_per_arm': 3600, 'inner': 15300, 'expiry': 15420, 'outer': 18300,
}
EFFECTIVE = {**ORIGINAL_BUDGETS, **REPLACEMENTS}


def model_runtime_seconds(arm):
    if arm not in ('A0', 'B', 'A2'):
        raise ValueError('only the frozen three model arms have derived time budgets')
    # Existing per-model margins stay unchanged. B's complete stability
    # wrapper is 1110s, independent of its unchanged 900s measurement bound.
    extra = 1110 + 130 if arm == 'B' else 240
    return EFFECTIVE['boot_including_maturity_per_arm'] + EFFECTIVE['matrix_per_arm'] + extra


def validate_revision(value, original):
    # Equal bool/float values must not pass as integer deadlines.
    for limits, expected in (
        (original, ORIGINAL_BUDGETS),
        (value['original_budgets_seconds'], ORIGINAL_BUDGETS),
        (value['replacements_seconds'], REPLACEMENTS),
        (value['effective_budgets_seconds'], EFFECTIVE),
    ):
        if limits != expected or any(type(v) is not int for v in limits.values()):
            raise ValueError('only the four registered integer time replacements are permitted')
    if (value['schema'] != 'step58-w2-execution-time-amendment-v1'
            or value['original_contract'] != {'name': 'contract.json', 'sha256': PINS['contract.json']}
            or value['original_matrix'] != {'name': 'matrix.frozen.json', 'sha256': PINS['matrix.frozen.json']}
            or value['launch_authorized_by_this_file'] is not False
            or value['partial_run3_reused'] is not False
            or type(value['new_HTTP_requests']) is not int or value['new_HTTP_requests'] != 0):
        raise ValueError('original fixed source identity or amendment scope differs')
    audit = value['time_audit']
    terms = [EFFECTIVE['inner'], EFFECTIVE['kill_grace'],
             550 + 5, 20 + 5, EFFECTIVE['restore'] + 20, 60 + 5, 30 + 5]
    expiry_stop_start = 7 * (5 + 1 + 50 + 5) + 20 + 5
    expected_audit = {
        'outer_terms_s': terms, 'outer_chain_s': sum(terms),
        'outer_margin_s': EFFECTIVE['outer'] - sum(terms),
        'expiry_restore_body_and_trap_upper_s': expiry_stop_start,
        'expiry_TimeoutStartSec_s': 470, 'expiry_TimeoutStopSec_s': 30,
        'expiry_timer_minutes': 257,
        'expiry_offset_after_inner_s': EFFECTIVE['expiry'] - EFFECTIVE['inner'],
        'derived_model_RuntimeMaxSec': {arm: model_runtime_seconds(arm) for arm in ('A0', 'B', 'A2')},
        'derived_model_formulas': {
            'A0': '960+3600+240=4800 (old 960+1800+240=3000)',
            'B': '960+3600+1110+130=5800 (old 960+1800+1110+130=4000)',
            'A2': '960+3600+240=4800 (old 960+1800+240=3000)',
        },
        'HTTP_matrix_RuntimeMaxSec_s': EFFECTIVE['matrix_per_arm'] + 10,
        'HTTP_matrix_command_budget_s': EFFECTIVE['matrix_per_arm'] + 30,
        'inner_phase_terms_s': [3 * (960 + EFFECTIVE['matrix_per_arm'] + 30), 1110 + 30, 60],
        'inner_phase_chain_s': 3 * (960 + EFFECTIVE['matrix_per_arm'] + 30) + 1140 + 60,
        'inner_remaining_nonphase_s': EFFECTIVE['inner'] - (3 * (960 + EFFECTIVE['matrix_per_arm'] + 30) + 1140 + 60),
    }
    if (audit != expected_audit or sum(terms) >= EFFECTIVE['outer']
            or expiry_stop_start >= audit['expiry_TimeoutStartSec_s']
            or audit['inner_remaining_nonphase_s'] != 330
            or 60 * audit['expiry_timer_minutes'] != EFFECTIVE['expiry']):
        raise ValueError('complete recovery/expiry time chain differs')
    return dict(EFFECTIVE)


def effective_budgets():
    if AMENDMENT.is_symlink() or sha(AMENDMENT) != AMENDMENT_SHA256:
        raise ValueError('the unique frozen execution budget amendment changed')
    value = json.loads(AMENDMENT.read_text())
    original = json.loads(require('contract.json').read_text())
    # Full contract and matrix hashes preserve every statistic, gate, payload
    # and request count; their original time fields remain unchanged.
    require('matrix.frozen.json')
    return validate_revision(value, original['initial_budgets_seconds'])


def binding():
    from common import PACKAGE
    effective_budgets()
    return {'path': str(PACKAGE / AMENDMENT.name), 'sha256': AMENDMENT_SHA256}


BUDGETS = effective_budgets()
