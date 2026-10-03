"""Explicit upper bounds, not expected elapsed time or an ETA."""
BUDGETS = {
    'inner_s': 27600, 'inner_kill_s': 20, 'expiry_s': 27720,
    'outer_s': 31200, 'restore_s': 2100, 'restore_kill_s': 20,
    'ready_s': 1080, 'maturity_s': 360, 'startup_binding_s': 60,
    'request_s': 600, 'group_s': 660, 'client_s': 7200,
    'client_runtime_s': 7210, 'client_wrapper_s': 7230,
    'model_runtime_s': 8730, 'start_s': 30, 'final_guard_s': 30,
    'stop_s': 50, 'drain_s': 65, 'inner_cleanup_s': 750,
    'outer_cleanup_s': 700, 'preflight_stop_s': 300,
    'end_audit_s': 60, 'analysis_s': 60,
}
ARM_CHAIN_S = sum(BUDGETS[k] for k in (
    'start_s', 'ready_s', 'startup_binding_s', 'client_wrapper_s',
    'final_guard_s', 'stop_s', 'drain_s'))
INNER_CHAIN_S = (3 * ARM_CHAIN_S + BUDGETS['preflight_stop_s'] +
                 BUDGETS['end_audit_s'] + BUDGETS['analysis_s'] +
                 BUDGETS['inner_cleanup_s'])
OUTER_CHAIN_S = (BUDGETS['inner_s'] + BUDGETS['inner_kill_s'] +
                 BUDGETS['outer_cleanup_s'] + 5 + 20 + 5 +
                 BUDGETS['restore_s'] + BUDGETS['restore_kill_s'] +
                 60 + 5 + 30 + 5)


def validate():
    if (ARM_CHAIN_S != 8545 or INNER_CHAIN_S != 26805 or
            INNER_CHAIN_S > BUDGETS['inner_s'] or OUTER_CHAIN_S != 30570 or
            OUTER_CHAIN_S > BUDGETS['outer_s'] or
            BUDGETS['expiry_s'] <= BUDGETS['inner_s'] + 20 or
            BUDGETS['model_runtime_s'] <= ARM_CHAIN_S):
        raise ValueError('explicit cross-context timing envelope does not close')
    return {'per_arm_s': ARM_CHAIN_S, 'inner_chain_s': INNER_CHAIN_S,
            'inner_margin_s': BUDGETS['inner_s'] - INNER_CHAIN_S,
            'outer_chain_s': OUTER_CHAIN_S,
            'outer_margin_s': BUDGETS['outer_s'] - OUTER_CHAIN_S,
            'bounds_only_not_ETA': True}
