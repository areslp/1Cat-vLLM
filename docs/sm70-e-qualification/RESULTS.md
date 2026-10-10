# Fixed-compiler serving results

The control is frozen upstream plus the same two compatibility fixes. Both arms use the ordinary deterministic compiler configuration recorded here. Negative latency changes mean E is faster. All qualified cohorts remain included.

Each arm completed 111 cold cohorts (37 groups × 3 repeats) and six measured cached reuses (32K/64K × 3); two prefix-priming cohorts per arm are excluded from cached timing. No unqualified cohort was excluded.

| Case | U wall (s) | E wall (s) | Wall change | Prefill change | TTFT change | E vs D wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode_1024_8_0 | 39.7941 | 28.3520 | -28.75% | -5.41% | -3.20% | +4.33% |
| main_1024_1_0 | 2.1576 | 2.0724 | -3.95% | -0.33% | -0.33% | -3.70% |
| main_1024_1_1 | 2.1916 | 2.1059 | -3.91% | -0.85% | -0.63% | +5.59% |
| main_1024_4_0 | 5.0666 | 3.9556 | -21.93% | -4.17% | -3.78% | -4.08% |
| main_1024_8_0 | 7.6840 | 5.9675 | -22.34% | -5.45% | -3.74% | +6.21% |
| main_1024_8_1 | 7.6168 | 6.0509 | -20.56% | -13.26% | -3.75% | +2.99% |
| main_32768_1_0 | 12.6874 | 12.1100 | -4.55% | -4.56% | -4.48% | -1.66% |
| main_65536_1_0 | 24.7088 | 23.5948 | -4.51% | -4.50% | -4.48% | -1.00% |
| prefill_main_1024_1_0 | 0.3438 | 0.3409 | -0.86% | -0.68% | -1.09% | -1.61% |
| prefill_main_1024_4_0 | 1.2773 | 1.2304 | -3.67% | -3.10% | -2.58% | +6.18% |
| prefill_main_1024_8_0 | 2.4692 | 2.3639 | -4.26% | -4.95% | -3.85% | +0.15% |
| prefill_main_32768_1_0 | 11.1321 | 10.6266 | -4.54% | -4.70% | -4.57% | -0.04% |
| prefill_main_65536_1_0 | 23.1317 | 22.1034 | -4.45% | -4.55% | -4.47% | -0.25% |

D is the prior production reference with its prior compiler configuration. E versus D is an operational comparison, not patch-only attribution.

## Reused prefixes

| Case | U wall (s) | E wall (s) | Wall change |
| --- | ---: | ---: | ---: |
| cached/main_32768_1_0 | 2.3438 | 2.2531 | -3.87% |
| cached/main_65536_1_0 | 2.5665 | 2.4818 | -3.30% |

## All historical single-request inputs

Summary versus U: {"equal_weight_geomean_wall_change_pct": -3.911859760452918, "faster_over_3pct": ["main_1024_1_0", "main_1024_1_1", "regression_c1_00", "regression_c1_01", "regression_c1_02", "regression_c1_03", "regression_c1_04", "regression_c1_05", "regression_c1_06", "regression_c1_07", "regression_c1_08", "regression_c1_09", "regression_c1_10", "regression_c1_11", "regression_c1_12", "regression_c1_13", "regression_c1_14", "regression_c1_15", "regression_c1_16", "regression_c1_17", "regression_c1_18", "regression_c1_19", "regression_c1_20", "regression_c1_21", "regression_c1_22", "regression_c1_23"], "historical_C1_cases": 26, "same_output_signature_sets": 26, "slower_any_amount": 0, "slower_over_3pct": []}

Summary versus D: {"equal_weight_geomean_wall_change_pct": -0.29988596937807754, "faster_over_3pct": ["main_1024_1_0", "regression_c1_04", "regression_c1_09", "regression_c1_13", "regression_c1_17", "regression_c1_19", "regression_c1_21"], "historical_C1_cases": 26, "same_output_signature_sets": 4, "slower_any_amount": 11, "slower_over_3pct": ["main_1024_1_1", "regression_c1_11", "regression_c1_14", "regression_c1_15", "regression_c1_18", "regression_c1_22"]}

| Case | Wall vs U | Wall vs D | MTP rounds vs U | Decode/round vs U | Same output set |
| --- | ---: | ---: | ---: | ---: | --- |
| main_1024_1_0 | -3.95% | -3.70% | +0.00% | -4.62% | True |
| main_1024_1_1 | -3.91% | +5.59% | +0.00% | -4.50% | True |
| regression_c1_00 | -3.79% | +0.72% | +0.00% | -4.59% | True |
| regression_c1_01 | -3.94% | -2.41% | +0.00% | -4.61% | True |
| regression_c1_02 | -3.90% | -0.90% | +0.00% | -4.55% | True |
| regression_c1_03 | -4.03% | +2.69% | +0.00% | -4.86% | True |
| regression_c1_04 | -3.95% | -7.91% | +0.00% | -4.62% | True |
| regression_c1_05 | -3.90% | -0.84% | +0.00% | -4.58% | True |
| regression_c1_06 | -3.86% | -0.87% | +0.00% | -4.55% | True |
| regression_c1_07 | -3.90% | -2.54% | +0.00% | -4.71% | True |
| regression_c1_08 | -3.84% | -2.49% | +0.00% | -4.62% | True |
| regression_c1_09 | -3.86% | -3.80% | +0.00% | -4.53% | True |
| regression_c1_10 | -3.84% | +0.81% | +0.00% | -4.46% | True |
| regression_c1_11 | -3.94% | +4.80% | +0.00% | -4.75% | True |
| regression_c1_12 | -3.94% | -2.34% | +0.00% | -4.77% | True |
| regression_c1_13 | -4.05% | -6.83% | +0.00% | -4.80% | True |
| regression_c1_14 | -3.88% | +12.53% | +0.00% | -4.50% | True |
| regression_c1_15 | -3.91% | +10.20% | +0.00% | -4.53% | True |
| regression_c1_16 | -3.89% | -2.42% | +0.00% | -4.73% | True |
| regression_c1_17 | -3.77% | -7.50% | +0.00% | -4.47% | True |
| regression_c1_18 | -3.93% | +12.94% | +0.00% | -4.60% | True |
| regression_c1_19 | -3.91% | -9.07% | +0.00% | -4.55% | True |
| regression_c1_20 | -4.05% | +0.61% | +0.00% | -4.62% | True |
| regression_c1_21 | -3.98% | -8.66% | +0.00% | -4.83% | True |
| regression_c1_22 | -3.89% | +5.47% | +0.00% | -4.58% | True |
| regression_c1_23 | -3.92% | +2.58% | +0.00% | -4.66% | True |

MTP work and generated output can change. Decode time per draft round is a serving-work proxy, not an isolated GPU kernel timer. The per-case JSON contains every repeat, output digest, cache count and phase count.

## Functional results

- D: 10/14 golden-answer cases. Failures: q_json_boolean_0, q_json_fields_0, q_parentheses_0, q_remainder_0.
- U-fixed: 10/14 golden-answer cases. Failures: q_json_boolean_0, q_json_fields_0, q_parentheses_0, q_remainder_0.
- E-fixed: 10/14 golden-answer cases. Failures: q_json_boolean_0, q_json_fields_0, q_parentheses_0, q_remainder_0.

New E failures and missing cases are reported separately in `FIXED_PERFORMANCE_ANALYSIS.json`; protocol, retrieval, concurrent isolation and image/video outcomes are included there.

## Limits

There is one process startup per fixed arm. Three repeats describe this matrix; they do not establish restart reproducibility, production p99 or broad model quality. Concurrent inputs are 1K tokens; 32K/64K inputs use concurrency one. Earlier automatic-tuning data remains separate.

Request prefill time is measured from first scheduling to first output. Concurrent batching can change this interval without the same change in request TTFT. Raw prefill, queue, TTFT and the scheduler-iteration strata are all retained; no cohort is removed based on its batch formation.
