# Expanded c1 losses: saved-observation accounting

The old/corrected source heads are `91377976388a60fd69aef87ac9190555f642b30b`
and `96cedecef355cf1a4865996080ad75021da9d9b7`. This follow-up performs no new
inference. Negative throughput change means slower; wall per draft is a
request-phase diagnostic, not GPU kernel time.

| Observation | Throughput change | Draft rounds old/corrected | Extra wall ms | Per-draft cost change | Equal text |
| --- | --- | --- | --- | --- | --- |
| ablation_c1_08 | -15.036% | 51/62 | +347.927 | -0.090% | no |
| ablation_c1_15 | -14.791% | 47/57 | +318.804 | -0.118% | no |
| ablation_c1_18 | -10.229% | 51/58 | +223.753 | +0.033% | no |
| ablation_c1_04 | -5.744% | 55/59 | +127.526 | -0.071% | no |
| ablation_c1_20 | -5.731% | 55/59 | +127.216 | -0.033% | no |
| ablation_c1_10 | -5.669% | 57/61 | +129.521 | -0.002% | no |
| ablation_c1_19 | -4.552% | 52/55 | +95.188 | -0.002% | no |
| ablation_c1_26 | -4.394% | 53/56 | +93.219 | -0.034% | no |
| ablation_c1_00 | -3.242% | 49/51 | +63.713 | -0.132% | no |
| ablation_c1_13 | -3.107% | 50/52 | +61.998 | -0.079% | no |
| ablation_c1_14 | -2.998% | 48/50 | +57.815 | -0.159% | no |
| ablation_c1_25 | -1.450% | 52/53 | +29.384 | -0.073% | no |
| ablation_c1_02 | -1.401% | 56/57 | +30.207 | -0.109% | no |
| ablation_c1_23 | -1.343% | 48/49 | +25.519 | -0.347% | no |
| ablation_c1_07 | -1.306% | 48/49 | +24.782 | -0.258% | no |
| ablation_c1_01 | -1.188% | 56/57 | +25.660 | -0.217% | no |
| ablation_c1_16 | -0.038% | 48/48 | +0.705 | +0.042% | yes |

Sixteen slower observations add 57 draft rounds. Their total wall difference
is +1.78223s; decode is +1.78710s. For decode `D = N × C`, the symmetric exact
decomposition is `ΔD = ΔN × mean(C) + ΔC × mean(N)`: extra work contributes
+1.81495s and per-round cost −0.02785s. Prefill and other endpoint costs do
not account for the measured losses. All 30 observations have zero
prefix-cache hits, cached prompt tokens, preemptions and recorded external
traffic contamination. The equal-work 0.705ms case is not enough to establish
repeatable performance loss.

Counts by proposal position for the 16 extra-work observations (old/corrected)
are 527/530, 334/311, 210/193, 143/134, over 828/885 rounds. Average accepted
proposals per round decrease from 1.46618 to 1.31977. These counters precede
completion-budget clipping; their sum plus rounds need not equal the reported
128 output tokens exactly. Do not call all accepted-counter differences
changes in delivered token counts.

Four c1 input pairs repeat: 04/20, 05/21, 06/22, 07/23. Their outputs and round
counts match within each arm. There are 26 distinct inputs, not 30. With
duplicate wall times averaged per input, 14 cases add draft work, one is the
equal-work timing case, and 11 gain throughput. Natural generations differ
on 28/30 observations. The first different text character is not necessarily
the first different token or logit; raw token/logit traces were not captured.

The saved lowered-source audit identifies mismatched valid autotune choices,
including target and drafter normalization reductions. This is a concrete
confound, not proof that a specific choice caused a specific sample to slow.
Same-configuration model controls and same-live-input projection replay remain
unperformed. No production policy or code change is justified by a cache-file
comparison alone.
