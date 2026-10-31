# Retrospective nonlinear CNN kernel scorer: evidence

Pooled photo-level RBF Nyström M3-C extension. This is post-test exploratory and noncausal; prior G2/G3/G4 outcomes were exposed. It does not replace the original primary model or deployment.

## Model and fitting

The preference logit is a displayed-A intercept plus the difference between two nonlinear photo scores. Up to 256 landmarks are chosen by fixed hash from unique fitting photos only. CNN centering/block-RMS, bandwidth median, landmark map and Nyström center/RMS are estimated on fitting photos only, including each learning-curve and transfer split. Three width multipliers and seven ridge values are tuned on the matching validation set; all candidates and solver acceptance are retained.

| Target | Responses / pairs | Held-out unit | Kernel log loss | Brier | Accuracy |
|---|---|---|---:|---:|---:|
| G2 | 3733 / 1207 | 1207 pair_id | 0.669272 | 0.238322 | 58.40% |
| G3 | 2250 / 718 | 64 held_out_photo | 0.672923 | 0.240023 | 58.62% |
| G4 | 3479 / 1101 | 23 held_out_cat | 0.673013 | 0.240108 | 57.20% |

## Matched paired differences

Reference minus kernel: positive favors the kernel. All intervals are 95% pointwise, unadjusted and conditional on selected fits; G2 resamples unordered pairs, G3 held-out photos and G4 held-out cats.

| Target | Original reference | Difference | 95% interval |
|---|---|---:|---|
| G2 | original:M3-C | -0.000070 | [-0.003638, 0.003602] |
| G2 | original:M4-C | -0.000976 | [-0.004621, 0.002685] |
| G3 | original:M3-C | -0.000754 | [-0.006277, 0.004684] |
| G3 | original:M4-C | -0.002957 | [-0.008812, 0.003082] |
| G4 | original:M3-C | -0.001204 | [-0.004555, 0.001846] |
| G4 | original:M4-C | -0.002974 | [-0.006463, 0.000211] |

The linear-versus-kernel comparisons above span zero; these data do not establish an advantage from this kernel. M4-C is contextual and the kernel scorer is pooled, so that comparison changes both model class and context; M3-C is the most direct like-for-like reference.

The separately saved [paired demographic-CNN comparisons](additional_paired_contrasts.csv) use the same rows/clusters but C+D also uses demographics. They cannot isolate nonlinearity.

## Controls, calibration and slices

The fixed selected G2 kernel settings passed 20 fair-coin fits: 0 failures, 0/20 improvements over log(2). This is an operational check, not a leakage-free certificate. Original-style development and test support/quality slices and reliability bins are in [validation slices](validation_slices.csv), [validation bins](validation_calibration.csv), [G2 test slices](g2_slices.csv) and [G2 bins](g2_slice_calibration.csv). Slice support is 100 responses/30 pairs and bin support 30 responses/10 pairs; low support is flagged.

G2 within-cat loss is **0.695371** on 455 responses/170 pairs; between-cat loss is **0.665649** on 3,278 responses/1,037 pairs. G3 within-cat loss is **0.706694** on 286 responses/106 pairs; its between-cat loss is **0.668005** on 1,964 responses/612 pairs. G4's approved target has between-cat comparisons only. These are point estimates, not paired subgroup improvement claims.

The matching [G3 within-cat results](g3_within_results.csv) and
[paired contrasts](g3_within_contrasts.csv) use 286 responses, 106 pairs
and 60 held-out photos. Original M3-C minus kernel is **-0.006808** with
95% pointwise interval **[-0.019018, 0.005108]**; direction is unresolved.
Original contextual M4-C minus pooled kernel is **-0.013030**, interval
**[-0.025989, -0.000467]**. This secondary, unadjusted comparison also
changes the contextual specification; it is not evidence that nonlinearity
alone worsens same-cat prediction.

## Learning curves, penalty path and temperature

The [learning curves](learning_curves.csv) use the original 1,836-response/589-pair validation intersection and refit all preprocessing, landmark selection, width and ridge at each nested subset. The [G2 candidate log](g2_attempts.csv) has all 21 accepted width/ridge combinations. The 100% seeds have the same fitting/validation data and are not independent replications.

The separate [penalty refits](penalty_refits.csv) independently reproduced
all 21 G2 width/ridge candidates from zero initialization, with maximum
validation-loss difference **0.0**. [Penalty provenance](penalty_provenance.json)
and [additional verification](additional_verification.json) record those checks.

| Seed | Fraction | Fit responses / pairs / photos | Width multiplier | Ridge | Validation loss |
|---|---:|---|---:|---:|---:|
| 20261030 | 0.25 | 4367 / 1408 / 1161 | 1 | 0.001 | 0.674606 |
| 20261030 | 0.50 | 8737 / 2817 / 1288 | 1 | 0.001 | 0.671815 |
| 20261030 | 0.75 | 13057 / 4225 / 1296 | 0.25 | 0.0001 | 0.669267 |
| 20261030 | 1.00 | 17421 / 5634 / 1296 | 0.25 | 0.0001 | 0.665630 |
| 20261031 | 0.25 | 4299 / 1408 / 1140 | 0.25 | 0.001 | 0.677476 |
| 20261031 | 0.50 | 8636 / 2817 / 1281 | 1 | 0.001 | 0.672431 |
| 20261031 | 0.75 | 13007 / 4225 / 1295 | 0.25 | 0.0001 | 0.669820 |
| 20261031 | 1.00 | 17421 / 5634 / 1296 | 0.25 | 0.0001 | 0.665630 |

Validation selected temperature **T=1**, so the temperature-scaled scorer equals the unscaled scorer exactly; this is not a new improvement.

## Abstention and utility

Fixed confidence thresholds 0.50–1.00 in 0.025 steps; +1 correct, -1 incorrect, 0 abstain; denominator is all 3,733 G2 responses. No test threshold is selected.

| Kernel cutoff | Retained responses / pairs | Coverage | Retained accuracy | Utility / original response |
|---:|---|---:|---:|---:|
| 0.50 | 3733 / 1207 | 100.00% | 58.40% | 0.167961 |
| 0.60 | 1583 / 597 | 42.41% | 65.07% | 0.127779 |
| 0.70 | 372 / 141 | 9.97% | 73.39% | 0.046611 |

Higher retained accuracy does not establish improved total utility. The fixed 0.60 cutoff reduces utility from 0.167961 to 0.127779 despite increasing selected accuracy.

The [selective-accuracy and cutoff figure](presentation_kernel_abstention.pdf)
and [utility table](g2_abstention_utility.csv) report support and uncertainty;
the original M3-C and M4-C rows are matched comparisons, not new independent tests.

## Verification and limitations

The [verification record](verification.json) independently reconstructs all 11 selected G2/G3/G4/curve fits, checks fitting-only landmarks and transforms, validates gradients and candidate selection, and replays 5,000 cluster-bootstrap draws per target and 63 abstention cells. Four focused kernel tests passed. The original model and all G2/G3/G4 outcomes were previously exposed; these post-hoc intervals are not primary 98.75% confirmatory contrasts.

Changing both nonlinear representation and width/penalty search makes this a procedure comparison, not an isolated causal effect of nonlinearity. Repeated photos/cats/respondents and training/model selection uncertainty are not fully represented by the cluster bootstrap.
