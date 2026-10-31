# Retrospective Cat + CNN additive scorer: evidence

This is a post-test, noncausal, nonconfirmatory pooled cat-identity + CNN procedure. The original G2/G3 outcomes were previously exposed; the original frozen model and official deployment are unchanged.

## Model and matching

The displayed-A logit is beta + cat(A)-cat(B) + w·(CNN(A)-CNN(B)). Separate positive ridge penalties for cat and CNN blocks were chosen from 49 fixed candidates on each split's validation rows. The original G2 and separate G3 fitting-only CNN preprocessing was refitted on unique fitting photos. Cat and CNN terms can represent overlapping information; penalty choices determine their allocation, so coefficient magnitudes are not a causal or behavioral decomposition. The frozen protocol's blanket claim of additional cat/CNN nonidentifiability is overstated: an outcome-blind G2 design check found rank 813 of 814, consistent with only the usual cat-offset gauge. See [rank check and protocol erratum](design_rank_and_protocol_erratum.json).

| Target | Responses / pairs | Resampling clusters | Cat + CNN loss | Brier | Accuracy |
|---|---|---|---:|---:|---:|
| G2 | 3733 / 1207 | 1207 pair_id | 0.661909 | 0.234835 | 59.58% |
| G3 | 2250 / 718 | 64 held_out_photo | 0.660637 | 0.234279 | 58.84% |

**G4: unsupported by explicit user choice.** No unseen-cat identity score was assigned, and no G4 predictions or outcomes were used. The CNN block could score the photo, but that is not the specified joint model. See [G4 status](g4_unsupported.json).

## Matched paired differences

Reference minus Cat+CNN: positive favors the hybrid. 95% pointwise intervals are unadjusted and conditional on fitted models/selection. G2 resamples unordered pairs; G3 resamples held-out photos.

| Target | Reference | Difference | 95% interval |
|---|---|---:|---|
| G2 | original:common:M1 | +0.001304 | [-0.002937, 0.005614] |
| G2 | original:M3-C | +0.007294 | [0.002735, 0.011613] |
| G2 | original:M4-C | +0.006387 | [0.001743, 0.010782] |
| G3 | original:M1 | +0.002088 | [-0.004448, 0.007905] |
| G3 | original:M3-C | +0.011531 | [0.004756, 0.017998] |
| G3 | original:M4-C | +0.009328 | [0.001680, 0.016668] |

The hybrid improves point loss over CNN-only on both retrospective splits, with the displayed exploratory intervals above zero. The incremental comparison against cat-only M1 is unresolved on both splits. Thus it is not established that CNN adds predictive value once cat identity is available. Comparison against contextual M4-C changes both context and model inputs. The separate [kernel/demographic comparisons](additional_paired_contrasts.csv) also use different inputs/tuning.

## Controls, reliability and comparison type

The fixed selected procedure passed 20 fair-coin fitting/validation repetitions: zero failures and 0/20 improvements over log(2). This is an operational control, not a leakage-free certificate. [Validation](validation_support_slices.csv) and [test](test_support_slices.csv) support/quality slices use original M3-C definitions; ten-bin reliability is in the corresponding support-calibration tables. Low support is flagged, and quality groups overlap.

| Target / type | Responses / pairs | Cat+CNN log loss | Accuracy |
|---|---|---:|---:|
| G2 within_cat | 455 / 170 | 0.691254 | 53.85% |
| G2 between_cats | 3278 / 1037 | 0.657835 | 60.37% |
| G3 within_cat | 286 / 106 | 0.696858 | 45.80% |
| G3 between_cats | 1964 / 612 | 0.655363 | 60.74% |

Within-cat, the cat identity score difference is exactly zero; only the CNN difference and intercept contribute. [G2 subgroup contrasts](g2_subgroup_contrasts.csv) and [G3 within-cat contrasts](g3_within_contrasts.csv) are post-test, unadjusted subgroup analyses. They do not support claiming a general causal photo effect.

## Learning curves, penalty path and temperature

Every nested subset refits unique-fitting-photo CNN preprocessing, cat keys and all 49 penalty pairs; common validation support is 1,836 responses / 589 pairs. The 100% seeds have identical fitting/validation populations and are not independent replicates. The [penalty replay](penalty_refits.csv) independently reconstructs all 49 candidates. Temperature selected on validation was T=1, so scaled and unscaled outputs coincide.

| Seed | Fraction | Fit responses / pairs / photos | Cat ridge | CNN ridge | Validation loss |
|---|---:|---|---:|---:|---:|
| 20261030 | 0.25 | 4367 / 1408 / 1161 | 0.01 | 0.001 | 0.673847 |
| 20261030 | 0.50 | 8737 / 2817 / 1288 | 0.001 | 0.001 | 0.665664 |
| 20261030 | 0.75 | 13057 / 4225 / 1296 | 0.001 | 0.001 | 0.664011 |
| 20261030 | 1.00 | 17421 / 5634 / 1296 | 0.001 | 0.0001 | 0.662325 |
| 20261031 | 0.25 | 4299 / 1408 / 1140 | 0.01 | 0.001 | 0.675481 |
| 20261031 | 0.50 | 8636 / 2817 / 1281 | 0.01 | 0.001 | 0.672411 |
| 20261031 | 0.75 | 13007 / 4225 / 1295 | 0.001 | 0.001 | 0.667245 |
| 20261031 | 1.00 | 17421 / 5634 / 1296 | 0.001 | 0.0001 | 0.662325 |

## Fixed-grid abstention

Correct +1, incorrect -1, abstain 0; utility denominator is all 3,733 original responses. No cutoff is selected from the test.

| Confidence threshold | Retained responses / pairs | Coverage | Retained accuracy | Utility / original response |
|---:|---|---:|---:|---:|
| 0.50 | 3733 / 1207 | 100.00% | 59.58% | 0.191535 |
| 0.60 | 1392 / 521 | 37.29% | 68.97% | 0.141441 |
| 0.70 | 373 / 145 | 9.99% | 76.94% | 0.053844 |

Higher retained accuracy does not establish improved total utility. See [fixed-grid results](g2_abstention_utility.csv) and [abstention chart](presentation_abstention.pdf).

## Verification and limitations

Four focused tests passed. The [verification record](verification.json) independently reconstructs 10 selected G2/G3/curve fits, validates fitting-only preprocessing and gradients, replays 5,000 cluster-bootstrap draws per tested target, and checks 84 abstention cells. Prior outcome exposure, post-test motivation, model-selection uncertainty, shared images/cats/respondents and conditional 95% pointwise unadjusted intervals prevent independent confirmation or causal claims.
