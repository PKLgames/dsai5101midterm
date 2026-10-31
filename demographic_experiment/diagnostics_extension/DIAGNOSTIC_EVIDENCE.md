# All-demographic diagnostic parity: retrospective evidence

All numbers are exploratory after original G2/G3/G4 outcomes were exposed. Do not interpret them as independently validated, causal, or a new primary-model/deployment choice.

## Existing controls, calibration, and within/between-cat slices

The original demographic supplement already contains 20 fair-coin controls per requested model (80 accepted fits; no investigation trigger), four-model overall calibration, and comparison-type subgroup metrics. The control recorded improvements over log(2): D 8/20, Cat+D 0/20, Y+D 1/20, C+D 0/20.
See [control results](../control_results.csv), [control provenance](../control_provenance.json), [existing calibration](../calibration.csv), and [existing subgroups](../subgroups.csv). Control passage does not certify leakage-free software or establish independent generalization.

New [validation support metrics](validation_support_slices.csv) and [validation calibration](validation_support_calibration.csv) use original M4-C support/quality group definitions; [validation demographic slices](validation_demographic_slices.csv) and [validation demographic calibration](validation_demographic_calibration.csv) cover all six demographics. The new [test support slices](test_support_slices.csv), [test support calibration](test_support_calibration.csv), and [test demographic calibration](subgroup_calibration.csv) use saved G2 predictions. Overall slice support is 100 responses/30 pairs; bin support is 30 responses/10 pairs. Quality flags overlap; do not sum overlapping slices. Low-support estimates/bins remain flagged.

## Nested-pair learning curves

Two original seeds and four nested fractions; demographic support, references, category effects, Y/C preprocessing and registered penalty grids are reselected at each size. All models use the same 1,836-response/589-pair strict-validation intersection. These development curves were motivated after the test was seen and are not an information-limit estimate. See [curves](learning_curves.csv), [fitting category support](learning_curve_support.csv), and individual curve attempt logs/models in this directory. The two 100% seeds have exactly the same ordered fitting and validation rows: one full-size fit was reused after identity checks, not treated as two independent fits. Partial logs from resource-management interruptions remain separately labeled.

| Seed | Fraction | Fit responses / pairs / photos | D | Cat+D | Y+D | C+D |
|---|---:|---|---:|---:|---:|---:|
| 20261030 | 0.25 | 4367 / 1408 / 1161 | 0.691447 | 0.681836 | 0.679983 | 0.676050 |
| 20261030 | 0.50 | 8737 / 2817 / 1288 | 0.691026 | 0.670841 | 0.679025 | 0.670976 |
| 20261030 | 0.75 | 13057 / 4225 / 1296 | 0.691223 | 0.670431 | 0.678217 | 0.670190 |
| 20261030 | 1.00 | 17421 / 5634 / 1296 | 0.691033 | 0.670781 | 0.676639 | 0.666076 |
| 20261031 | 0.25 | 4299 / 1408 / 1140 | 0.690720 | 0.684690 | 0.679684 | 0.674902 |
| 20261031 | 0.50 | 8636 / 2817 / 1281 | 0.690581 | 0.676236 | 0.677460 | 0.672189 |
| 20261031 | 0.75 | 13007 / 4225 / 1295 | 0.690903 | 0.673011 | 0.677563 | 0.670768 |
| 20261031 | 1.00 | 17421 / 5634 / 1296 | 0.691033 | 0.670781 | 0.676639 | 0.666076 |

Exact no-score-interaction fallbacks were selected in **6 of 24** Cat/Y/C curve cells; see the curve table for the affected seed and size.

## Reproduced penalty paths and temperature

The [penalty-path refit table](penalty_path_refits.csv) reproduces the original 94 fitting/validation candidate records (91 numerical refits and three exact fallback copies) and includes non-intercept coefficient norms. A path diagnoses shrinkage and fit/validation patterns, not causation.

| Model | Validation-selected temperature | Unscaled minus scaled G2 loss | 95% pair interval |
|---|---:|---:|---|
| D | 1.5 | -0.000586 | [-0.001487, 0.000342] |
| Cat+D | 1 | +0.000000 | [0.000000, 0.000000] |
| Y+D | 1 | +0.000000 | [0.000000, 0.000000] |
| C+D | 1 | +0.000000 | [0.000000, 0.000000] |

Positive means temperature scaling reduces log loss. Temperature candidates were chosen on validation and sealed before this saved-test reanalysis, but this experiment was proposed after the test had already been seen. T=1 implies exact identity, not a new improvement.

## Separately refitted retrospective G3/G4 transfer

These models were refit and retuned on the existing G3/G4 fitting and validation populations. Existing test outcomes were previously exposed; no fresh independent test is claimed. G3 uncertainty resamples held-out photos; G4 resamples held-out cats. G4 Cat+D assigns the unseen cat zero for both pooled and all demographic-specific cat-score components. This is a fixed cold-start convention, not a learned unseen-cat preference.

| Target | Model | Validation-selected pooled fallback | Test response loss | 95% cluster interval |
|---|---|---|---:|---|
| G3 | D | False | 0.689100 | [0.686252, 0.691784] |
| G3 | Cat+D | False | 0.657926 | [0.646415, 0.669205] |
| G3 | Y+D | False | 0.678040 | [0.671594, 0.684268] |
| G3 | C+D | False | 0.666009 | [0.654564, 0.676922] |
| G4 | D | False | 0.688684 | [0.685652, 0.691689] |
| G4 | Cat+D | False | 0.676922 | [0.670194, 0.683216] |
| G4 | Y+D | True | 0.672731 | [0.663856, 0.680670] |
| G4 | C+D | False | 0.665266 | [0.657928, 0.672747] |

Interpret score-interaction gains only using the side-adjusted comparisons in [G3 contrasts](g3_contrasts.csv) and [G4 contrasts](g4_contrasts.csv). A gain versus the original base model can combine side effects, score effects and refitting. G3/G4 output/model/attempt files remain isolated here; original records are unchanged.

## Abstention and utility

The [105-cell table](abstention_utility.csv) uses four demographic models and original M4-C on the same saved G2 outcomes, thresholds 0.50–1.00 by 0.025, correct +1, incorrect -1, abstain 0, and all original responses as the denominator. Confidence ties remain together; no test-optimal cutoff or deployment decision is selected.

| Model | Cutoff | Retained | Coverage | Selective accuracy | Utility / original response |
|---|---:|---:|---:|---:|---:|
| D | 0.50 | 3733 | 100.00% | 53.36% | 0.067238 |
| D | 0.60 | 0 | 0.00% | undefined | 0.000000 |
| D | 0.70 | 0 | 0.00% | undefined | 0.000000 |
| Cat+D | 0.50 | 3733 | 100.00% | 60.84% | 0.216716 |
| Cat+D | 0.60 | 1715 | 45.94% | 67.11% | 0.157246 |
| Cat+D | 0.70 | 729 | 19.53% | 74.76% | 0.096705 |
| Y+D | 0.50 | 3733 | 100.00% | 57.67% | 0.153496 |
| Y+D | 0.60 | 1159 | 31.05% | 63.85% | 0.085990 |
| Y+D | 0.70 | 129 | 3.46% | 77.52% | 0.019020 |
| C+D | 0.50 | 3733 | 100.00% | 59.71% | 0.194214 |
| C+D | 0.60 | 1701 | 45.57% | 66.26% | 0.148138 |
| C+D | 0.70 | 526 | 14.09% | 73.00% | 0.064827 |
| M4-C | 0.50 | 3733 | 100.00% | 58.85% | 0.177069 |
| M4-C | 0.60 | 1575 | 42.19% | 65.52% | 0.130994 |
| M4-C | 0.70 | 437 | 11.71% | 70.25% | 0.047415 |

Higher accuracy among retained cases does not necessarily mean greater utility across all original cases. Intervals are pointwise 95%, pair-clustered and unadjusted; low-support/empty subsets are explicitly flagged. See the [four-page abstention dashboard](presentation_abstention_full.pdf), [fixed diagnostics figure](presentation_fixed_diagnostics.pdf) and [learning/penalty/transfer figure](presentation_learning_penalty_transfer.pdf).

## Boundaries

Displayed A is not established to mean left; demographics-only cannot rank cats/photos. All six missing demographic columns coincide in the original fitting data, so individual missing-field effects are not separately identifiable. CLIP categories are automated measurements, not verified pedigree. No causal demographic, context or feature effect follows.

Diagnostic summary SHA-256: `c00930e97a5bb75066db65a4421f0f38924bf4fd49a15c3ba3358a7f5144aabe`; verification SHA-256: `8fb91857a60be101ac7cec810cf9dd331ffddc2bc5b5c667d8d98d6df81bc759`.
