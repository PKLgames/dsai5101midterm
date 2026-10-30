# Stage 0 protocol: predictive diagnosis of cat-photo preferences

Status: candidate registry recorded; prior outcome-access history remains
unresolved. Do not start real-data modeling until the team has documented how
the unknown history will be handled. Stage 0 is not signed off.

The supplied [project plan](project-plan-source.txt) is the authoritative
execution contract. Its SHA-256 is
`85ecf8eb38e8a9bf5574d3de388cd887d00441fa6732e475f5939ec5243ea823`.
This protocol records Stage 0 decisions; it does not replace the plan.
[decisions.yaml](decisions.yaml) is the machine-readable registry,
[claims.csv](claims.csv) registers planned questions, and
[contributions.csv](contributions.csv) records assignments, not completed work.
Record changes and their reasons before acting on them.

## Research scope

- Primary question: which representations improve prediction of unseen
  matchups between familiar cat photographs, and what do their gains, failures,
  and remaining weaknesses reveal about the preference models?
- Primary target: strict G2, unseen unordered photo pairs whose endpoints are
  fitting photos in the same fitting-photo component. The primary metric is
  response-weighted natural-log loss in nats per response.
- T2 and T3 organize the project; T1 and T4 support it. Permitted activity
  labels are Predict, Explain, Show evidence, Clarify, and Visualize.
  Evidence-status labels are Real-data claim, Known-truth demonstration, and
  Methodological rule. A planned real-data claim is not a finding.
- For each substantive conclusion distinguish measured result, supported
  interpretation, alternative explanations, unestablished conclusion, and
  next discriminating evidence.
- Do not claim objective attractiveness, causal visual or context effects,
  percentages of disagreement attributable to behavioral causes, a numerical
  information limit, the dominant source of remaining uncertainty, individual
  consistency, or respondent-disjoint generalization.

## Candidate registry

The following are registered before model comparison. Admission of each feature
pipeline is conditional on the plan's quality and coverage gates; registration
does not imply successful integration.

| Model | Registered role |
| --- | --- |
| M0 | Smoothed constant displayed-A baseline |
| M1 | Cat identity and shared displayed-A intercept |
| M2 | Photo identity and shared displayed-A intercept |
| M3-H, M3-C, M3-Y | Shared handcrafted, frozen CNN, and YOLO-CLIP scorers |
| M4-H, M4-C, M4-Y | Matched contextual extensions using `pet_care` |
| M2-no-position | Secondary photo-identity position ablation |
| M3-side, M4-side | Secondary context-dependent side diagnostics for the selected representation |

Select M3* from admitted M3 models using common strict-validation loss, with
the plan's H, C, Y tie order. Select M4* only within the same representation,
including the exact pooled M3* fallback. A selected pooled fallback has zero
context contrast by construction. The four registered primary contrasts are
M0 minus M1, M1 minus M2, M2 minus M3*, and M3* minus M4*, on the common
visual strict-G2 test set. Preserve the plan's full-identity and
representation-specific fallback analyses if common visual coverage fails.

Enabled secondary options: development-only CNN PCA, temperature scaling for
M2, M3*, and M4*, and a descriptive in-sample pair-entropy summary. PCA is not
a primary candidate; entropy is not an information-limit estimate. The
temperature grid and the plan's validation-only selection and pre-test freeze
apply separately to the registered models. Official predictions are enabled
only after the internal analysis is finished and frozen, following the
deployment and fallback rules in the plan.

G3, all-block fusion, the photo-residual hybrid, and grouped permutation are
not enabled. Fine-tuning and real-data cycle testing are outside the core.
Test-driven repair or selection is not confirmatory. Amendments require an
explicit decision record and disclosure; do not silently promote a diagnostic.

## Splits, selection, and safeguards

Retain an existing valid split if one is discovered; do not replace it merely
to match a seed. The user reports no existing split, and no notebook or
split-named file was found in the current non-dataset working tree. This is
not evidence that no split exists in Git history or elsewhere.

If a new split is needed, use seed 20261028 and approximately 70/15/15
fitting/validation/internal-test allocation by unordered pair. Stratify by
comparison type and response bands 1, 2-4, 5-9, and 10+; collapse the bands
within a comparison type if any has fewer than 20 pairs. Allocate by largest
remainder, breaking ties in fitting, validation, test order. All rows for a
pair stay together. Strict G2 eligibility is structural and uses fitting
photos and components, not held-out outcomes.

Fit the common visual comparison on the same eligible fitting responses and
evaluate on identical eligible rows, as specified in sections 13-14 of the
plan. Select penalties on validation, save all attempts and failures, and
freeze before accessing internal-test outcomes. Maintain separate full-data
identity models. Never silently drop missing predictions or overwrite raw data.

Prior analysis is documented in [proj-docs.md](proj-docs.md), so a fresh folder
does not imply outcome blindness. The user reports no available official test
labels, but prior team access to training outcomes and internal holdout results
is unknown. See [the access log](artifacts/outcome_access_log.csv). The team
must resolve or explicitly document the residual uncertainty and its effect on
confirmatory interpretation before real-data modeling. Do not imply that
internal test outcomes were previously unseen.

## Stage 0 inspection

The root `notebooks`, `src`, and `tests` directories contained no files at
inspection. The root README describes notebooks that are not present, and
`proj-docs.md` describes an earlier analysis rather than results from this
protocol. No existing root `.venv` was found in the earlier readiness check;
the root requirements describe a different legacy environment. These are not
validated run instructions for this plan.

Retained source under `legacy` is reference material, not an admitted pipeline
or an implementation of M4. The handcrafted source defines 20 features; the
backbone source specifies a 576-dimensional MobileNetV3-Small representation;
the photo-labeling source contains YOLO-CLIP definitions. The legacy extractor
uses Unix-specific `resource` and the `fork` context, and its paths point
under `legacy` rather than the root dataset. Its CNN preprocessing uses
short-side 232 at size 224, so the actual preprocessing must be documented and
audited rather than assumed to match another recipe. The legacy feature loader
can standardize over a complete cache, and the label encoder derives levels
from its supplied table, drops duplicate keys, and replaces missing numeric
differences with zero. These behaviors are not accepted as satisfying the
plan's fitting-only preprocessing and explicit failure rules.

## Feature-integration manifest requirements

Before admission, create a separate manifest for H, C, and Y. Each cache must
have exactly one record per image ID, with ordered feature values, extraction
status and failure reason, pipeline version, and source-image hash. Record:

1. Source commit or code hash; ordered feature names, formulas, transforms,
   units, dimensions and dtype; source-image hashes.
2. EXIF handling, color conversion, resize/crop/interpolation, architecture,
   checkpoint and weight hash, libraries, device, precision, randomness and
   determinism, missing-image and extraction-failure behavior.
3. For Y, detection checkpoint and thresholds, class mapping, selection,
   mask and coordinate rules, no-detection behavior, exact CLIP prompts and
   labels, aggregation, image views, normalization, selection/uncertainty
   rules, HSV ranges, and pattern rules.

H must audit the actual 20 implemented definitions rather than inventing
them. C must verify the intended 576-dimensional pre-classifier layer,
normalization, evaluation/inference mode, checkpoint, and preprocessing.
Y is an automated measurement, not an annotation or true pedigree. Preference
outcomes must not enter feature extraction. Fit imputation, scaling, category
pooling, PCA, feature selection, and learned fusion on unique fitting photos
only, including within learning-curve subsets.

Admission requires recoverable provenance, passing key tests, explicit
failures, at least 95% strict-validation response coverage with usable
endpoints, and no unresolved systematic extraction bug. Poor predictive loss
alone is not a reason to reject a pipeline. No-detection, unknown attribute,
failed extraction, and missing numeric values must remain distinct.

## Remaining Stage 0 gate

The team must review and sign off this registry, assignments, and access log;
resolve or explicitly record the limits of unknown historical outcome access;
and document any discovered prior split before real-data modeling. This gate
does not assert that Stage 1's data audit, feature extraction, tests, or any
subsequent stage has run.
