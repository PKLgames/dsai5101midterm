# All-demographic models: measured results and bounded interpretation

This is a retrospective, post-test exploratory supplement on the original
common strict-G2 population: 17,421 fitting responses, 3,742 validation
responses, and 3,733 saved test responses spanning 1,207 test pairs.
The experiment used all six recorded demographics, not a selected subset.
No raw test outcomes were parsed; test outcomes came only from the saved
original frozen prediction artifact. Original models, splits, claims,
report, ZIP, comprehensive evidence catalog and official predictions were
not replaced.

## Models and validation-only choices

All four models contain additive demographic displayed-A terms. Cat, YOLO and
CNN additionally contain pooled scores and additive demographic-category score
interactions. No joint demographic profiles or cross-demographic products
were searched. Cat scores pool all photos of the same cat. Y uses the original
100-dimensional encoded representation and C uses 576 CNN coordinates.

| Requested model | Stored coefficients | Selected ridge | Interaction multiplier | Validation log loss | Saved G2 test log loss | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| D: demographics only | 21 | 0.001 | Not applicable | 0.691141 | 0.688863 | 53.362% |
| Cat + all demographics | 4,998 | 0.0001 | 100 | 0.659513 | 0.658462 | 60.836% |
| YOLO-CLIP + all demographics | 2,121 | 0.0001 | 100 | 0.674380 | 0.675778 | 57.675% |
| CNN + all demographics | 12,117 | 0.0001 | 100 | 0.662324 | 0.663627 | 59.711% |

These are stored coefficients, not effective degrees of freedom. Separate
side-adjusted no-score-interaction comparators store 258, 121 and 597
coefficients for Cat, Y and C respectively. Their exact-copy fallbacks were
considered, but interactions were selected on validation in all three branches.
There were 94 recorded development candidates and no acceptance failures.

## Comparisons against original frozen models

All differences below are reference minus candidate, in nats per response;
positive favors the candidate. Intervals use 5,000 identical unordered-pair
resamples, retaining all responses and recomputing sampled-response denominators.
They are 95% pointwise exploratory intervals, not multiplicity adjusted or
independent confirmation.

| Reference minus new candidate | Difference | 95% interval |
|---|---:|---|
| Original M0 minus D | +0.003781 | [0.001330, 0.006287] |
| Original M1 minus Cat+D | +0.004750 | [0.002063, 0.007542] |
| Original M3-Y minus Y+D | +0.004095 | [0.001055, 0.007105] |
| Original M3-C minus C+D | +0.005576 | [0.002684, 0.008569] |
| Original pet-care M4-C minus C+D | +0.004669 | [0.001890, 0.007567] |
| Original pet-care side-adjusted M4-side minus C+D | +0.003856 | [0.001383, 0.006422] |

Measured result: the new models have lower saved-test point losses than these
comparators, with the stated intervals above zero.

Supported interpretation: all-demographic predictor information can improve
the tested procedures on this previously exposed G2 population under the
specified conditional bootstrap. It is not a new primary result or evidence
that those gains generalize independently.

Alternative explanations: demographic side tendencies, different composition
of encountered image pairs, regularization, correlated respondent attributes,
post-test selection motivation and unmodeled respondent/shared-image dependence
can contribute. A model improvement alone does not identify a causal
demographic preference mechanism.

## Crucial control: side tendencies versus score interactions

| New side-adjusted model minus its requested interaction model | Difference | 95% interval |
|---|---:|---|
| Cat+side minus Cat+D | +0.000485 | [-0.000001, 0.000960] |
| Y+side minus Y+D | +0.000120 | [-0.001294, 0.001527] |
| C+side minus C+D | +0.001786 | [0.000660, 0.002951] |

The Cat and YOLO interaction increments are **not resolved** by these
exploratory paired intervals. In particular, the improvement of Y+D over
original M3-Y does not prove that demographic groups prefer particular
semantic traits: much of that improvement is already present when only
demographic side effects are added. The extra CNN scoring interaction
increment is positive under the stated approximation, but CNN coordinates
are not semantic labels and the result remains noncausal and retrospective.

## Within-cat and between-cat results must remain distinct

| Model | Within-cat loss: 455 responses / 170 pairs | Between-cat loss: 3,278 responses / 1,037 pairs |
|---|---:|---:|
| D | 0.696692 | 0.687776 |
| Cat+D | 0.697547 | 0.653037 |
| Y+D | 0.692176 | 0.673502 |
| C+D | 0.703724 | 0.658061 |

Cat pooled scores and demographic cat-score deviations cancel within a cat;
only demographic displayed-A terms remain. These subgroup point estimates
have no new subgroup-specific superiority claim. CNN+demographics does not
eliminate within-cat difficulty; its within-cat point loss is higher than
the original M4-C's 0.700987, despite aggregate improvement.

## What demographics-only actually means

D predicts displayed A versus B using respondent metadata and no image
information. It cannot compare Fluffy with Whiskers, rank photos, or establish
which semantic trait a group prefers. A is not established to mean screen-left,
and randomized presentation is unverified. Call this a conditional predictive
displayed-A tendency, not proof that a group inherently clicks left.

## Support, missingness and nonidentifiability

There are 20 nonreference demographic-category columns. Category encoding
uses only fitting counts/pairs; references are man, never (pet_history),
never (pet_care), rarely, yes, and I. Missing is distinct from documented
MBTI unknown. Unsupported/unseen levels use zero deviation and are explicitly
counted in the support tables.

The six missing indicators are identical in the fitting data. The side design
with intercept has rank 16, but stores 21 coefficients. Separate missing-field
effects cannot be identified by the likelihood: ridge selects a finite
allocation across redundant columns. The full large interaction-design rank
is not computed. Do not attribute the joint missing-data association separately
to gender, history, care, contact, future plans or MBTI.

Full signed coefficients are available in `coefficients.csv` to reproduce
model output. They are affected by coding, scaling, correlated demographics
and ridge; they are not independent preference estimates or causal importance.
YOLO breed labels are CLIP-inferred categories, not pedigree annotations.

## Validation and operational checks

- Eight focused tests pass, covering encoding/support, invalid codes, orientation,
  cat-score cancellation, demographic-only behavior, finite-difference gradient,
  ridge penalty layout and exact fallback selection.
- Twenty independent fair-coin fit/validation repetitions per requested model:
  80 accepted fits, no failures, no repository investigation trigger.
- Counts of fair-coin improvements over constant 0.5: D 8/20, Cat+D 0/20,
  Y+D 1/20, C+D 0/20. Mean losses are 0.693334, 0.699831, 0.694975 and
  0.700181 respectively, versus log(2)=0.693147.
- An occasional chance improvement is not leakage evidence; the existing
  repository trigger is >=16 of 20 improvements or a failed fit.
- Saved predictions, all 14 compared model results, nine paired intervals,
  112 subgroup support/point-loss records and 40 calibration bins were checked.
- Exact validation predictions and 5,000 bootstrap replicates were independently
  replayed. The write-once saved-test analysis guard is verified.
- The original-file hashes remain unchanged.

Passing these checks is an operational assurance, not a leakage-free certificate,
causal identification or fresh independent holdout.

## Next discriminating evidence

Use a genuinely independent future sample with verified display-position
randomization, respondent linkage, adequate repeated pair/group support,
and demographic hypotheses specified before outcomes. Evaluate side-adjusted
and score-interaction models separately. Until then, describe these results
as retrospective predictive associations and keep them separate from the
original frozen G2 primary claims and deployment predictions.
