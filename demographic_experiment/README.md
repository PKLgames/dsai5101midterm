# All-demographic retrospective predictive experiment

Four requested models use all six recorded categorical demographics:
gender, pet_history, pet_care, pet_contact, future_pet, and mbti_ie.

- D: additive demographic displayed-A tendencies only.
- Cat+D: pooled cat scores plus demographic side effects and additive
  demographic-category deviations in cat scores.
- Y+D: original YOLO-CLIP numeric/categorical representation, demographic side
  effects and additive category-by-feature-difference interactions.
- C+D: all 576 frozen CNN coordinates with the same demographic design.

Each Cat/Y/C branch has a validation-tuned side-adjusted no-score-interaction
comparison and an exact-copy fallback to it. No joint demographic profile or
cross-demographic interaction products are searched. Missing values are separate;
MBTI unknown is preserved. Category eligibility/reference uses fitting data only.
Unsupported/unseen categories use explicitly reported zero deviations.

The original common G2 fit/validation/test population remains unchanged.
Preprocessing is refitted on unique fitting photos and checked against frozen
specifications. Penalties use the original grid and strong interaction multipliers;
all attempts and numerical acceptance are recorded. Models are frozen before
this experiment reads already-saved test outcomes. Raw test labels are not parsed.

This is post-test, retrospective exploration, not an independent confirmatory
study. D cannot rank cats/photos or establish which traits a demographic prefers.
Displayed A is not established to mean screen-left or randomized position.
Cat/semantic/CNN interactions are model-based conditional predictive associations,
not causal preferences, valid pedigree or individual effects. Correlated
demographics and limited support can make coefficients unstable. Pair bootstrap
intervals are pointwise, unadjusted for multiplicity and omit some dependence.

All original G2/G3/G4 and official deployment artifacts are preserved.
Presentation evidence is copied, not moved, with separate historical manifests.
The existing comprehensive evidence catalog is a prior snapshot and is not
silently rewritten to claim these new results were collected earlier.

From the project root with the existing CPU environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe demographic_experiment\tests.py
.\.venv\Scripts\python.exe demographic_experiment\experiment.py develop
.\.venv\Scripts\python.exe demographic_experiment\control.py
.\.venv\Scripts\python.exe demographic_experiment\experiment.py evaluate
.\.venv\Scripts\python.exe demographic_experiment\verify.py
.\.venv\Scripts\python.exe demographic_experiment\experiment.py publish
```

Development/evaluation/publication refuse to overwrite existing records.
Do not remove the test-access guard to rerun or retune on test outcomes.
Before saved-test analysis, `control.py` runs 20 independent fair-coin
fit/validation label repetitions for each requested procedure at its fixed
selected settings. It uses the repository's existing investigation trigger
(at least 16 of 20 improvements or any failed fit). This is an operational
control, not a leakage-free certificate. Its separate provenance records
the selected-model freeze and control source/results hashes.

`design_conditioning.json` records an additional important limitation:
the six missing-demographic indicator columns coincide in the fitting data.
The side design has rank 16 but stores 21 coefficients including the intercept.
Ridge chooses a finite allocation across redundant columns; separate missing
gender/history/care/contact/future/MBTI effects cannot be distinguished by
the likelihood. This is not six separately measured missing-category effects.
The full interaction-design rank is not computed, and stored coefficients
must not be interpreted as effective degrees of freedom.
