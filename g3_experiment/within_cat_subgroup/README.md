# Retrospective G3 within-cat subgroup

This post-test descriptive analysis filters the saved frozen G3 prediction
artifact to comparisons between two photos of the same cat. It does not parse
raw outcomes, rerun the G3 test evaluation, refit models, or select a model
using subgroup results. The G3 test itself was retrospective, overlaps earlier
outcome access, and is not an independent confirmatory test.

The analysis reports response-weighted log loss, Brier score, accuracy and
paired log-loss differences. Its 95% pointwise intervals use 5,000 shared
resamples of held-out photos represented in the subgroup, with the response
denominator recomputed for each resample. Intervals are not multiplicity
adjusted and do not account for all potential dependence among known photos,
cats or respondents.

Within a cat, the M1 cat-score difference cancels; M1's prediction is its
displayed-side intercept and does not distinguish the two photos. The M1/M3
comparison is not evidence that cat identity itself separates those photos.
None of the visual or contextual comparisons identifies a causal effect.

Run the focused checks from the project root using the existing environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe g3_experiment\within_cat_subgroup\tests.py
.\.venv\Scripts\python.exe g3_experiment\within_cat_subgroup\verify.py
```

`analyze.py` and `publish.py` refuse to overwrite existing outputs. Hash and
provenance records document the preserved G2/G3 artifacts and copy-only
updates to the presentation evidence.
