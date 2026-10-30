# Retrospective G4: one unseen cat versus one known cat

This experiment is quarantined and exploratory. G4 was not reserved before G2
or G3 outcome access, and its fitting and test sets overlap previously exposed
outcomes. Its test is therefore not independent or confirmatory. Nothing here
changes the frozen G2/G3 models, splits, claims, reports, or code archive.

The user selected comparisons with exactly one unseen cat and one known cat.
`protocol.json` records the fixed duplicate-aware cat reservation, separate
fitting split, coverage gate and validation-only selection rules. Both test
and validation passed the 200-response, 50-pair, 20-held-out-cat operational
gate. The frozen G4 test results use 5,000 shared held-out-cat bootstrap draws.
The 95% pointwise intervals are unadjusted for multiplicity and do not account
for all shared known-cat, photo or respondent dependence.

M1-zero is a cold-start comparison: the unseen cat is assigned a fixed score
of zero while the known fitting-cat score is retained. The unseen score is not
estimated, and zero is not evidence that the unseen cat is average. M2 is not
used because an unseen cat's photos lack fitted free photo scores. Visual and
contextual model comparisons are predictive, not causal.

From the project root using the existing environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe g4_experiment\tests.py
.\.venv\Scripts\python.exe g4_experiment\verify_saved.py
```

Reservation, development, test evaluation and evidence publication are
write-once. `evaluation_started.json` records test access before outcomes are
loaded; do not remove it to rerun evaluation or tune after inspecting results.
