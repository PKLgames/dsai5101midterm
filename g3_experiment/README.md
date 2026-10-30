# Retrospective known-cat G3 experiment

This is a quarantined, exploratory supplement. G3 was not reserved at Stage 0,
and the original G2 test had already been evaluated. Some G3 outcomes were
therefore previously exposed, and the separate G3 fitting set includes original
G2 test responses. This is not a new independent or confirmatory holdout.
The original G2 split, frozen models, claims, report, and archive are unchanged.

The registered procedure is in `protocol.json`. `prepare.py` reserves
duplicate-content photo groups without reading outcomes, constructs an
independent pair split among nonreserved photos, applies known-cat/known-photo
eligibility, and checks the plan's coverage gate. Models fit only on the new
core fitting rows; feature preprocessing is fitted on their unique photos.
Penalties and representations are selected on the reserved G3 validation
rows. `freeze.json` and `freeze.sha256` seal these selections and their inputs
before the G3 test outcomes are requested. G3 test intervals use 5,000 shared
held-out-photo bootstrap resamples; they do not capture all possible shared
known-photo or respondent dependence.

There is deliberately no M2 unseen-photo free score, no G2 artifact modification,
and no deployment or original-claim update. The copy-only presentation
evidence update is recorded separately from the original evidence manifests.

From the project root, using the existing project environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe g3_experiment\tests.py
```

The development and evaluation scripts are write-once and deliberately refuse
to rerun after they have started. In particular, `evaluation_started.json`
records test access before outcomes are loaded; do not remove it to repeat an
evaluation or tune models after inspecting results.
