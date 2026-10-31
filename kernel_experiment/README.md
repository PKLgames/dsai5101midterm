# Retrospective nonlinear CNN kernel experiment

This is a quarantined pooled M3-C extension. The selected nonlinear RBF
Nyström model uses one scalar score per photo; it does not add context or
demographic interactions. The original G2/G3/G4 outcomes were previously
exposed, so all results are exploratory, not independently confirmatory.

The fixed protocol specifies unique-fitting-photo CNN preprocessing, a
fitting-distance bandwidth scale, three kernel-width candidates, up to 256
hash-ranked fitting-photo landmarks, fitting-only Nyström centering/RMS,
seven ridge values, zero-start analytic-gradient logistic fitting and
validation-only selection. Every learning-curve and transfer subset refits
all learned transformations. Original G3 and G4 fitting/validation splits
are used separately; test comparisons use saved prediction outcomes only.

`RESULTS.md` summarizes the evidence and its noncausal limits. G2 is compared
with original linear M3-C and contextual M4-C on the same 3,733 responses.
G3 resamples held-out photos and G4 held-out cats; G4 is the one-unseen-cat
target, not two unseen cats. The separate C+D comparison uses demographics
and cannot isolate nonlinearity.

The fixed-setting fair-coin control, original-style calibration/support
slices, within/between-cat metrics, G3 within-cat follow-up, nested curves,
21 independently replayed penalty refits, validation-fixed temperature and
fixed-grid +1/-1/0 abstention are recorded locally. Original outcomes were
previously exposed. Pointwise 95% intervals are unadjusted and conditional
on the fitted models; there is no test-selected deployment model or cutoff.

Four focused tests passed. `verification.json` reconstructs selected
fit-only transformations/gradients and saved-test cluster intervals;
`additional_verification.json` checks penalty refits, additional comparisons
and original support-slice matching. Full-size learning-curve rows share the
same underlying fitting population, so the two seeds are not independent.
Two interrupted 25% curve attempt logs are retained with explicit names after
workers were stopped to reduce shared memory pressure; the final complete
candidate logs were subsequently generated from zero initialization.

All commands run from the project root using the existing CPU environment.
Set `$env:PYTHONDONTWRITEBYTECODE='1'`. The scripts are write-once and refuse
to overwrite existing development, freeze, evaluation or publication files.
Do not delete access guards to repeat or tune on the test. The original
models, primary claims, report, ZIP, G3/G4/demographic results and official
prediction file remain untouched.
