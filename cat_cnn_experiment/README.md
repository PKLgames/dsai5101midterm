# Retrospective Cat + CNN additive scorer

The model is a pooled Bradley–Terry-style score:

`eta = beta + c(a_cat) - c(b_cat) + w.T @ (z_A - z_B)`.

The two positive ridge penalties are selected on validation from the recorded
7 × 7 grid. The CNN embedding is centered and block-RMS normalized using
unique fitting photos. Cat identity and CNN features may encode overlapping
information, so regularization determines their allocation; do not interpret
the coefficient decomposition as causal or behavioral.

**Protocol correction:** the frozen protocol's blanket statement that the
cat/CNN components are not uniquely identified by the likelihood is too
broad. `design_rank_and_protocol_erratum.json` records an outcome-blind G2
design-rank check: rank 813 of 814 columns at the stated tolerance, consistent
with only the usual one-dimensional cat-score offset gauge. After fixing that
gauge, no further exact rank deficiency was detected numerically. Penalties
and correlated predictors still affect fitted coefficients, and neither this
rank result nor the coefficient split establishes causal interpretation.

G2 uses the original common fitting/validation/test populations. G3 uses a
separate fit and validation on its existing reserved-photo split. **G4 is
unsupported by explicit user choice**: no zero-score cold start was applied
and no G4 predictions were produced. All G2/G3 outcomes were previously
exposed, so the results are post-test, retrospective and not independently
confirmatory.

The study records fixed-setting fair-coin controls, matched validation/test
calibration and support slices, within/between-cat and G3 within-cat
comparisons, two-seed learning curves, a full independently reproduced
penalty path, validation-selected temperature and fixed-grid abstention.
All displayed 95% intervals are pointwise unadjusted and conditional on
selected fits. No outcome-driven deployment or threshold choice was made.

`RESULTS.md` contains the evidence-linked synthesis. `verification.json`
reconstructs the selected models, fit-only preprocessing, G2/G3 bootstrap
draws and abstention cells. The full-size curve inputs are identical between
seeds and are not independent replications. Original G2/G3/G4/demographic/
kernel models, primary claims, reports, ZIP and official predictions remain
unchanged.

Use the existing project CPU environment and set
`$env:PYTHONDONTWRITEBYTECODE='1'`. Study stages and publication are write-once;
do not remove a test-access guard to rerun or retune after looking at outcomes.
