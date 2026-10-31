# Demographic diagnostic parity

This is a quarantined, retrospective extension of the all-demographic G2
experiment. It does not modify the original demographic/G2/G3/G4 freezes,
claims, report, package or official predictions.

The existing fair-coin controls, overall calibration and within/between-cat
results are preserved and reused. New evidence includes:

- Original-style validation/test graph-support and overlapping feature-quality
  slices, plus calibration by all six demographics.
- Validation-fixed temperature diagnostics and fixed-cutoff abstention utility
  using the previously saved G2 test predictions.
- Full registered demographic penalty-grid refits and two-seed nested-pair
  learning curves with fit-only preprocessing, demographic support and
  validation retuning.
- Separate G3/G4 refits using only their respective fitting and validation
  rows. Existing saved test outcomes and held-out-photo/cat bootstrap resamples
  are used after each new transfer freeze. G4's unseen cat has both pooled and
  demographic-specific cat-score components fixed to zero.

All results are post-test exploratory and not independent or causal. No
temperature, utility cutoff, transfer outcome or model is promoted to the
original analysis or deployment. Demographics-only cannot rank cats/photos.
Displayed A is not established to mean left.

The original and supplemental evidence inventories are historical snapshots;
the catalog's diagnostic parity update supplies a later inventory delta.
The copied evidence has its own new manifest and verification record.

## Reproducibility and resource handling

Use the project CPU environment and set `PYTHONDONTWRITEBYTECODE=1`. Each
analysis writes to a new, write-once destination. Existing test reanalysis
guards must not be deleted to retune or repeat after outcome inspection.

High-dimensional CNN fits are memory intensive. During the original run,
learning-curve workers were paused to reduce pressure on a shared machine.
Partial candidate logs were preserved under explicit `interrupted` names;
completed 25%/50% artifacts were hash-verified and retained. The final 100%
fitting set and validation rows are identical for the two seeds, so one
complete full-size fit is reused only after checking exact identity. This is
not two independent fits. See `resource_adjustment.json` and
`full_size_reuse_provenance.json`.

The tables retain every final candidate/fallback and numerical acceptance;
the 95% diagnostic intervals are pointwise and unadjusted for multiplicity.
See `protocol.json`, `diagnostic_summary.json`, `verification.json` and
`DIAGNOSTIC_EVIDENCE.md` for scope, exact counts, numerical results and limits.
