# Cat Image Arena

See [protocol.md](protocol.md) and [decisions.yaml](decisions.yaml) for the
registered scope, user-acknowledged governance gate, and prior-analysis limits.
The setup and notebook instructions below describe an earlier workflow: the
referenced notebooks are currently absent and the environment has not been
validated for the current plan. [proj-docs.md](proj-docs.md) contains prior
analysis, not results from the current protocol.

Pairwise cat-image preferences, and the midterm analysis of those choices. The analysis notes are in `proj-docs.md`.

## Stage 1 raw-data audit

From the project root, using Python 3.12:

```powershell
python -m src.run --config decisions.yaml --stage audit
```

The audit writes `tables/data_audit.csv`,
`tables/official_test_coverage.csv`, `artifacts/cleaning_log.csv`, and
`artifacts/raw_hashes.json`. It reads but does not modify the raw dataset. It
retains identical-looking responses and records self-comparisons for later
quarantine. A blocking audit issue produces a nonzero exit status after
writing its findings. Existing output files are not replaced unless
`--overwrite` is explicitly supplied.

Official overlap is classified against valid, non-self rows from the full
training data, not a fitting split. Component support and feature availability
remain explicitly unavailable until later stages. Image-file hashes and
existence do not establish that images decode successfully; feature extraction
must check that separately. The Stage 0 governance gate remains independent.

Run the focused audit tests with:

```powershell
python -m unittest discover -s tests -p test_audit.py -v
```

## Stage 2 canonicalization and pair split

The Stage 2 Parquet dependency is pinned separately in
`requirements-stage2.txt`. In PowerShell, create the project environment and
install that dependency if they are not already present:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-stage2.txt
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage split
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
```

Stage 2 requires an open governance gate and matching, unblocked Stage 1 audit
artifacts. It writes `artifacts/split_manifest.parquet` and
`artifacts/rows_canonical.parquet`; a validated existing split is retained,
not overwritten to match the seed. Pair assignment uses a seeded Python 3.12
`random.Random` shuffle in sorted comparison-type and band order.

The canonical rows contain fitting and validation outcomes, but internal-test
outcome columns are null. The original training CSV still contains those
outcomes: this is a procedural safeguard, not access control. After the model
freeze, evaluation must recover test outcomes by row ID from the unchanged raw
CSV and apply the recorded orientation. No modeling is performed here.

## Stage 3 feature integration and quality review

Install the pinned H/C visual dependencies in the project environment with
`.\.venv\Scripts\python.exe -m pip install -r requirements-stage3.txt`.
The following commands extract H and C without using preference outcomes:

```powershell
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage features --block H
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage features --block C
```

For the RTX 4050 with a CUDA 12.5-capable driver, Y uses a separate environment
with CUDA 12.4 PyTorch wheels. This preserves the CPU environment used for H/C.
Create the GPU environment only if it does not already exist:

```powershell
python -m venv .venv\gpu
.\.venv\gpu\Scripts\python.exe -m pip install -r requirements-y-gpu.txt
.\.venv\gpu\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
.\.venv\gpu\Scripts\python.exe -m src.run --config decisions.yaml --stage features --block Y --device cuda
```

The Y command fails rather than falling back to CPU if CUDA is unavailable.
Its manifest records the device, driver, runtime, precision, and library
versions. Once all available feature blocks are extracted, prepare the quality
review using the original CPU environment:

```powershell
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage feature-qa
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage feature-panels
```

Feature tables and their provenance manifests are written under `features/`.
The Y adapter uses the existing detector and CLIP rules, but does not calculate
coat measurements from the full frame when no cat is detected. Such values are
null, with an explicit no-detection category. CLIP's existing full-frame
fallback is retained and documented; its scores are diagnostic, not calibrated
attribute probabilities. Review software and checkpoint licenses before
redistributing weights or derived materials.

The QA command selects up to 48 hash-ranked fitting images and 16 additional
flagged fitting images without preference outcomes. The flagged subset uses a
fixed flag-family priority and hash order to represent rare categories. Flags include no detection,
dog fallback, multiple cat detections, small or box-fallback masks, low breed
margin, and handcrafted values outside fitting-photo 1st/99th percentiles.
It writes `tables/feature_quality.csv`, `artifacts/feature_review.csv`, and
`artifacts/feature_admission_decisions.csv`. Review panels show the original,
CNN crop, and detector mask; their paths are in the review CSV.

The two assigned reviewers must independently examine each panel. In their
rows, set `review_status` to `reviewed` and each review criterion to `pass`,
`issue`, `not_visually_verifiable`, or `not_applicable`; explain any issue in
`notes`. Do not change image IDs, selection metadata, or reviewer assignments.
Then complete each admission decision with `admitted` or `quarantined`, a
reviewer, a rationale, and `systematic_bugs_resolved=yes` for admission.
Automated checks and at least 95% strict-validation coverage are also required.

Only after these reviews and decisions:

```powershell
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage feature-freeze
```

This freezes `artifacts/common_population_manifest.parquet` and records its
coverage. Empty common validation or test populations are marked insufficient;
nonempty populations still require an adequacy assessment before model
selection. No internal-test outcomes are read by this workflow.
`tables/feature_quality.csv` is the pre-review automated-check snapshot; the
reviewed admissions are recorded in `artifacts/feature_admission_decisions.csv`
and `artifacts/common_population_report.json`.

## Stage 4 fitting graphs and eligibility

After feature review and population freeze, use the CPU environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-stage4.txt
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage graphs
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_graphs.py -v
```

The command creates full and matched fitting-photo and fitting-cat graphs,
with outcome-directed edges from fitting responses only. It saves graph
summaries, node components and support, edges, row-level strict-G2 masks,
split coverage, an enriched official-test coverage table, and a support
figure. Existing outputs are not overwritten. Strong connectivity describes
the ordinary unpenalized Bradley-Terry MLE condition; it does not determine
whether a penalized estimator returns finite scores.

The held-out eligibility masks use only image identities and fitting-graph
components. They do not use validation or internal-test outcomes. The frozen
Stage 3 population still requires an adequacy assessment before model
selection; official-test overlap is not a performance estimate.

## Stage 5 estimator tests and known-truth simulations

Install the numerical dependencies and run the focused checks before the
synthetic simulations:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-stage5.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_objective.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_recovery.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_orientation.py -v
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage simulations
```

The shared estimator uses float64 stable logit loss, per-coefficient ridge,
analytic gradients, zero-start L-BFGS-B, the registered acceptance threshold,
and one recorded retry. Weighted rows support exact count aggregation only
when all predictors match. Callers must inspect `FitResult.accepted`; a failed
fit is not a candidate. The pooled fallback copies the matched M3 predictions
exactly rather than approximating zero interactions through a matrix product.

The simulation output contains each fit and solver attempt. The three
demonstrations use 100 repetitions at 10, 50, and 200 responses per pair per
group, with ridge 1e-6 and no position intercept because the known-truth
examples have no position effect. The cycle's pair oracle is not a fitted BT
model. These are known-truth demonstrations, not real-data claims.

The required 20-repetition fair-coin synthetic-label control is **not yet
complete**: it must include the selected visual/contextual procedure at fixed
development settings, which does not exist until Stage 7. Run it before
final evaluation and investigate consistent unexpected gains. A successful
control does not prove all code paths leakage-free.

## Stage 6 development-only descriptive investigation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_descriptive.py -v
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage describe
```

The command verifies Stage 4 and Stage 5 artifact hashes, then reads only
fitting and validation outcomes. Tables label each summary as `fit`,
`validation`, or `development` (their union). Pair proportions use canonical
image ordering. Their 95% Wilson intervals are binomial working-model
summaries; respondent linkage is unavailable, so their assumptions cannot be
fully verified.

The pair, context-pair, context-overlap, context-support, photo-support and
support-band tables are written under `tables/`. Group contrasts require
shared pairs and at least five responses per pair-group cell; low overlap does
not support broad group-preference claims. The 100-response/30-pair interaction
rule is operational rather than a power guarantee. The example table applies
the fixed 20-response minimum without lowering it. Registered pair entropy is
an in-sample descriptive fit, not an oracle estimate or an information limit.
Outputs are protected against overwrite.

## Stage 7 development models

Install `requirements-stage7.txt` after the earlier stage environments. Run
`.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage develop`
from the project root. The command checks prior artifact hashes and reruns the
estimator, recovery, orientation, preprocessing, and modeling tests before
fitting. It refuses to overwrite an existing run. Its candidate log is written
incrementally, including failures; an interrupted run is not a completed
selection.

Preprocessing is fitted on unique fitting photos and saved under
`artifacts/preprocessing/`. Full and matched identity models and matched visual
and contextual models use the frozen populations. Validation predictions,
candidate diagnostics, selected model coefficients, and selection metadata are
saved separately. Internal-test outcomes are not read. The fair-coin control
and development diagnostics are still required before final evaluation.
`artifacts/stage7_verification.json` records a reconstruction check of the
selected validation predictions and hashes the saved development artifacts.

## Stage 8 development diagnosis

The Stage 8 registry is in `artifacts/stage8_protocol.json`; the separate probe
and contribution registries document their fixed rules. The analysis uses
fitting and common strict-validation responses only. It does not change the
Stage 7 model winner or load internal-test outcomes.

After earlier stage artifacts have been verified:

```powershell
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage diagnose-core
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage diagnose-curves
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage diagnose-curve-support
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage diagnose-representations
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage diagnose-reports
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage diagnose-contributions
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage diagnose-finalize
```

The fair-coin control uses 20 independent fitting/validation label draws at the
fixed selected development settings. A passing control does not certify
leakage freedom. Learning curves refit preprocessing, context support, and
penalties within nested fitting-pair subsets. The minimum common validation
coverage of 100 responses and 30 pairs is operational, not a power guarantee.
Per-subset preprocessing and context fallbacks are saved.

Penalty-path refits reproduce Stage 7 validation losses and add coefficient
norms. H/Y grouped ablations, the C PCA probe, temperature scaling, and
reconstruction of estimated photo scores are secondary development
diagnostics. Contributions explain model outputs, not causal human preferences.
Every stage refuses to overwrite an existing output; an interrupted run is not
a completed diagnostic or test freeze.

## Stage 9 freeze and internal-test evaluation

Run the evaluation tests, then freeze before loading internal-test outcomes:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_evaluation.py -v
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage freeze
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage evaluate
```

`artifacts/model_freeze.json` and its SHA-256 seal record the selected models,
preprocessing, context mappings, population masks, optional diagnostics, Git
state, input hashes, and four primary contrasts. Evaluation refuses a missing
or changed freeze, missing predictions, mismatched paired IDs, or existing
test outputs. It records test-outcome access before loading the original
training choices by internal-test response ID. If evaluation is interrupted
after outcome access, it must not be silently repeated or retuned.

The evaluation saves row-level predictions, overall and subgroup metrics,
calibration, and pair-cluster bootstrap intervals. The four primary contrasts
receive approximate 98.75% pointwise intervals; secondary intervals are
descriptive and not multiplicity-adjusted. Intervals are conditional on fitted
models and development decisions, and do not capture all training variability,
respondent dependence, shared-image dependence, or prior analysis exposure.
Temperature scaling is secondary and uses the values fixed during Stage 8;
it does not alter the primary model choices.
For temperature-diagnostic rows in `tables/main_results.csv`, the fitting
loss, penalized objective, and validation loss columns describe the original
unscaled source model. The scaled validation results and selected temperatures
are recorded in `tables/temperature_diagnostic.csv`; the test metrics in
`main_results.csv` use the frozen temperature.

## Setup

Python 3.12. From the project root:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Use `.venv` as the Jupyter kernel, then run the notebooks in order:

1. `notebooks/01_train_eda_cleaning.ipynb`
2. `notebooks/02_train_statistical_indices.ipynb`
3. `notebooks/03_bradley_terry.ipynb`

`dataset/train.csv` and `dataset/test.csv` are left unchanged.

## Dataset

A dataset of pairwise cat-image preferences for data analysis and prediction.

## Objectives

- **Explore and interpret the data:** examine human preferences, disagreement between annotators, and relationships between annotator characteristics and image preferences.
- **Predict preferences:** use the training data to predict each annotator’s image preference in the test data. In `test.csv`, fill in `prob_a` with the predicted probability of choosing image A (between 0 and 1). Fill in `choice` with `A` if `prob_a ≥ 0.5`, or `B` otherwise.

Each row represents one annotator's response to one image pair. A pair may have multiple responses from different annotators.

## Files

| File | Description |
| --- | --- |
| `dataset/train.csv` | 24,896 responses with observed `choice` values. |
| `dataset/test.csv` | 6,224 test rows with `choice` and `prob_a` left blank. |
| `dataset/images/` | 1,558 images organized into cat folders. |

## Image paths

Get the images from [Hugging Face](https://huggingface.co/datasets/missing12/cat-image-arena-images/tree/main/images).

The `a_image` and `b_image` fields contain paths relative to `dataset/images/`. For example:

```text
Table value: cat_000002/cat_000002_0001.jpg
Local file:  dataset/images/cat_000002/cat_000002_0001.jpg
```

## Fields

| Field | Meaning |
| --- | --- |
| `train_row_id` | Unique training-row identifier; present only in the training file. |
| `test_row_id` | Unique test-row identifier; present only in the test files. |
| `a_cat`, `b_cat` | Cat identifiers for sides A and B. |
| `a_image`, `b_image` | Image paths for sides A and B. |
| `choice` | Observed preference: `A` or `B`. Withheld in the test files. |
| `prob_a` | Predicted probability of choosing image A, between 0 and 1; present only in the test file. |
| `comparison_type` | `within_cat`: different photos of the same cat; `between_cats`: photos of different cats. |
| `gender` | Self-reported gender. |
| `pet_history` | Household pet ownership: `current`, `past`, or `never`. |
| `pet_care` | Personal pet-care experience: `never`, `occasionally`, `past_regular`, or `current_regular`. |
| `pet_contact` | Frequency of interaction with pets: `never`, `rarely`, `sometimes`, `often`, or `daily`. |
| `future_pet` | Intention to own or continue owning pets: `yes`, `no`, or `unsure`. |
| `mbti_ie` | `I` or `E`, derived from self-reported MBTI; `unknown` means unsure or not tested. |

For research and educational use only.
