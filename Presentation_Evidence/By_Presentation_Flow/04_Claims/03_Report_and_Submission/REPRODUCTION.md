# Group 19 reproduction and submission guide

## Status and restrictions

The report is a presentation-ready **working draft**. The user requested an
explicit contribution placeholder; the team must replace it with verified
completed work and complete the course Chapter 7 reference before submission.
All five proposed presenters cover substantive sections. The official upload
channel is **not verified**. No submission has been uploaded.

The course deadline is 2 November 2026 at 12:00 noon HKT; the internal target
is 11:00 a.m. HKT. Required filenames are `Group_19_Report.pdf` and
`Group_19_Code.zip`.

The archive is for the teaching team and education/research only. It contains
course CSVs and cached derived artifacts with that restriction. Do not publish
it publicly or use it commercially. Raw photographs, pretrained weights,
virtual environments, `.git`, secrets, and download caches are excluded.
See `tables/third_party_resources.csv`. In particular, review Ultralytics
AGPL/checkpoint terms for the distribution context; this guide is not blanket
license clearance.

## Portable cached-feature reproduction

Tested platform: Windows AMD64, Python 3.12.0. Use the extracted project root,
not the parent workspace. This statistical route needs the included CSVs and
cached feature tables, but not raw photographs or model-weight downloads:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-cpu.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'
.\.venv\Scripts\python.exe -m src.reproduction_check --output artifacts\reproduction_fresh.json
```

Choose a new output filename for each check; existing logs are never
overwritten. The wrapper creates and removes its own temporary working copy,
runs a new interpreter process, and preserves all submitted logs and frozen
artifacts. It uses portable paths relative to the extracted root. Absolute
checkpoint paths inside historical manifests are provenance from the original
machine; the cached statistical route does not open them.

The check refits the **selected models at the frozen penalties** in memory,
recomputes preprocessing on unique fitting photos, checks validation metrics,
and reconstructs test metrics from **already-saved test prediction outcomes**.
It recomputes paired bootstrap intervals from saved losses. It does not retune,
reselect models, reopen raw internal-test outcomes, overwrite predictions, or
constitute a new evaluation. The original freeze is still the evidence for the
single internal holdout.

Recorded results and observed timing are in
`artifacts/reproduction_statistical.json`. The acceptance tolerance is
`1e-6` for reported metrics/probabilities; inspect the recorded maximum
differences rather than assuming exact agreement on another platform. This is
not a claim that the entire raw-image-to-report process was re-executed in a
newly installed environment.

## Environment roles

- `requirements-lock-cpu.txt`: exact CPU/report/statistical environment
  snapshot, including PyTorch 2.14.1 CPU, Torchvision 0.29.1 and pypdf.
- `requirements-lock-y-gpu.txt`: exact Y extraction environment snapshot,
  including PyTorch 2.6.0+cu124 and Torchvision 0.21.0+cu124.
- The `requirements-stage*.txt` files document the staged dependencies.
- Root `requirements.txt` describes the earlier legacy setup. It is not the
  tested Stage 11 statistical/report setup.

Python, Windows, hardware, runtime and numerical differences are recorded in
the reproduction logs and feature manifests. The original machine has 16 GB
RAM, 20 logical processors and an RTX 4050 Laptop GPU with about 6 GB VRAM.
The installed driver 555.99 reports CUDA 12.5 capability; CUDA 12.4 PyTorch
wheels were used successfully. GPU availability must be checked in the actual
GPU environment; the CPU environment deliberately reports no CUDA support.

## Feature-regeneration check and full extraction

Get the source images from the course's image dataset:
<https://huggingface.co/datasets/missing12/cat-image-arena-images/tree/main/images>.
Place them as `dataset\images\cat_...\cat_...jpg`, without changing the CSVs.
The original source file hashes are in `artifacts\raw_hashes.json`.

H and C can be checked on the fixed subset in the CPU environment:

```powershell
.\.venv\Scripts\python.exe -m src.reproduction_check --mode H --output artifacts\feature_H_fresh.json
.\.venv\Scripts\python.exe -m src.reproduction_check --mode C --output artifacts\feature_C_fresh.json
```

For Y, create the separate GPU environment only if absent:

```powershell
python -m venv .venv\gpu
.\.venv\gpu\Scripts\python.exe -m pip install -r requirements-lock-y-gpu.txt
.\.venv\gpu\Scripts\python.exe -c "import torch; print('GPU Available:', torch.cuda.is_available())"
.\.venv\gpu\Scripts\python.exe -m src.reproduction_check --mode Y --output artifacts\feature_Y_fresh.json
```

The source images and caches are copied only into the wrapper's temporary
working copy. Pretrained weights are downloaded separately when not already
cached; they are not in the ZIP. MobileNet uses roughly 10 MB, YOLO roughly
6 MB, and the OpenAI CLIP checkpoint roughly 600 MB; the CUDA PyTorch wheel is
roughly 2.5 GB. Exact checkpoint identities and hashes are in
`features\manifests\C.json` and `features\manifests\Y.json`.

The fixed-subset logs `artifacts\feature_regeneration_H.json`,
`feature_regeneration_C.json`, and `feature_regeneration_Y.json` record actual
observed differences. H/C were identical on their checked subset; Y's maximum
numeric difference was approximately `2.52e-8`, with matching hard labels and
selected masks, below the `1e-5` operational check tolerance. These checks do
not promise bitwise equality across devices, library versions, or all images.
Original full Y extraction took roughly 10 minutes on this laptop; do not
assume the CLIP pipeline finishes in a few seconds just because CUDA exists.

To regenerate all features, use a separate clean working copy with the raw
dataset, audit artifacts, fixed split, Stage 0 decisions and code. Follow the
Stage 1-3 commands in `README.md`, with no `features\H.parquet`,
`features\C.parquet`, or `features\Y.parquet` already at the destinations:

```powershell
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage features --block H
.\.venv\Scripts\python.exe -m src.run --config decisions.yaml --stage features --block C
.\.venv\gpu\Scripts\python.exe -m src.run --config decisions.yaml --stage features --block Y --device cuda
```

Do not delete or overwrite the submitted frozen feature caches to force this.
New extraction manifests can differ in timestamps, device details and machine
paths. Feature admission still requires the recorded QA and human-review
process before any new model development.

## Analysis order and report regeneration

The source modules and original artifacts record the order:

1. `src.run --stage audit`: raw validation/hashes, no modifications to raw data.
2. `src.run --stage split`: preserve or create pair allocation; mask test outcomes.
3. `features`, `feature-qa`, `feature-panels`, `feature-freeze`: extraction,
   outcome-blind human review, then admission and matched populations.
4. `graphs`: fitting graphs, support and structural strict-G2 eligibility.
5. `simulations`: known-truth estimator checks.
6. `describe`: development-only disagreement/support summaries.
7. `develop`: fitting, validation tuning and matched model selection.
8. `diagnose-*`: registered development diagnostics and independent fair-coin
   controls; see the exact command order in `README.md`.
9. `freeze` then `evaluate`: the original single frozen internal evaluation.
   The submitted evaluated copy refuses re-evaluation.
10. `python -m src.synthesis`, presentation coefficient figures, and the
    exploratory semantic append: these were generated from saved evidence.
11. Fresh-process checks, PDF report and restricted code archive.

For a complete new analysis, use a separate clean project root and the exact
saved split where valid. Do not try to rerun write-once stages in the submitted
evaluated tree. Scope extensions after test access are exploratory, not a way
to create a fresh confirmatory test.

The report is generated directly from saved tables:

```powershell
.\.venv\Scripts\python.exe -m src.report --output Group_19_Report_Revised.pdf
```

The report deliberately contains a contribution placeholder, as requested.
Edit that slide or its source before submitting. If the PDF is edited manually,
retain that fact and rebuild the package/checksums; do not present the original
report hash as identifying a changed file.

## Expected structure

```text
project-root\
  Group_19_Report.pdf
  README.md
  REPRODUCTION.md
  decisions.yaml, protocol.md, claims*.csv, contributions.csv, references.csv
  requirements-lock-cpu.txt, requirements-lock-y-gpu.txt
  src\, tests\, legacy\
  dataset\train.csv, dataset\test.csv
  features\H.parquet, C.parquet, Y.parquet, manifests\
  models\, tables\, artifacts\, figures\
```

The ZIP contains no `.venv`, `.git`, raw image files, or pretrained `.pt`/`.pth`
weights. An official prediction file is optional and separate from the
reported frozen evaluation; it is not produced or claimed by this report.
