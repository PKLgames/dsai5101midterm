# Cat Image Arena — feature extraction and preference modelling (CPU only)

This project turns the 1,558 cat photos of the Cat Image Arena dataset into fixed-length feature
vectors and uses them to predict **whether a given observer will choose photo A or photo B**. It
runs end to end on a GPU-less laptop (12th Gen Intel i7-1260P, 7 GB RAM) in a few minutes.

It also borrows a few reusable components — per-cat strengths and the YOLO/CLIP photo-label
table — from a second project (`dsai-5101-sdm-proj`, described by `README_allie_chan.md`), and
compares feature families under one estimator and one set of folds.

---

## Results at a glance

### Feature pipeline

| Stage | Measured |
|---|---|
| Download 1,558 images from Hugging Face | 183.8 MB in **115 s** (16 threads, resumable) |
| Extract features (MobileNetV3-Small @224, frozen) | **36–38 s ≈ 41–43 img/s**, peak RSS **≈670 MB** |
| Feature quality (same-cat vs different-cat cosine) | 0.803 vs 0.692, gap **+0.111**, AUC **0.844**, 5-NN cat retrieval **27.7 %** (chance 0.38 %) |

### Which feature family helps predict preferences

Pair-grouped 5-fold over all 24,896 rows, one estimator (L2 logistic regression + side column,
`C = 1`). Details and per-observer metrics: `docs/feature_choices.md`.

| feature family | log loss | accuracy | both photos unseen | both cats unseen |
|---|---:|---:|---:|---:|
| observer demographics only | 0.6900 | 0.532 | 0.6901 | 0.6893 |
| coarse labels (fur colour) | 0.6828 | 0.553 | 0.6823 | **0.6805** |
| coarse labels (all) | 0.6800 | 0.560 | 0.6833 | 0.6841 |
| hand features (20) | 0.6865 | 0.543 | 0.6878 | 0.6892 |
| frozen embedding (576) | 0.6674 | 0.591 | **0.6797** | 0.6869 |
| embedding + hand | 0.6658 | 0.592 | 0.6786 | 0.6876 |
| per-cat memory | 0.6600 | 0.600 | 0.6776 | 0.6913 |
| cat memory + embedding + hand (joint) | 0.6565 | 0.606 | 0.6847 | 0.6945 |
| **routed: joint if cat+photo seen, else fur colour** | **0.6565** | 0.606 | **0.6823** | **0.6805** |
| family median (log loss) | — | — | 0.6837 | 0.6892 |

Three conclusions that shaped the code:

1. **Annotator demographics do not predict preferences** (0.6900, worse than a constant).
2. **Seen cat → memory wins; unseen cat → appearance wins**; a single joint hybrid therefore
   loses on brand-new cats (0.6945) even though it is best overall (0.6565).
3. **Coarse labels calibrate best (ECE 0.007) and transfer best to new cats** (0.6805), while the
   dense embedding carries most of the accuracy on seen cats.
4. **More features do not fix the corner.** Concatenating a second backbone raises the average
   (0.6659 → 0.6607) and *worsens* unseen cats (0.6874 → 0.6938). Routing to a transfer model
   fixes it at no cost on the average: 0.6565 / 0.6823 / 0.6805, both cold slices below the
   family medians (0.6837 / 0.6892).

---

## Layout

```
dsai5101midterm/
├── README.md                     ← this file (English)
├── README.zh-CN.md               ← Chinese version, same content
├── requirements.txt              ← pinned versions (torch must come from the CPU index)
├── Dataset/                      ← course CSVs + brief (no images)
├── src/catarena/
│   ├── paths.py                  relative paths, category levels, approach-A locations
│   ├── index.py                  CSV -> image index / pair keys
│   ├── download.py               concurrent HF download, resume, manifest
│   ├── preprocess.py             deterministic inference transform (224 centre crop)
│   ├── backbones.py              frozen backbones (mnv3s / resnet18 / efficientnet_b0)
│   ├── handcrafted.py            20 interpretable low-level statistics
│   ├── extract.py                batching, throughput, npz + meta cache
│   ├── features.py               cache loader, pairwise design matrices, demographics
│   ├── qa.py                     similarity / PCA / nearest-neighbour diagnostics
│   ├── hybrid.py                 hybrid model (cat memory + appearance + labels)   [this project]
│   └── approach_a/               reusable components copied from the other project
│       ├── strength.py           per-item strengths + side term (their fit_bt)
│       ├── labels.py             reader/encoders for their YOLO+CLIP label table
│       └── metrics.py            log_loss / Brier / accuracy with their definitions
├── scripts/
│   ├── 00_setup_env.sh           CPU-only venv (uv + download.pytorch.org/whl/cpu)
│   ├── 01_download_images.py     1,558 images, resumable
│   ├── 02_build_index.py         image_index.csv + overlap checks
│   ├── 03_extract_features.py    feature cache (backbone / size / TTA / limit flags)
│   ├── 04_feature_qa.py          diagnostics + figures
│   ├── 05_baseline_pairs.py      embedding baseline, two leak-aware protocols, submission
│   ├── 07_feature_families.py    feature-family comparison + per-observer metrics
│   ├── 08_cold_start_fix.py      better features vs routing, on the cold-start corner
│   └── 09_hybrid_submission.py   fits the final model on all rows, writes the predictions
├── docs/
│   ├── pipeline_design.md        ← design rationale (English)
│   ├── pipeline_design.zh-CN.md  ← same, Chinese
│   ├── feature_choices.md        ← which image features help, with metrics (English)
│   └── feature_choices.zh-CN.md  ← same, Chinese
└── data/                         ← generated, git-ignored (teaching/research use only)
    ├── images/<cat>/<photo>.jpg  1,558 photos, 183.8 MB
    ├── manifest.csv              bytes, dimensions, source URL
    ├── image_index.csv           cat / split / appearance counts per photo
    ├── features/
    │   ├── <tag>.npz             paths, deep (N,576), hand (N,20), ok
    │   ├── <tag>.meta.json       backbone, versions, timings, RSS, seed
    │   ├── <tag>.csv             per-photo table (1,558 × 597)
    │   ├── qa_*_<tag>.{png,json} feature diagnostics
    │   ├── baseline_metrics.{json,csv}
    │   ├── feature_families.{json,md,png}
    │   ├── cold_start_fix.{json,md}
    │   └── hybrid_fit.json       final spec, routing shares, slice summary
    └── submissions/
        ├── pred_test.csv         embedding baseline
        ├── pred_hybrid_test.csv  joint hybrid (--structure joint)
        └── pred_routed_test.csv  routed model (recommended)
```

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/). The course CSVs must sit in `Dataset/` (already
included). Images are not shipped — they are downloaded.

```bash
cd dsai5101midterm

# 1) CPU-only environment (torch/torchvision from the PyTorch CPU index, ~200 MB)
bash scripts/00_setup_env.sh

# 2) images (183.8 MB, about two minutes)
.venv/bin/python scripts/01_download_images.py

# 3) index -> features -> diagnostics -> baseline
.venv/bin/python scripts/02_build_index.py
.venv/bin/python scripts/03_extract_features.py --workers 4 --threads 8
.venv/bin/python scripts/04_feature_qa.py
.venv/bin/python scripts/05_baseline_pairs.py
```

This yields `data/features/mobilenet_v3_small_224.npz` (1,558 × 576 deep + 1,558 × 20 hand) and
`data/submissions/pred_test.csv` (6,224 rows).

The comparison and the hybrid need the other project cloned next to this one
(`../dsai-5101-sdm-proj`) for its `prepared/photo_labels.csv`; it is opened read-only:

```bash
.venv/bin/python scripts/03_extract_features.py --backbone resnet18    # optional 2nd backbone, ~1 min
.venv/bin/python scripts/07_feature_families.py      # ~3.5 min
.venv/bin/python scripts/08_cold_start_fix.py        # ~7 min
.venv/bin/python scripts/09_hybrid_submission.py     # ~30 s, writes the routed submission
```

## Scripts

| Script | Key flags | Purpose |
|---|---|---|
| `00_setup_env.sh` | `PYTHON_TAG`, `TORCH_VER` | build `.venv`; CPU wheels avoid ~2.5 GB of nvidia dependencies |
| `01_download_images.py` | `--workers`, `--limit` | 16 threads, 3 retries with backoff, atomic `.part` writes, skip-if-present |
| `02_build_index.py` | — | `image_index.csv`, train/test overlap and missing-file checks |
| `03_extract_features.py` | `--backbone`, `--size`, `--batch-size`, `--workers`, `--threads`, `--tta-flip`, `--limit`, `--tag`, `--list-backbones` | extract, cache, report throughput |
| `04_feature_qa.py` | `--tag`, `--no-plots` | same/different-cat similarity, 5-NN retrieval, PCA, neighbour grid |
| `05_baseline_pairs.py` | `--tag`, `--C`, `--p2-folds`, `--seed` | pair-grouped CV + image-holdout, submission |
| `07_feature_families.py` | `--folds`, `--p2-folds`, `--seed`, `--no-coldstart`, `--replot` | ten feature families, three protocols, per-observer and calibration metrics |
| `08_cold_start_fix.py` | `--reuse` | multi-backbone features and routing against the cold-start corner |
| `09_hybrid_submission.py` | `--structure routed\|joint`, `--spec`, `--C`, `--sweep` | fit the final model, write predictions |

Backbones (identical interface, one flag):

| Backbone | Dim | Params | Throughput (128-image probe) | Peak RSS |
|---|---|---:|---:|---:|
| `mobilenet_v3_small` | 576 | 2.5 M | **24.2 img/s** | 654 MB |
| `resnet18` | 512 | 11.7 M | 15.2 img/s | 766 MB |
| `efficientnet_b0` | 1280 | 5.3 M | 17.3 img/s | 1.08 GB |

## Feature cache format

`data/features/<tag>.npz` (`np.load`, no pickle):

| Key | Shape | Meaning |
|---|---|---|
| `paths` | (N,) str | path relative to `data/images/`, byte-identical to `a_image` / `b_image` |
| `deep` | (N, 576) f32 | L2-normalised frozen-CNN pooled embedding |
| `hand` | (N, 20) f32 | interpretable low-level statistics (`hand_features` in the meta) |
| `ok` | (N,) bool | decoded successfully (1,558/1,558 here) |
| `hand_names` | (20,) str | column names |

`<tag>.meta.json` records backbone and weight file, input size, normalisation constants,
torch/torchvision versions, thread count, batch size, wall time, img/s, peak RSS, CPU model, seed
and timestamp — enough to satisfy a report's hardware/runtime section.

Dimensions: **596 per photo** (576 deep + 20 hand); a pair row is **578** (`f_A − f_B` +
`comparison_type`), **604** with the 26 demographic one-hot columns, or **1180** with
`diff_absdiff`.

## Using the features

```python
import sys; sys.path.insert(0, "src")
from catarena import features, index

feats = features.load_features("mobilenet_v3_small_224")   # None = most recent cache
X, present_a, present_b = features.pair_matrix(df, feats, mode="diff", with_demo=True)
y = features.label_a(df)        # 1 when side A was chosen
# X: (n_rows, 604) = 576 embedding difference + 26 demographics + 2 comparison types
```

Image/feature correspondence is by **path string**, not row order; `lookup()` returns a zero row
plus `present=False` for anything missing (0 misses across all 31,120 train/test photo positions).

Pair design matrices are computed on the fly (~0.1 s) and never cached to disk, so changing the
encoding does not require re-exporting anything.

## Hybrid model

`catarena.hybrid.HybridModel` is one L2 logistic regression over difference-coded blocks:

```
[ cat one-hot (cats seen in the fit) | 576-d embedding | 20 hand (z-scored) | labels | side ]
```

* Cat memory comes from `catarena.approach_a.strength`; an unseen cat contributes 0.
* Appearance blocks are this project's; they carry the signal for unseen photos and for
  within-cat comparisons (where the cat gap is 0).
* Labels are optional and read from the other project's `prepared/photo_labels.csv`.
* `scripts/09` fits `cat + embedding + hand` at `C = 0.3` on all 24,896 rows (834 columns) and
  writes `data/submissions/pred_hybrid_test.csv`.

Measured trade-off (see `docs/feature_choices.md` §6b): the joint model is the best when the cat
is known (0.6565 overall), but it is **worse than the embedding alone for entirely new cats**
(0.6945 vs 0.6869) because the cat dummies absorb cat-level variance. A second backbone does not
help (0.6938 on new cats); **routing does**:

* joint model when both cats **and** both photos were present in training (50 % of test rows);
* a fur-colour-only transfer model otherwise.

Result: average unchanged (0.6565), both photos unseen 0.6823, both cats unseen 0.6805 — below
the family medians 0.6837 / 0.6892. `scripts/09_hybrid_submission.py --structure routed`
writes `data/submissions/pred_routed_test.csv`.

## Provenance of the borrowed components

Kept in a dedicated subpackage so the boundary is obvious; only reusable pieces are copied, and
no code re-runs the other project's analysis.

| File | Origin |
|---|---|
| `src/catarena/approach_a/strength.py` | adapted from `analysis/03_cat_strength_model.ipynb` (`fit_bt`, `predict_bt`) |
| `src/catarena/approach_a/labels.py` | reads `prepared/photo_labels.csv` produced by `photo-labelling/*.py`; encoders mirror their `numeric_diff` / `one_hot_diff` / `with_side` |
| `src/catarena/approach_a/metrics.py` | their `log_loss` / `brier` / `accuracy` definitions |

Nothing else is imported from that project, and nothing here writes to it. Everything outside
`catarena/approach_a/` is original to this project.

## Reproducibility

* Fixed: path ordering, `torch.manual_seed(0)`, fold seeds (`0` for baseline and feature-family
  runs), `C` selected on a pair-grouped holdout where reported.
* Versions are pinned in `requirements.txt`; each extraction records the actual environment in
  `<tag>.meta.json`.
* Known numerical variation: torch thread count changes floating-point summation order (4th
  decimal of log loss); everything else reproduces bit-for-bit (no random augmentation, no
  dropout, no GPU non-determinism). Re-running `05` reproduces `baseline_metrics.json` exactly;
  re-running `07` reproduces `feature_families.json` (seeded KFold, deterministic estimator).
* Sandboxes that forbid POSIX semaphores make DataLoader workers unavailable; the extractor falls
  back to `num_workers=0` automatically (43 img/s here) and records `num_workers_requested` vs
  `num_workers` in the meta.

## Limitations

* Human preference is noisy: two observers agree on 55.7 % of repeated pairs. Any model is capped
  well below "certainty", and on disputed pairs every family sits at 0.50 accuracy.
* There are no annotator IDs, only 70 demographic profiles, so "different observers" is an
  approximation; that is also why demographics turned out useless.
* The two cold starts (new photos vs new cats) give opposite rankings. Report both.
* Test labels are withheld, so none of these numbers is a leaderboard score.
* Low-level hand features add little accuracy; their value is interpretability.
