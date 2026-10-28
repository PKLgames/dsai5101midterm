# Cat Image Arena — feature extraction and preference modelling (CPU only)

This project turns the 1,558 cat photos of the Cat Image Arena dataset into fixed-length feature vectors, then predicts **whether a given observer will choose photo A or photo B**. It runs end to end on a GPU-less laptop (12th Gen Intel i7-1260P, 7 GB RAM) in a few minutes.

Cat strength and the YOLO/CLIP labeller live in this repository (`photo-labelling/`, `src/model/labels/`). Their label table is `artifacts/labels/photo_labels.csv`. Feature families are compared under one estimator and one set of folds.

Chinese version: [README.zh-CN.md](README.zh-CN.md).

---

## Results at a glance

### Feature pipeline

| Stage | Measured |
|---|---|
| Download 1,558 images from Hugging Face | 183.8 MB in **115 s** (16 threads, resumable) |
| Extract features (MobileNetV3-Small @224, frozen) | **36–38 s ≈ 41–43 img/s**, peak RSS **≈670 MB** |
| Feature quality (same-cat vs different-cat cosine) | 0.803 vs 0.692, gap **+0.111**, AUC **0.844**, 5-NN cat retrieval **27.7 %** (chance 0.38 %) |

### Which feature family helps predict preferences

Pair-grouped 5-fold over all 24,896 rows, one estimator (L2 logistic regression + side column, `C = 1`). Details and per-observer metrics: [`docs/feature_choices.md`](docs/feature_choices.md).

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

What shaped the code:

1. **Annotator demographics do not predict preferences** (0.6900, worse than a constant).
2. **Seen cat → memory wins; unseen cat → appearance wins.** A single joint hybrid is best overall (0.6565) but loses on brand-new cats (0.6945).
3. **Coarse labels calibrate best (ECE 0.007) and transfer best to new cats** (0.6805). The dense embedding carries most of the accuracy on seen cats.
4. **More features do not fix the corner.** Concatenating a second backbone raises the average (0.6659 → 0.6607) and *worsens* unseen cats (0.6874 → 0.6938). Routing to a transfer model keeps the average at 0.6565 and puts both cold slices (0.6823 / 0.6805) below the family medians (0.6837 / 0.6892).

---

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and `make`. Course CSVs are already in `course/`. Images are not shipped.

The CPU-only `.venv` (Python 3.14) works on **macOS arm64** and **Linux x86_64 / arm64**. Intel Mac is not supported (PyTorch 2.14.1 has no x86_64 macOS wheels).

```bash
cd dsai5101midterm
make all          # env + images + YOLO/CLIP labels + ImageNet features + models
```

`make help` lists the stage targets. `make all` runs YOLO+CLIP when `artifacts/labels/photo_labels.csv` is missing. `make labels` forces a re-label.

To use `python3` in your shell, `source scripts/00_setup_env.sh` (creates `.venv` if needed and activates it).

| Target | What it runs |
|---|---|
| `make env` | `scripts/00_setup_env.sh` — CPU torch (Linux CPU index / macOS PyPI) |
| `make images` | download 1,558 photos (~184 MB, ~2 min) |
| `make labels` | YOLO11n-seg + CLIP → `artifacts/labels/` (slow on CPU) |
| `make features` | frozen ImageNet: MobileNetV3-Small + ResNet18 |
| `make qa` | same-cat vs different-cat diagnostics |
| `make models` | baseline, feature families, cold-start routing, routed hybrid |
| `make tables` | `tables/{train,test}_pairs.csv.gz` |
| `make quickstart` | one photo → 576-d + 20-d vector |
| `make distclean` | wipe `.venv`, images, labels, features, tables, weight caches |

Override load: `make all WORKERS=2 THREADS=4`. Stages are incremental: Make skips work whose outputs already exist.

That writes `artifacts/features/mobilenet_v3_small_224.npz` (1,558 × 576 deep + 1,558 × 20 hand) and `artifacts/submissions/pred_routed_test.csv` (6,224 rows).

Same pipeline without Make:

```bash
source scripts/00_setup_env.sh                            # create .venv if needed and activate it
python3 scripts/01_download_images.py
python3 scripts/02_build_index.py
python3 photo-labelling/label_photos.py
python3 scripts/03_extract_features.py --workers 4 --threads 8
python3 scripts/03_extract_features.py --backbone resnet18
python3 scripts/04_feature_qa.py --tag mobilenet_v3_small_224
python3 scripts/05_baseline_pairs.py --tag mobilenet_v3_small_224
python3 scripts/07_feature_families.py --tag mobilenet_v3_small_224
python3 scripts/08_cold_start_fix.py
python3 scripts/09_hybrid_submission.py --structure routed
python3 scripts/06_export_tables.py
python3 scripts/10_quickstart.py --image artifacts/images/cat_000002/cat_000002_0001.jpg
```

---

## Layout

```
dsai5101midterm/
├── Makefile                  make all / make help
├── README.md / README.zh-CN.md
├── requirements.txt          pinned versions; setup script picks the CPU torch wheel
├── course/                   course CSVs + brief (no images)
├── src/model/                extraction, pair design, hybrid model
│   └── labels/               cat strength + photo-label encoders
├── scripts/                  00 setup → 10 quickstart
├── photo-labelling/          YOLO11n-seg + CLIP → artifacts/labels/
├── docs/                     pipeline design + feature-family writeup
├── artifacts/                generated, git-ignored (teaching/research use only)
│   ├── images/<cat>/<photo>.jpg
│   ├── labels/               photo_labels.csv, cat_labels.csv
│   ├── features/<tag>.npz    paths, deep (N,576), hand (N,20), ok
│   └── submissions/          pred_test.csv, pred_hybrid_test.csv, pred_routed_test.csv
└── tables/                   ready-to-use csv.gz, git-ignored (see tables/README.md)
```

## Scripts

| Script | Key flags | Purpose |
|---|---|---|
| `00_setup_env.sh` | `PYTHON_TAG` | build `.venv` from `requirements.txt`; CPU torch avoids ~2.5 GB of nvidia CUDA wheels on Linux. `source` it to create (if needed) and activate; `make env` always recreates. |
| `photo-labelling/label_photos.py` | `--fresh`, `--limit`, `--batch` | YOLO11n-seg + CLIP labels → `artifacts/labels/`. Invoked by `make labels`. |
| `01_download_images.py` | `--workers`, `--limit` | 16 threads, 3 retries with backoff, atomic `.part` writes, skip-if-present |
| `02_build_index.py` | — | `image_index.csv`, train/test overlap and missing-file checks |
| `03_extract_features.py` | `--backbone`, `--size`, `--batch-size`, `--workers`, `--threads`, `--tta-flip`, `--limit`, `--tag`, `--write-csv`, `--list-backbones` | extract, cache, report throughput |
| `04_feature_qa.py` | `--tag`, `--no-plots` | same/different-cat similarity, 5-NN retrieval, PCA, neighbour grid |
| `05_baseline_pairs.py` | `--tag`, `--C`, `--p2-folds`, `--seed` | pair-grouped CV + image-holdout, submission |
| `06_export_tables.py` | `--tag`, `--out-dir`, `--no-images` | write `tables/{train,test}_pairs.csv.gz` + `image_features.csv.gz` |
| `07_feature_families.py` | `--folds`, `--p2-folds`, `--seed`, `--no-coldstart`, `--replot` | ten feature families, three protocols, per-observer and calibration metrics |
| `08_cold_start_fix.py` | `--reuse` | multi-backbone features and routing against the cold-start corner |
| `09_hybrid_submission.py` | `--structure routed\|joint`, `--spec`, `--C`, `--sweep` | fit the final model, write predictions |
| `10_quickstart.py` | `--image`, `--batch`, `--backbone`, `--tag` | one photo → 576-d + 20-d vector, cache cross-check |

Backbones (identical interface, one flag):

| Backbone | Dim | Params | Throughput (128-image probe) | Peak RSS |
|---|---|---:|---:|---:|
| `mobilenet_v3_small` | 576 | 2.5 M | **24.2 img/s** | 654 MB |
| `resnet18` | 512 | 11.7 M | 15.2 img/s | 766 MB |
| `efficientnet_b0` | 1280 | 5.3 M | 17.3 img/s | 1.08 GB |

## Feature cache format

`artifacts/features/<tag>.npz` (`np.load`, no pickle):

| Key | Shape | Meaning |
|---|---|---|
| `paths` | (N,) str | path relative to `artifacts/images/`, byte-identical to `a_image` / `b_image` |
| `deep` | (N, 576) f32 | L2-normalised frozen-CNN pooled embedding |
| `hand` | (N, 20) f32 | interpretable low-level statistics (`hand_features` in the meta) |
| `ok` | (N,) bool | decoded successfully (1,558/1,558 here) |
| `hand_names` | (20,) str | column names |

`<tag>.meta.json` records backbone, weights, input size, normalisation, torch/torchvision versions, threads, batch size, wall time, img/s, peak RSS, CPU model, seed, and timestamp.

Dimensions: **596 per photo** (576 deep + 20 hand). A pair row is **578** (`f_A − f_B` + `comparison_type`), **604** with the 26 demographic one-hots, or **1180** with `diff_absdiff`.

## Ready-to-use tables (`tables/`)

`scripts/06_export_tables.py` writes one row per annotator response with image features, observer profile, and the observed choice already joined, so a model can be trained without this repository's code:

| File | Shape | Contents |
|---|---|---|
| `train_pairs.csv.gz` | 24,896 × 636 | `d_0000…d_0575` (embedding A−B) + `h_*` (20) + 6 raw profile columns + 26 profile one-hots + `choice`, `y` |
| `test_pairs.csv.gz` | 6,224 × 634 | same columns without labels |
| `image_features.csv.gz` | 1,558 × 598 | per-photo table, for reconstructing per-side vectors |

Verified: `LogisticRegression(C=0.003)` trained from `train_pairs.csv.gz` reproduces the pipeline's **0.665** log loss. Column reference: [tables/README.md](tables/README.md).

## Using the features

```python
import sys; sys.path.insert(0, "src")
from model import features, index

feats = features.load_features("mobilenet_v3_small_224")   # None = most recent cache
X, present_a, present_b = features.pair_matrix(df, feats, mode="diff", with_demo=True)
y = features.label_a(df)        # 1 when side A was chosen
# X: (n_rows, 604) = 576 embedding difference + 26 demographics + 2 comparison types
```

Image/feature correspondence is by **path string**, not row order. `lookup()` returns a zero row plus `present=False` for anything missing (0 misses across all 31,120 train/test photo positions). Pair design matrices are computed on the fly (~0.1 s) and never cached, so changing the encoding does not require a re-export.

## Hybrid model

`model.hybrid.HybridModel` is one L2 logistic regression over difference-coded blocks:

```
[ cat one-hot (cats seen in the fit) | 576-d embedding | 20 hand (z-scored) | labels | side ]
```

- Cat memory comes from `model.labels.strength`; an unseen cat contributes 0.
- Appearance blocks carry the signal for unseen photos and for within-cat comparisons (where the cat gap is 0).
- Labels are optional and read from `artifacts/labels/photo_labels.csv` (written by `photo-labelling/label_photos.py`).
- `scripts/09` fits `cat + embedding + hand` at `C = 0.3` on all 24,896 rows (834 columns) and writes `artifacts/submissions/pred_hybrid_test.csv`.

The joint model is best when the cat is known (0.6565 overall) but **worse than the embedding alone for entirely new cats** (0.6945 vs 0.6869), because the cat dummies absorb cat-level variance. A second backbone does not help (0.6938 on new cats). Routing does:

- joint model when both cats **and** both photos were present in training (50 % of test rows);
- a fur-colour-only transfer model otherwise.

Average unchanged (0.6565); both photos unseen 0.6823; both cats unseen 0.6805 — below the family medians 0.6837 / 0.6892. `scripts/09_hybrid_submission.py --structure routed` writes `artifacts/submissions/pred_routed_test.csv`. See [`docs/feature_choices.md`](docs/feature_choices.md) §6b.

## Cat strength

How much of a choice is a stable preference for the cat, how much is a preference for side A, and how much is disagreement that does not repeat? Two photos of the same cat are a smaller question. Cat identity is the part that replicates. Side A has a small, precise advantage. A second photo of the same cat does not have a reproducible winner. Field definitions: [course/README.md](course/README.md).

There is no respondent id. The six background fields form 69 complete profiles. 508 training rows have all six blank together; they stay in the fit (rate of choosing A is 0.604, against 0.517 on complete rows) and are omitted from background breakdowns only. Every Cramér's V between a background field and `choice` is under 0.05, so demographics are not in the strength model. On the 8,750 between-cat ordered pairs answered twice, both responses match 0.561 of the time. That share rises from 0.512 to 0.647 as the two cats' win rates move apart.

| Subset | Rows | What a strength refers to |
|---|---:|---|
| `between_cats` | 21,893 | the cat |
| `within_cat` | 3,003 | the photo, in the comparison fit only |
| Test | 6,224 | scored with the rule below |

237 cats appear 170–245 times (median 212). 1,296 photos appear 7–73 times (median 39).

For a between-cat row, `P(choose A) = logistic(strength(cat A) − strength(cat B) + side)`. Strengths are L2-penalized logistic regression (`sklearn`, `C = 1`, no extra intercept), then centered, so an average cat has strength 0 and a cat absent from the fit has strength 0. `side` is the log-odds of choosing A when the two cats are equal. On the held-out path, `C` of 0.1, 1, 10, and 100 all sit within 0.002 log loss. Seed is 21. The runnable fit is `src/model/labels/strength.py`. Pair folds keep both side orders of a comparison together.

Five-fold pair holdout:

| Model | Log loss | Brier | Accuracy |
|---|---:|---:|---:|
| Between, side only | 0.6924 | 0.2496 | 0.520 |
| Between, photo strength + side, C = 1 | 0.6675 | 0.2367 | 0.603 |
| Between, cat strength + side, C = 1 | 0.6552 | 0.2316 | 0.608 |
| Within, side only | 0.6933 | 0.2501 | 0.514 |
| Within, photo strength + side, C = 0.1 | 0.6901 | 0.2485 | 0.536 |
| Within, photo strength + side, C = 1 | 0.7025 | 0.2538 | 0.543 |

Cat strength beats the side-only rate. Photo strength beats it by less, and on within-cat rows the best penalty on the grid (`C = 0.1`) gains only 0.003 log loss. Weaker penalties overfit, so within-cat test rows get the training rate of choosing A (0.514) and no photo strength. Full between-cat fit: side effect 0.088 log-odds (P(choose A) = 0.522 when the cats are equal; pair-bootstrap 95% interval 0.516–0.529). Split-half correlation of side-adjusted strength is about 0.69 for cats and about 0.15 for within-cat photos. Cohen's kappa on a repeated ordered pair is 0.11 overall and 0.05 within a cat.

Cat holdout, mean of five repeats (a quarter of the cats removed, then a fifth of the remaining pairs held out):

| Slice | Mean rows | Model log loss | Side-only log loss |
|---|---:|---:|---:|
| Both cats seen, pair held out | 2,467 | 0.662 | 0.692 |
| One cat unseen | 8,250 | 0.685 | 0.693 |
| Both cats unseen | 1,322 | 0.694 | 0.694 |

The gain is about 0.030 when both cats were in the fit, about 0.007 when one was not, and zero when neither was. Fur-colour shares still predict in that last case: log loss 0.6813 on the same pair folds, about 0.013 better than the side rate when both cats are unseen. Pose, eyes, breed, and shot size gain 0.002 or less. Adding fur colour on top of cat strength changes log loss by 0.0002, because the coat is already inside the cat's strength. That is why the routed hybrid keeps cat memory for seen cats and fur colour for the rest.

On the strength-only test file, `choice` is A when `prob_a ≥ 0.5`: 4,317 A and 1,907 B. Both-cats-unseen and within-cat slices are constant probabilities above one half, so every one of those rows is labelled A. The current submission is `artifacts/submissions/pred_routed_test.csv`.

## Photo labels

`photo-labelling/label_photos.py` labels each photo. Fur colour and pattern come from the pixels inside the YOLO11n-seg mask. Breed, framing, view, eyes, age, and posture are CLIP labels on the crop. Weights go in `models/yolo11n-seg.pt` (downloaded on first run if missing, from [yolo11n-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt)). The script writes `artifacts/labels/photo_labels.csv` and `artifacts/labels/cat_labels.csv`, and captioned copies under `artifacts/captioned-photos/`.

| Path | Role |
|---|---|
| `photo-labelling/` | YOLO11n-seg masks and CLIP labels |
| `src/model/labels/strength.py` | per-item strengths (`fit_bt` / `predict_bt`) |
| `src/model/labels/labels.py` | reads the label table; encoders match `numeric_diff` / `one_hot_diff` / `with_side` |
| `src/model/labels/metrics.py` | `log_loss` / `brier` / `accuracy` |
| `requirements.txt` | feature pipeline plus ultralytics, open-clip, and opencv |

```bash
python3 photo-labelling/label_photos.py   # venv active; images must already be downloaded
```

## Reproducibility

- Fixed: path ordering, `torch.manual_seed(0)`, fold seeds (`0` for baseline and feature-family runs), `C` selected on a pair-grouped holdout where reported.
- Versions are pinned in `requirements.txt`. Each extraction records the actual environment in `<tag>.meta.json`.
- Known numerical variation: torch thread count changes floating-point summation order (4th decimal of log loss). Everything else reproduces bit-for-bit (no random augmentation, no dropout, no GPU non-determinism). Re-running `05` reproduces `baseline_metrics.json` exactly; re-running `07` reproduces `feature_families.json`.
- Sandboxes that forbid POSIX semaphores make DataLoader workers unavailable; the extractor falls back to `num_workers=0` automatically and records `num_workers_requested` vs `num_workers` in the meta.

## Limitations

- Human preference is noisy: two observers agree on 55.7 % of repeated pairs. Any model is capped well below certainty, and on disputed pairs every family sits at 0.50 accuracy.
- There are no annotator IDs, only 70 demographic profiles, so "different observers" is an approximation — which is also why demographics turned out useless.
- The two cold starts (new photos vs new cats) give opposite rankings. Report both.
- Test labels are withheld, so none of these numbers is a leaderboard score.
- Low-level hand features add little accuracy; their value is interpretability.

## What is not in git

See `.gitignore`. The 183 MB of photos, feature caches, 78 MB `tables/*.csv.gz`, `.venv`, and the course name list are regenerable:
`git clone` → `make all`.

To share the large tables:

| Option | How |
|---|---|
| GitHub Release asset | upload `tables/*.csv.gz` (each file is under 100 MB) |
| Git LFS | `git lfs track "tables/*.csv.gz"` then add and commit |
| Course package | zip `tables/` together with the code for submission |