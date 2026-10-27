# Design note: extracting simple features from the cat photos

> Why each step in `src/catarena/` is built the way it is. Numbers are measured on this machine;
> run instructions live in the `README.md`.

## 1. Task and hard constraints

The work splits into two stages:

1. **Now** — turn 1,558 photos into fixed-length vectors (the feature layer).
2. **Later** — use those vectors to predict whether an annotator prefers photo A or B.

Constraints, all measured rather than assumed:

| Constraint | Measurement |
|---|---|
| No GPU | `torch.cuda.is_available() == False`, no `/dev/dri` |
| Little memory | 7 GB total, ~4 GB free |
| Small data | 1,558 photos / 284 cats; 1,296 photos and 24,896 responses in train |
| Images are external | 1080×1440 JPEG, 183.8 MB, ~1.6 MB/s from Hugging Face with 16 connections |
| Test is partly cold start | 262 of the 1,389 test photos never appear in train; **25 % of test rows have two unseen photos**; train/test share **0** image pairs |

The last two constraints decide the architecture: **do not train end-to-end**; freeze a generic
vision backbone, bake every photo into a vector once, and keep all modelling on top of those
vectors.

## 2. Why the feature layer is decoupled from the preference layer

| Option | CPU cost | Overfit risk | Cold start | Iteration speed |
|---|---|---|---|---|
| A. End-to-end (pixels → preference) | high (full pass per epoch) | high (1,296 photos vs millions of parameters) | poor (new photos need re-running) | slow |
| B. **Frozen backbone + cached features** (chosen) | one-off 36 s | low (fit only 576 weights) | **good**: a new photo costs one forward pass | seconds |
| C. Handcrafted features only | negligible | low | good | instant |

B wins because the photos are only ever forwarded once, after which every modelling experiment
(different regularisation, interactions, annotator strata) runs on the cache. C survives as a
20-dimension interpretable add-on.

## 3. Data preparation

1. **The image list comes from the CSVs**, not from scraping: `set(a_image) | set(b_image)`
   deduplicates to exactly 1,558 paths.
2. **Download**: 16 threads, three retries with exponential backoff, `.part` file plus
   `os.replace` for atomic writes, skip when the existing file has the expected size
   (resume). Measured 115 s / 183.8 MB.
3. **Index**: `image_index.csv` records each photo's cat, which split(s) it appears in, and how
   often. `02_build_index.py` prints the train/test overlap — the first leakage check.
4. **Paths are relative throughout** (`cat_xxxxxx/cat_xxxxxx_0001.jpg`), so caches and code can
   be moved without rewriting anything.

## 4. Preprocessing: why these five operations

```
JPEG → exif_transpose → convert("RGB") → Resize(short side 232, bicubic)
     → CenterCrop(224) → ToTensor → ImageNet normalisation
```

| Decision | Reason |
|---|---|
| `exif_transpose` | phone photos carry an orientation flag; ignoring it rotates a subset of images |
| `convert("RGB")` | sources may be greyscale or CMYK; the pretrained weights need 3 channels |
| 232 → centre crop 224 | 224 is the resolution the ImageNet weights were trained at; the extra 8 px avoids blurring the border during resize |
| **no random augmentation** | the cache must be reproducible; augmentation only helps when training, and the backbone is frozen |
| ImageNet mean/std | matches the weights, otherwise batch-norm statistics and activations drift |
| centre crop instead of whole-image resize | cats sit near the centre; cropping keeps native detail (fur texture, sharpness) instead of shrinking 1080×1440 into a 224×224 frame |

Cost: the crop discards the frame edges, which matters little for "which cat photo looks
better" and is a one-parameter change in `preprocess.build_transform`.

## 5. Feature design

### 5.1 Deep features (the main block)

* **MobileNetV3-Small**, ImageNet-pretrained, global average pooling before the classifier:
  **576 dims**. 2.5 M parameters, ~0.06 GFLOPs at 224 px.
* Measured **43 img/s on 8 threads**: 1,558 photos in 36–38 s, peak RSS 661–678 MB.
* Why not a bigger backbone: the bottleneck here is domain shift (ImageNet object classes vs
  human aesthetic preference), not capacity. Measured cost of the alternatives: `resnet18`
  runs at ~0.63× the throughput, `efficientnet_b0` peaks at 1.08 GB. Whether they improve
  accuracy was **not** verified in this project — this is a design judgement, not a measured
  claim.
* Outputs are **L2-normalised** so cosine similarity equals the inner product, which keeps the
  linear model's scale stable and makes the same-cat/different-cat diagnostic unit-free.

### 5.2 Interpretable low-level statistics (20 dims, nearly free)

Computed on a 256 px thumbnail of the **original** image (not the 224 crop), so file-level
properties such as aspect ratio survive: brightness, contrast, saturation mean/std,
Hasler–Süsstrunk colourfulness, Laplacian variance (sharpness), greyscale entropy, edge density,
warm-pixel ratio, circular hue mean and concentration, per-channel means, aspect ratio,
log pixel count, over/under-exposed fractions, greyscale flag, log file size.

Two purposes: (a) interpretable dimensions for the report; (b) a test of whether preference is
driven by obvious properties such as brightness or sharpness.

### 5.3 Dimensions

| Object | Dimensions | Notes |
|---|---|---|
| One photo | **596** | 576 deep + 20 hand; 3.7 MB as float32 for 1,558 photos |
| Pair row, `diff` | **578** | `f_A − f_B` (576) + `comparison_type` (2) |
| Pair row, `diff` + demographics | **604** | + 26 one-hot columns (gender 3, pet_history 4, pet_care 5, pet_contact 6, future_pet 4, mbti_ie 4, each with a `missing` level) |
| Pair row, `diff_absdiff` | 1180 | adds the absolute difference |
| `deep_hand` variant | 624 | difference of the 596-dim photo vector + 2 |

Even the widest configuration is far below the 7 GB limit, so **no PCA is needed**; PCA is used
only for the 2-D diagnostic plot.

## 6. Cache and interface (what makes reuse cheap)

`data/features/<tag>.npz` has five keys: `paths / deep / hand / ok / hand_names`.
`paths` matches `a_image` and `b_image` **byte for byte**, so downstream code needs no mapping
table. Correspondence is by **path string, not row index**; a failed decode yields
`ok=False`, and `lookup()` returns a zero row with `present=False` so the caller can fall back
explicitly instead of silently shifting rows. Measured: 0 missing across all 31,120
train/test photo positions.

`<tag>.meta.json` records backbone, weight file, input size, normalisation constants,
torch/torchvision versions, threads, batch size, wall time, img/s, peak RSS, CPU model, seed
and timestamp — any feature artefact can prove its own provenance.

`extract.FEATURE_SPEC_VERSION` invalidates caches when preprocessing changes, so a stale cache
cannot be used by accident.

Pair design matrices are **derived on the fly** (`features.pair_matrix`, ~0.1 s) and never
written to disk: changing the encoding requires no re-export.

## 7. CPU engineering

| Problem | Solution |
|---|---|
| Thread oversubscription | `torch.set_num_threads(8)` in the main process, `set_num_threads(1)` inside DataLoader workers |
| Autograd overhead | all weights `requires_grad_(False)`, `model.eval()`, `torch.inference_mode()` |
| Memory | forward batch by batch and write straight into pre-allocated arrays; never hold the decoded images (1,558 × 1080 × 1440 × 3 ≈ 7 GB) |
| Corrupt images | `try/except` inside the dataset, zero row + `ok=False`, batch never aborts |
| Sandbox limits | if POSIX semaphores are unavailable, fall back to `num_workers=0` and record `num_workers_requested` vs `num_workers` in the meta |
| Cache locations | `TORCH_HOME` / `MPLCONFIGDIR` point inside the project so a read-only `~/.cache` cannot break weight or font downloads |
| Throughput reporting | report `steady_img_per_sec` (excluding weight loading and the first batch) next to the wall time |

## 8. Feature quality checks (prove the features before modelling)

`qa.py` computes three complementary diagnostics:

1. **Same-cat vs different-cat cosine similarity**: 0.803 vs 0.692 (gap +0.111), pairwise
   AUC **0.844**. The embedding does encode "what this cat looks like".
2. **5-NN same-cat retrieval**: **27.7 %** against a 0.38 % chance level (sum of squared class
   frequencies). It looks low in absolute terms — 284 cats, ~5.5 photos each, and ImageNet
   features lean towards pose/background rather than identity — but it is ~70× chance and is
   enough to give the preference model a transferable signal.
3. **PCA and nearest-neighbour grids**: human-checkable. Black-and-white, grey and tabby cats
   cluster together, which makes a good report figure.

Running this before modelling means a failure points at the extractor (backbone, size), not at
the classifier.

## 9. Downstream model: why `f_A − f_B`

The key choice is to use **no image identity**. 262 test photos never appear in train and no
test pair appears in train, so memorising either is useless.

A linear model on the difference vector is exactly Bradley–Terry / Thurstone pairwise comparison:

```
score(photo) = w · f(photo)
P(annotator chooses A) = σ( w·(f_A − f_B) + β·annotator )
```

It is antisymmetric by construction (swapping A/B flips the prediction), generalises to unseen
photos (as long as `w` captures "what makes a cat photo appealing"), and has only 576
parameters, which 1,296 photos can support.

`features.pair_matrix` also supports `diff_absdiff` and demographic one-hot columns for later
extensions.

## 10. Leakage: two protocols, both needed

| Protocol | Split unit | Question answered | Baseline result |
|---|---|---|---|
| P1 pair-grouped 5-fold | unordered image pair | new people / new pairs, same photos | logloss 0.6651 / acc 0.593 |
| P2 image-holdout 3-fold | photo (the whole photo disappears from the fit) | **both photos unseen** (matches 25 % of test rows) | logloss 0.6863 / acc 0.565 |

In P2 the evaluation set keeps only rows where **both** photos are held out, the fit set only
rows where **neither** is; straddling rows are dropped from both (clean but conservative).

P2 is mandatory here: 25 % of test rows have two unseen photos, so reporting P1 alone
systematically overstates the expected score (0.665 vs 0.686). This is exactly the "attention
to uncertainty and leakage" criterion in the brief.

A third protocol — holding out whole **cats** — is implemented in
`scripts/07_feature_families.py`. It is the harder cold start: with both cats unseen no item
memory can help, so only a shared appearance model can beat the baseline. The two cold starts
give opposite rankings, so `docs/feature_choices.md` reports both.

## 11. What the numbers say

* Demographics alone barely help: logloss 0.6899 vs the 0.6920 constant baseline. Annotator
  background is a weak predictor of which photo is chosen, consistent with `choice` being close
  to 50/50 and side bias being tiny.
* Image features help: under P2, 0.6920 → 0.6863 (accuracy 0.525 → 0.565). The gain is small
  but it is obtained on **photos that were never seen**, so it is transfer, not memorisation.
* Within-cat comparisons (two photos of the same cat) are the hard slice: 0.707 under
  P2 (chance) and 0.6836 under pair-grouped CV. `between_cats` is where appearance features can
  work.
* Feature families were compared head to head in `scripts/07_feature_families.py`: a 576-d frozen
  embedding reaches 0.6674, per-cat memory 0.6600, coarse semantic labels 0.6800, hand statistics
  0.6865, annotator demographics 0.6900 (worse than a constant). Per-observer and calibration
  metrics are in `docs/feature_choices.md`.
* Hand features add little accuracy (0.6876 vs 0.6891 under P2) but provide interpretability.

## 12. Failure modes handled

| Case | Handling |
|---|---|
| Download failure / missing photo | three retries with backoff; failures go to `download_errors.csv`; `02` re-checks files on disk |
| Corrupt JPEG / non-RGB / EXIF rotation | `try/except` + `exif_transpose` + `convert("RGB")`; failed photo gets `ok=False` and a zero row |
| DataLoader multiprocessing blocked | automatic `num_workers=0`, recorded in the meta |
| Memory pressure | batch-wise processing, peak 661–678 MB; `--batch-size` tunable |
| Read-only `~/.cache` | torch/matplotlib caches redirected into the project |
| Preprocessing version change | `FEATURE_SPEC_VERSION` + meta fields |
| Submission format | `05` / `09` assert column order, 6,224 rows, row-id order, probability range, and `choice` ↔ `prob_a ≥ 0.5` consistency |

## 13. Extension paths, cheapest first

1. **Multi-backbone concatenation** (mnv3s 576 + resnet18 512): extraction ~2 minutes, downstream free.
2. **Flip TTA** (`--tta-flip`): 2× cost; for a left/right-symmetric preference task usually noise-level.
3. **Routing instead of one joint model** — implemented (`scripts/09_hybrid_submission.py
   --structure routed`). Use the joint cat+appearance model when both cats and both photos were
   seen in training, and a fur-colour-only transfer model otherwise. Average stays at 0.6565
   while the two cold slices improve from 0.6847/0.6945 to **0.6823/0.6805** (family medians
   0.6837/0.6892). Enlarging the feature set does **not** achieve this: a ResNet18 concatenation
   raises the average (0.6659 → 0.6607) but makes new cats worse (0.6874 → 0.6938), because the
   extra capacity encodes identity rather than transferable aesthetics.
4. **Stratified models**: model `within_cat` and `between_cats` separately — their difficulty
   differs by a factor of ~4.
5. **CLIP / SigLIP backbone**: semantically closer to human aesthetics; ~10–20 min on CPU and
   1–2 GB RSS for 1,558 photos. A sensible "bigger, more semantic backbone" comparison, not a
   default.
6. **Light fine-tuning**: unfreeze 1–2 blocks; needs early stopping and strong regularisation on
   1,296 photos, ~10–20 min/epoch on CPU, uncertain payoff given label noise.

## 14. Reproducibility checklist (matches the brief)

- [x] Environment file: `requirements.txt` (pinned) + `scripts/00_setup_env.sh`
- [x] Seeds: `torch.manual_seed(0)`, KFold `random_state=0`, path dictionary order
- [x] Hardware and runtime: README table + per-tag `<tag>.meta.json`
- [x] External resources and pretrained weights: `meta.backbone_weights` (e.g. `mobilenet_v3_small-047dcff4.pth`)
- [x] Relative paths everywhere; data lives in `Dataset/` and `data/images/`
- [x] Numerical variation documented: thread count changes summation order (4th decimal of
      log loss); everything else is bit-reproducible
