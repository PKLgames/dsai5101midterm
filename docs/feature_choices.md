# Which image features actually help predict different observers?

The end goal is not to describe a cat photo but to predict **whether a given observer will
choose photo A or photo B**. A feature family is therefore judged on preference metrics, not on
how detailed its description looks.

Everything below is measured on this dataset with one estimator family (L2 logistic regression,
`fit_intercept=False`, explicit side column, `C = 1`) and one set of pair-grouped folds
(`scripts/07_feature_families.py`).

## 1. The metrics, and why these ones

| Metric | What it answers | Why it matters here |
|---|---|---|
| log loss / Brier | probability quality | the submission is a probability (`prob_a`), not just a sign |
| accuracy | how often the argmax is right | easy to read, but a constant predictor already gets 0.52 |
| **per-profile gain** | does the family help *each* of the 70 observer profiles over that profile's own constant predictor? | features that only work for some observer types are not a good general feature |
| **agreement on repeated pairs** (0.557) and **accuracy on agreed pairs** | how much is stable across two different people seeing the same pair | an upper bound on what any pair-level model can learn |
| **calibration** (ECE, recalibration slope) | are the probabilities usable as probabilities? | downstream decisions and any probability blend need this |
| **cold start 1 / 2** | both photos unseen / both cats unseen | 25 % of test rows have two unseen photos; ~40 % involve an unseen cat |

Note on the per-profile metric: each of the 70 profiles saw a different subset of pairs, so part
of the spread is "which pairs" rather than "which observer". The gain column (model log loss
minus that profile's constant-predictor log loss) removes the part that comes from a profile
being close to 50/50, so it is the fairer of the two.

## 2. The families

| Family | What it is | What it can express | What it cannot |
|---|---|---|---|
| observer only | 26 demographic one-hot columns from the 6 background fields | "men like X" style population effects | anything about the photos |
| labels (colour) | fur-colour shares (YOLO mask + pixel colours) | coat colour composition | pose, breed, composition, anything not labelled |
| labels (all) | colour shares + pattern/breed/pose/view/eyes/shot size | a coarse human vocabulary of the scene | fine texture, background, lighting, aesthetics |
| hand (20) | brightness, contrast, saturation, colourfulness, sharpness, entropy, edges, hue, exposure, size | photographic quality properties | *what* is in the photo |
| embedding (576) | frozen ImageNet MobileNetV3-Small pooled features | arbitrary appearance directions learned from 1.3 M photos | nothing is named; no per-cat memory |
| cat memory | one strength per cat (per-cat item effect) | "cat 000123 is generally preferred" | any cat that was not in the fit |
| hybrids | cat memory + embedding (+ hand / labels) in one fit | both of the above | still no help for unseen cats in the joint design |

## 3. Results

Pair-grouped 5-fold over all 24,896 rows (both `between_cats` and `within_cat` included, as in
the test set):

| feature family | log loss | Brier | accuracy | per-profile gain (mean ± sd) | profiles helped | ECE | recal. slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| observer only | 0.6900 | 0.2484 | 0.532 | −0.0063 ± 0.013 | 7 % | 0.017 | 0.88 |
| labels (colour) | 0.6828 | 0.2449 | 0.553 | +0.0012 ± 0.029 | 59 % | 0.007 | 0.99 |
| labels (all) | 0.6800 | 0.2435 | 0.560 | +0.0042 ± 0.032 | 67 % | 0.009 | 0.94 |
| hand | 0.6865 | 0.2467 | 0.543 | −0.0024 ± 0.028 | 56 % | 0.009 | 0.91 |
| embedding | 0.6674 | 0.2375 | 0.591 | +0.0169 ± 0.044 | 73 % | 0.006 | 0.96 |
| embedding + hand | 0.6658 | 0.2367 | 0.592 | +0.0187 ± 0.045 | 74 % | 0.010 | 0.94 |
| cat memory | 0.6600 | 0.2339 | 0.600 | +0.0253 ± 0.058 | 76 % | 0.015 | 0.88 |
| cat + embedding | 0.6570 | 0.2324 | 0.606 | +0.0277 ± 0.062 | 73 % | 0.019 | 0.84 |
| **cat + embedding + hand** | **0.6565** | **0.2322** | **0.606** | **+0.0282 ± 0.062** | 73 % | 0.018 | 0.84 |
| cat + embedding + labels | 0.6566 | 0.2323 | 0.607 | +0.0281 ± 0.062 | 73 % | 0.016 | 0.84 |

Cross-observer consistency (9,733 ordered pairs answered twice by different people; the two
answers agree 55.7 % of the time):

| feature family | accuracy on agreed pairs | accuracy on disputed pairs |
|---|---:|---:|
| observer only | 0.544 | 0.526 |
| labels (colour) | 0.599 | 0.487 |
| labels (all) | 0.617 | 0.495 |
| hand | 0.580 | 0.500 |
| embedding | 0.673 | 0.504 |
| embedding + hand | 0.675 | 0.503 |
| cat memory | 0.690 | 0.505 |
| cat + embedding | 0.696 | 0.502 |
| cat + embedding + hand | 0.699 | 0.502 |
| cat + embedding + labels | 0.699 | 0.507 |

Cold start — the two kinds are not the same thing:

| feature family | both photos unseen | both cats unseen |
|---|---:|---:|
| observer only | 0.6901 | 0.6893 |
| **labels (colour)** | 0.6823 | **0.6805** |
| labels (all) | 0.6833 | 0.6841 |
| hand | 0.6878 | 0.6892 |
| **embedding** | **0.6797** | 0.6869 |
| embedding + hand | 0.6786 | 0.6876 |
| cat memory | 0.6776 | 0.6913 |
| cat + embedding | 0.6819 | 0.6958 |
| cat + embedding + hand | 0.6847 | 0.6945 |
| cat + embedding + labels | 0.6818 | 0.6938 |

![feature families](../artifacts/features/feature_families.png)

## 4. Plain reading: strengths and weaknesses

**Observer demographics — a non-feature.** 0.6900 log loss, *worse* than the constant baseline
(0.6925 is the same number, and the profile-constant is better than the model for 93 % of
profiles). Every Cramér's V between a background field and `choice` is below 0.05. Annotator
background tells you almost nothing about which photo they will pick; do not spend a feature
budget here.

**Low-level statistics (hand, 20 dims) — cheap, interpretable, weak.** 0.6865 (−0.006 vs
constant). It helps a little for everyone (56 % of profiles) with small spread, but it cannot
distinguish content. Useful as a reported covariate ("brighter photos are chosen slightly more
often"), not as a predictor.

**Coarse semantic labels — weak prediction, excellent calibration, best transfer to new cats.**
Colour shares reach 0.6828; the full label block 0.6800. Two things stand out:
the probabilities are almost perfectly calibrated (ECE 0.007, slope 0.99), and on **both cats
unseen** this is the best family in the study (0.6805, ahead of the 576-d embedding's 0.6869).
That is the quantitative version of the argument that a coarse attribute can stand in for a
missing item score. The cost is resolution: on pairs where two observers agreed, labels pick the
agreed side only 60–62 % of the time, versus 67 % for the embedding and 70 % for the hybrid.

**Dense frozen embedding — the best purely visual description.** 0.6674, −0.025 vs constant,
73 % of profiles helped, best calibration among the strong models (ECE 0.006). It transfers to
new photos (0.6797) and still beats the baseline for entirely new cats (0.6869). Weaknesses:
it is uninterpretable (576 unnamed dimensions), and it *is* mostly cat identity — its advantage
shrinks from 0.025 to 0.006 once the cat is new.

**Per-cat memory — the strongest seen-cat model, and the most fragile.** 0.6600 overall, the
best accuracy on pairs where observers agreed (0.690), and it is the only family that is clearly
better than everything else once two observers agree. But it is a lookup table: with both cats
unseen it collapses to the baseline (0.6913), and it is the least uniform across observer
profiles (per-profile gain sd 0.058) and least calibrated (slope 0.88).

**Hybrids — best average, worst cold start.** `cat + embedding + hand` is the best model overall
(0.6565, 73 % of profiles helped, 0.699 on agreed pairs), but on entirely new cats it is
noticeably *worse* than either part alone (0.6945 vs 0.6869 for the embedding, 0.6913 for the
memory). Once cat dummies absorb cat-level variance during fitting, the embedding weights only
explain the residual and transfer worse. This is a design flaw of a single joint fit, not of
embeddings: route by "is the cat known?" instead.

## 5. What to use for what

| Goal | Best choice | Reason from the numbers |
|---|---|---|
| best expected probability score on the given test set | hybrid (cat + embedding + hand) | 0.6565 overall; best on agreed pairs |
| predictions that must survive unseen cats | colour labels or the embedding alone | 0.6805 / 0.6869 on both-cats-unseen vs 0.6945 for the hybrid |
| interpretable story for the report | colour labels + hand statistics | nearly calibrated (ECE 0.007) and named dimensions |
| smallest, fastest pipeline | hand only (no CNN) | 0.6865 for a few milliseconds per photo |
| anything involving observer background alone | none | worse than a constant predictor |

## 6. Caveats

* **No annotator IDs.** "Different observers" is approximated by the 70 complete demographic
  profiles (508 rows have all six fields blank). Profiles also differ in which pairs they saw,
  so the per-profile spread mixes observer difficulty with pair composition; the gain column
  mitigates this but does not remove it.
* **Agreement caps accuracy.** Two people agree on 55.7 % of repeated pairs (Cohen's kappa ≈ 0.11
  from the cat-strength fit). On disputed pairs every model sits at 0.50 — there is nothing
  stable to predict. Treat ~0.70 accuracy on agreed pairs as close to the ceiling.
* **Two cold starts, only one of which is usually reported.** Hiding the cat-holdout column makes
  item-memory models look better than they are.
* **The label table is read, not regenerated** (the YOLO11n-seg + CLIP table in `artifacts/labels/`), so the
  label rows inherit that pipeline's choices; the embedding and hand rows are produced here.
* One dataset, one seed, `C = 1` for every family. The differences between neighbouring families
  (0.001–0.003) are within noise; the differences between families (0.005–0.03) are not.
* Test labels are withheld, so none of this is a leaderboard score; the three protocols above are
  the honest proxies.

## 6b. Can better features close the gap? (No — routing does)

The hybrid is the best model on average but the worst in the cold-start corner. Two remedies
were measured (`scripts/08_cold_start_fix.py`); acceptance was *average ≤ 0.6565* **and**
*both-cats-unseen ≤ 0.6892* (the family median).

**Remedy 1 — richer features.** A second backbone (ResNet18) concatenated with MobileNetV3-Small
(1088-d) improves every seen-cat score, and makes the cold start *worse*:

| appearance features | avg | both photos unseen | both cats unseen |
|---|---:|---:|---:|
| emb + hand + colour (596 + 5) | 0.6659 | 0.6798 | 0.6874 |
| + ResNet18 concatenation (1113) | **0.6607** | 0.6801 | **0.6938** |
| joint model with the multi-backbone features | 0.6567 | 0.6897 | 0.6968 |
| joint model with single-backbone features | **0.6565** | 0.6847 | 0.6945 |

More capacity buys in-distribution accuracy and *costs* transferability: the extra dimensions
encode cat identity better, which is exactly what a new cat cannot use. Adding the fur-colour
shares to the appearance block changes the cold start by ~0.001 (noise).

**Remedy 2 — split the job instead of enlarging it.** Fit the joint model and a transfer model
(fur-colour shares only) separately, then choose at prediction time using information that is
known without labels — was this cat and this photo present in training?

| model / routing | avg | both photos unseen | both cats unseen | meets both |
|---|---:|---:|---:|:--:|
| joint (one model for everything) | 0.6565 | 0.6847 | 0.6945 | no |
| routed: joint if **cat+photo** known, else labels | **0.6565** | **0.6823** | **0.6805** | **yes** |
| routed: joint if **cat** known, else labels | 0.6565 | 0.6844 | 0.6805 | yes |
| family medians (the bar to clear) | — | 0.6837 | 0.6892 | — |

The routed model keeps the best average and beats both cold-start medians, by 0.0014 on unseen
photos and by 0.0087 on unseen cats. On the test set the joint branch is used for 50 % of rows.
The routing signal is "was this item in the training data", which is available at submission time
and uses no test labels.

**Take-away for feature design:** the weakness of the best-on-average model was not a feature
deficiency that more or better features could remove; it was the joint fit letting item memory
crowd out the shared representation. Better features made the average better and the corner
worse. The fix is to keep a dedicated transfer model and route to it.

## 7. Reproduce

```bash
cd dsai5101midterm
.venv/bin/python scripts/03_extract_features.py --backbone resnet18   # second backbone, ~1 min
.venv/bin/python scripts/07_feature_families.py                      # ~3.5 min
.venv/bin/python scripts/08_cold_start_fix.py                        # ~7 min
```

Outputs: `artifacts/features/feature_families.{json,md,png}`,
`artifacts/features/cold_start_fix.{json,md}` and the routed submission
`artifacts/submissions/pred_routed_test.csv`.
