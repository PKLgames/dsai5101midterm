# Held-out repeat agreement and pair-only oracle comparison

## What was measured

This is a **post-test, exploratory** analysis of already saved predictions and their saved outcomes; no model was fitted and no original result was replaced. G2 is the original common held-out evaluation. G3 and G4 are separate retrospective transfers with different splits and populations; their samples overlap and are not independent confirmatory replications. Historical outcome exposure and retrospective model development also limit confirmation.

For each held-out unordered **photo pair**, outcomes are oriented to the same photo before comparing repeats; otherwise a displayed-side swap could falsely look like disagreement. For an unordered **cat pair**, outcomes are oriented to the same cat, and within-cat comparisons are excluded because they cannot indicate which of two different cats was selected. Only groups with at least two held-out responses are used for repeat statistics. Models and reference estimates are evaluated on exactly the same repeated-response subset; full-scope model Brier scores are provided separately.

**Important cat-pair limitation:** within each of G2, G3 and G4, every observed between-cat pair corresponds to only one distinct unordered photo pair. Therefore these cat-pair results are a between-cat subset of photo-pair evidence, **not** evidence of consistency across multiple photo combinations for the same two cats. G4 contains only between-cat pairs, so its two descriptive point estimates are identical.

The independent unit is not known to be a person: these are response records, and repeated respondents may exist. Values do not identify individual consistency, preference causes, a causal information limit, or generalization to new people.

## Repeat agreement and the Brier reference

For a group with `n` responses and `k` choices of the canonical first photo/cat, the disagreement fraction over all unordered response-record pairs is `2k(n-k)/(n(n-1))`; agreement is one minus that quantity. We average group agreement weighted by `n`, corresponding to a randomly selected response in the repeated-group subset, then a different response from its group. Equal-group and unordered-response-pair-weighted variants are also provided in [agreement_summary.csv](agreement_summary.csv).

Under the **working assumption** that responses within a group are conditionally independent, identically distributed Bernoulli draws with a stable group probability `p`, half the disagreement estimates `p(1-p)`, the Brier risk of an oracle that knows that group's probability. We average it using the same response weights as model Brier. This pair-only reference is not directly observed and is not guaranteed to be a lower bound for models using additional information such as respondent context or displayed side. Dependence, nonexchangeable respondent composition, display effects, or selection into repeated groups can invalidate the oracle interpretation. In particular, repeat agreement does **not** by itself prove irreducible human randomness.

| Held-out cohort and level | Repeated responses / scope | Repeated groups | Median / max records per group | Response-weighted agreement | Estimated pair-only oracle Brier | 95% cluster-bootstrap interval |
|---|---:|---:|---:|---:|---:|---:|
| G2 photo pair | 3,681 / 3,733 | 1,155 | 3 / 4 | 56.15% | 0.21923 | [0.21026, 0.22781] |
| G2 different-cat pair | 3,237 / 3,278 | 996 | 3 / 4 | 56.26% | 0.21872 | [0.20956, 0.22787] |
| G3 photo pair | 2,201 / 2,250 | 669 | 3 / 4 | 55.44% | 0.22278 | [0.21191, 0.23355] |
| G3 different-cat pair | 1,927 / 1,964 | 575 | 3 / 4 | 55.34% | 0.22332 | [0.21118, 0.23507] |
| G4 photo pair | 3,428 / 3,479 | 1,050 | 3 / 4 | 56.48% | 0.21762 | [0.20879, 0.22657] |
| G4 different-cat pair | 3,428 / 3,479 | 1,050 | 3 / 4 | 56.48% | 0.21762 | [0.20848, 0.22637] |

Intervals are 3,000 percentile resamples of observed photo-pair or cat-pair clusters, conditional on the observed repeated-group subset. They do **not** account for shared respondents or crossed photo/cat dependence; for G3/G4 they also do not reproduce the original transfer-specific held-out-photo/cat bootstrap. They are descriptive working-model intervals, not a fully robust population uncertainty analysis. Pointwise intervals are unadjusted for multiple comparisons.

The tiny per-pair sample is consequential. On G2, **234 pairs have two**, **471 have three**, and **450 have four** held-out records. A same-data per-pair frequency has an optimistic Brier of **0.15032**; it uses the target outcome itself and is **not a valid oracle estimate**. A leave-one-out per-pair frequency has Brier **0.32527**; that is an honest within-pair held-out-record calculation but uses other held-out responses and is not a deployable unseen-pair predictor. Its poor score illustrates how noisy estimating a pair-specific probability from only one to three other responses can be. Neither number is the true oracle.

## Model comparison on exactly the G2 repeated-photo-pair subset

The reference Brier is **0.21923** on these 3,681 responses. The paired gap is model Brier minus the estimated reference; lower Brier is better. The 95% intervals resample the same photo-pair clusters for both terms. They rely on the same caveats above; positive gaps are not an identified feature-level explanation.

| Saved model | Brier | Accuracy | Brier gap vs estimated reference | Working 95% interval for gap |
|---|---:|---:|---:|---:|
| Original M0 | 0.24976 | 51.62% | +0.03053 | [0.02184, 0.03936] |
| Original M1 | 0.23495 | 59.30% | +0.01572 | [0.00810, 0.02402] |
| Original M2 | 0.23866 | 58.84% | +0.01943 | [0.01130, 0.02763] |
| Original M3-H | 0.24731 | 53.46% | +0.02808 | [0.01941, 0.03709] |
| Original M3-C | 0.23824 | 58.82% | +0.01901 | [0.01104, 0.02726] |
| Original M3-Y | 0.24330 | 56.48% | +0.02407 | [0.01551, 0.03283] |
| Original M4-C | 0.23780 | 58.92% | +0.01857 | [0.01050, 0.02682] |
| Demographic-only D | 0.24788 | 53.33% | +0.02865 | [0.01973, 0.03779] |
| Cat + side | 0.23293 | 60.99% | +0.01370 | [0.00598, 0.02218] |
| Cat + demographics | 0.23272 | 61.02% | +0.01349 | [0.00573, 0.02198] |
| CNN + demographics | 0.23562 | 59.74% | +0.01639 | [0.00811, 0.02480] |
| Kernel M3-C | 0.23814 | 58.52% | +0.01891 | [0.01073, 0.02713] |
| Cat + CNN | 0.23461 | 59.71% | +0.01537 | [0.00756, 0.02349] |

The complete [model comparisons CSV](model_comparisons.csv) contains **204 cohort/level/source/model rows**, including all stored G2, G3 and G4 prediction groups, aliases, side-only and temperature variants, and repeated saved comparator copies. Aliases and copies are not independent models or datasets. Each row gives both full-scope Brier and matched repeated-subset Brier, matched accuracy, reference Brier and pointwise paired gap interval. Cat + CNN G4 is explicitly unsupported and has no saved G4 predictions; no value is imputed.

For example, the repeated-photo-pair reference is 0.22278 in G3 and 0.21762 in G4. Original M1 (G3) has Brier **0.23485**, gap **+0.01207** with working interval **[0.00207, 0.02195]**; original M4-C (G4) has Brier **0.23865**, gap **+0.02103**, interval **[0.01261, 0.02964]**. G3 Cat+demographics has gap **+0.00971**, interval **[-0.00034, 0.01975]**. These retrospective results are not independent confirmations, and conditional pair-bootstrap intervals do not account for the original transfer clustering.

## Can we estimate an accuracy ceiling?

Not reliably from these repeats. If the working Bernoulli assumptions held and the oracle knew **only** the pair, its conditional accuracy would be `max(p, 1-p)`. A mean pair-only Brier risk `q` restricts mean oracle accuracy to `[1-2q, 0.5+sqrt(0.25-q)]`; this follows from bounds on `E|p-0.5|`. Plugging in G2's estimated `q=0.21923` gives approximately **56.15%–67.54%**. These are **plug-in bounds, not a confidence interval**: they inherit sampling uncertainty in `q`, and an estimated `q` may even exceed 0.25. They are not a ceiling for a model with extra information, not a guaranteed bound on realized sample accuracy, and not a universal human ceiling.

## Synthesis

Repeated held-out judgments of the same photo pair agree only about **55%–56%**, so disagreement is substantial in these observations. Under the stated exchangeability and independence assumptions, the estimated pair-only Brier risk is around **0.22**, while saved G2 models have repeated-subset Brier around **0.23–0.25**. That is consistent with both substantial outcome variability **and** a possible remaining probability-prediction gap relative to a hypothetical pair-only oracle. It does not show how much of that gap is recoverable by the available features, whether the models missed a particular signal, or whether disagreement is irreducible. The estimated accuracy range is too broad and assumption-dependent to call 60% close to a proven ceiling.

## Verification

[Provenance](analysis_provenance.json) records input and output hashes, bootstrap settings and the pre-existing-file snapshot. [Independent verification](verification.json) explicitly enumerates response pairs, recomputes leave-one-out and in-sample Brier, and checks all 204 saved-model rows and six agreement summaries. Three synthetic tests cover orientation, within-cat exclusion and small-sample formulas. Bootstrap intervals were not independently replayed. The [presentation PDF](presentation_heldout_agreement.pdf) summarizes these same observations.
