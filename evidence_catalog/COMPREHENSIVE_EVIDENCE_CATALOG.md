# Comprehensive project evidence catalog

**Project:** Group 19, *Predicting to Diagnose: Cat Identity, Visual Representations, and Context in Cat-Photo Preferences*  
**Collection date:** 31 October 2026, HKT  
**Purpose:** explain the available figures, tables, numerical results, fitted artifacts, integrity records, and limitations across the entire project, including material not copied into the presentation folder.

## 1. How to use this catalog

This document is the interpreted evidence list. The companion [file inventory](file_inventory.csv) contains **every inventoried project file individually**: path, canonical source, presentation-copy flag, purpose, supported claim, interpretation limit, byte size, SHA-256, CSV/Parquet row count, column count, and PDF page count where applicable. The [integrity audit](integrity_audit.json) records the checks performed during collection.

The inventory covers **2,086 files**, including 1,558 dataset photographs, 64 review-panel images, executable sources, tests, and presentation duplicates. It excludes the root Git database, virtual environments, downloaded-model cache, this new catalog directory, and runtime `__pycache__`/plotting-configuration directories. An incidental retained legacy plotting cache is inventoried as a legacy file, not scientific evidence.

**Do not count files as independent evidence.** A PDF, its source CSV, its row-level predictions, and its two presentation copies usually describe the same observations. The inventory's `canonical_source` field makes this explicit. It is not meaningful to add their sample sizes together.

### 1.1 Verification performed now

- The **G2, G3, and G4 freeze seals** match their recorded artifacts.
- **202 recorded provenance hash references** were checked across the core freeze, graph/development/evaluation, report/package, and G3/G4 records.
- Both current synthesis CSVs preserve the original Stage 10 prefix and satisfy the Stage 10.5 append lineage.
- **120 presentation copies from 60 selected source files** match their source hashes.
- The original code ZIP matches the hash in its historical clean-environment reproduction record and passes ZIP CRC verification: **233 entries**, comprising 232 payload files plus its manifest.
- The inventoried original files did not change during this collection.

These are **integrity checks**, not new empirical findings. No preference model, simulation, evaluation, or existing test suite was rerun for this catalog. Historical successful tests and reconstruction logs are described as historical records, not newly executed tests.

### 1.2 Technical-plan fidelity

The supplied plan snapshot has SHA-256:

`a37501097b424e24694019ba706c7d8b55add467311862bedf8c5a76beed769b`

The originally persisted [technical plan](../project-plan-source.txt) has:

`85ecf8eb38e8a9bf5574d3de388cd887d00441fa6732e475f5939ec5243ea823`

The documented difference is the Stage 11 PDF-only reporting requirement. The principal statistical target, model definitions, diagnostic rules, G3 restrictions, and Section 23 deployment rules remain the basis of interpretation.

**Important deviations/extensions:**

1. Original G3 was **disabled at Stage 0**. The later G3 experiment is a separately authorized retrospective analysis, not faithful pre-development G3 reservation.
2. **G4 is not defined in the original plan.** Its separately authorized target is one unseen cat versus one known cat, with its own cat-disjoint procedure.
3. Stage 10.5 demographic semantics, frozen-test abstention utility, and the G3 within-cat follow-up are explicitly exploratory supplements.
4. Official prediction is a **deployment refit on all labeled responses**, not another held-out evaluation.
5. The original G2 pipeline, report, archive, and claims were not rewritten to absorb the later quarantined G3/G4/utility/deployment results.

## 2. Evidence vocabulary and claim map

The original framework uses:

- **T1:** latent score estimation and ranking.
- **T2:** descriptive behavioral/disagreement analysis.
- **T3:** out-of-sample pairwise probability prediction/calibration.
- **T4:** contextual or multimodal reward modeling.

Permitted activity labels are Predict, Explain, Show evidence, Clarify, and Visualize. Evidence-status labels are Real-data claim, Known-truth demonstration, and Methodological rule. Below, “primary,” “secondary,” “exploratory,” and “administrative” describe the evidential role; they do not replace those labels.

### 2.1 Original claim IDs

| ID | Question | Main supporting evidence | Current bounded answer |
|---|---|---|---|
| C01 | What agreement/disagreement is observed at different support levels? | Development pair/support/Wilson/entropy tables and figures | Observed disagreement exists, but individual pairs have at most four development responses; no high-support example qualifies. |
| C02 | Does cat identity improve over a constant baseline? | Frozen G2 main results and primary contrast | Yes under this qualified internal holdout and conditional pair bootstrap: +0.029432 nats/response. |
| C03 | Does photo identity improve over cat identity? | Matched M1/M2 primary contrast | Not established; the point estimate is worse for M2, and the paired interval spans zero. |
| C04 | How do H/C/Y shared scorers compare with photo identity? | Matched validation/test results, visual-sharing contrast | CNN is the strongest tested shared representation by point loss; its advantage over M2 is not resolved by the primary interval. |
| C05 | Does pet-care context improve the matched selected shared scorer? | M3-C/M4-C primary contrast; side-adjusted secondary comparison | A small conditional predictive improvement is observed for C; H/Y contextual procedures were pooled fallbacks. |
| C06 | Which diagnostics support explanations for gains/underperformance? | Fit/validation gaps, paths, learning curves, temperature, ablations, recovery, calibration, QA | Several patterns constrain explanations; none identifies a unique cause of residual uncertainty. |
| C07 | Do simulations demonstrate recovery, pooling and scalar-ranking restrictions? | Known-truth simulations and fair-coin controls | Synthetic demonstrations behave as designed; they are not findings of real-data cycles or leakage-free certification. |

The original seven-question registry is [claims.csv](../claims.csv). Its `planned` fields are historical registration, not current result status. The current ledger is [claims_final.csv](../claims_final.csv).

### 2.2 Additional recorded synthesis rows

The current ledger has **23 rows**: C01-C07, S01-S10, and X105-01-X105-06.

| ID | Subject | Evidence and permitted interpretation |
|---|---|---|
| S01 | Representation sensitivity | H/C/Y/M1/M2 test losses; CNN strongest shared point result; pooled H/Y context is zero by construction. |
| S02 | Disagreement/context support limitations | Pair bands and shared-group support; no adequately supported pair-group contrasts. |
| S03 | Aggregate context does not imply every subgroup improves | Missing-context test slice has only 78 responses; its M4-C point loss is worse. |
| S04 | Nested learning curves | Same eligible validation intersection at every size; context support and effective model may change. |
| S05 | Recovery of estimated photo-score pattern | CNN has higher in-sample estimated-score reconstruction correlation than H/Y. |
| S06 | Feature-group conditional predictive value | H/Y deletion refits; not percentages of behavior caused by groups. |
| S07 | PCA diagnostic | PCA64 did not improve the original CNN validation loss. |
| S08 | Feature operational quality | Extraction, coverage and review passed; this does not establish semantic correctness. |
| S09 | Within-/between-cat distinction | Different subgroup point losses; aggregate results cannot substitute for within-cat prediction. |
| S10 | Unresolved questions | Primary photo/visual intervals span zero; no unique explanation of remaining loss. |
| X105-01-X105-06 | Six demographic semantic descriptions | Development-only, raw unadjusted rates for gender, pet_history, pet_care, pet_contact, future_pet and mbti_ie; not causal or matched preference effects. |

G3, G4, within-cat G3, utility and deployment results are **not additional completed primary C02-C05 claims**. Where this catalog names them, it is describing separate evidence families, not silently adding new IDs to the frozen claim ledger.

### 2.3 Global interpretation safeguards

1. Natural-log loss is in **nats per response**; lower is better.
2. A contrast is **reference minus candidate**; a positive value favors the candidate.
3. Main G2 inference uses the four registered approximate **98.75% intervals**. Secondary/G3/G4/subgroup/utility intervals are **95% pointwise**, with their stated multiplicity limitations.
4. An interval spanning zero means the direction is unresolved under that approximation. An interval excluding zero is not a causal finding or a guarantee under every dependence model.
5. Repeated responses, pairs, photos and cats are not interchangeable units. Respondent IDs are unavailable.
6. No numerical information limit, objective attractiveness, respondent-disjoint generalization, causal visual/context/demographic effect, or decomposition of disagreement into behavioral causes is established.

## 3. Stage 0-2: governance, raw data, and original G2 partition

### 3.1 Governance and input records

| Item | What it is | Claim supported / limitation |
|---|---|---|
| [protocol.md](../protocol.md) | Original research scope, candidate registry, gates, nonclaims and prior-analysis acknowledgement | Methodological rules for every original claim; not proof of a pristine test. |
| [decisions.yaml](../decisions.yaml) | Machine-readable seeds, grids, thresholds, option flags and authorized changes | Verifies original G3/fusion/hybrid/permutation were disabled; later experiments must remain separate. |
| [project-plan-source.txt](../project-plan-source.txt) | Persisted authoritative original plan | Execution contract, not empirical evidence. |
| [outcome_access_log.csv](../artifacts/outcome_access_log.csv) | 14 original access-history entries, A01-A14 | Documents earlier analyses, development-only access and the one frozen Stage 9 test access. Later isolated experiments have their own access records. |
| [contributions.csv](../contributions.csv) | Assigned team responsibilities and reviews | Ownership assignments, not certification of work actually completed. |
| [proj-docs.md](../proj-docs.md) | Historical analyses/notes | Evidence of prior outcome exposure and possible selection effects; do not present its historical statistics as current frozen results. |
| [README.md](../README.md) | Current stage instructions plus older dataset/legacy setup documentation | Dataset schema and routes; newer locks/reproduction records are authoritative for tested environment. |

### 3.2 Raw inputs and audited counts

| Input/evidence | What it contains | Verified facts | Supports / does not support |
|---|---|---|---|
| [dataset/train.csv](../dataset/train.csv) | Original labeled course responses | 24,896 unique IDs; 12,921 A and 11,975 B; 21,893 between-cat and 3,003 within-cat responses | Defines training observations; historical label exposure must be disclosed. |
| [dataset/test.csv](../dataset/test.csv) | Original unlabeled official rows | 6,224 unique IDs, all original choice/prob_a blank; 5,225 between-cat and 999 within-cat rows | Official target structure, **not measured official performance**. |
| [dataset/images](../dataset/images) | 1,558 original photographs | 1,296 photos appear in labeled responses; 237 labeled cats | Image assets, not 1,558 independent preference observations. Each image is individually hashed in the inventory. |
| [data_audit.csv](../tables/data_audit.csv) | Schema, category, missingness, duplicate and count checks | No blocking audit errors; zero duplicate-content image groups; 46 identical-looking training response groups and 29 official groups retained | Raw validity and scope, not extraction quality or prediction success. |
| [cleaning_log.csv](../artifacts/cleaning_log.csv) | Explicit anomaly actions/reasons | Retention and quarantine history | Avoids undocumented deletion; identical-looking responses are not established duplicates. |
| [raw_hashes.json](../artifacts/raw_hashes.json) | Raw-file SHA-256 manifest | 1,560 files: two CSVs and 1,558 photos | Original-byte integrity; does not certify label truth. |
| [official_test_coverage.csv](../tables/official_test_coverage.csv) | 6,224 row-level overlap classifications against all valid labeled data | 3,112 rows with at least one unseen cat; 3,112 unseen pairs of known photos; zero in the other two categories | Target-aware routing motivation; no official accuracy. |

Each of the six recorded demographics has **508 missing training entries** and **130 missing official entries**. Missing demographic values and documented MBTI `unknown` are distinct. Gender was audited from observed values rather than an invented vocabulary.

The source CSV hash/check did not establish image decoding or semantic extraction quality; those belong to Stage 3.

### 3.3 Original canonical split

| Item | Contents and scale | Claim support |
|---|---|---|
| [rows_canonical.parquet](../artifacts/rows_canonical.parquet) | 24,896 rows; 19 fields including IDs, endpoints, orientation, canonical pair, partition and masked original G2 test outcomes | C01-C06 coordinate consistency and partition provenance. |
| [split_manifest.parquet](../artifacts/split_manifest.parquet) | 8,049 unordered photo pairs; deterministic seed 20261028; count-band collapse rule | Original pair-disjoint G2 allocation. |
| [test_orientation.py](../tests/test_orientation.py), [test_split.py](../tests/test_split.py) | Coordinate/display and allocation tests | Software safeguards. File presence is not a new test-run result. |

| Original partition | Responses | Unordered photo pairs |
|---|---:|---:|
| Fitting | 17,421 | 5,634 |
| Validation | 3,742 | 1,208 |
| Internal G2 test | 3,733 | 1,207 |
| Development = fitting + validation | 21,163 | 6,842 |

All rows for a pair follow one partition. Photos may recur across G2 partitions **by design**. The original test outcome fields in canonical data are null, but the raw training CSV still contains those labels: masking is procedural, not access control. Earlier analyses prevent calling this split historically pristine.

## 4. Stage 3-4: features, review, matched populations, and graphs

### 4.1 Extracted feature evidence

| Artifact | What it is | Verified scale / modeling use | Supported claim / limitation |
|---|---|---|---|
| [H.parquet](../features/H.parquet) | Handcrafted image-statistics cache | 1,558 rows; 20 original measurements; 19 retained fitting dimensions | C04/C06: a usable fixed handcrafted representation. Statistics are not validated “aesthetic quality.” |
| [C.parquet](../features/C.parquet) | Frozen MobileNetV3-Small feature-layer embeddings | 1,558 rows; 576 L2-normalized coordinates; classifier excluded | C04/C06: fixed shared CNN representation, not semantic labels for coordinates. |
| [Y.parquet](../features/Y.parquet) | YOLO-CLIP automated attributes, numeric coat/mask measurements, hard labels, quality fields and compressed selected masks | 1,558 rows; nine numeric + ten categorical modeling fields; 100 encoded fitting dimensions | C04/C06 and X105: automated measurements, not ground-truth breed/pose annotations. |
| [H manifest](../features/manifests/H.json) | Formulas, feature order/units, EXIF/RGB rules, thumbnail and source hashes | Aspect-preserving maximum 256 thumbnail; original oriented dimensions used for size features | Measurement reproducibility, not causal interpretation. |
| [C manifest](../features/manifests/C.json) | Exact model/checkpoint/device/transforms and cache hashes | **Short side 232**, bicubic, center crop 224, ImageNet normalization | Faithful actual pipeline; do not substitute a presumed resize of 256. |
| [Y manifest](../features/manifests/Y.json) | Actual YOLO/OpenCLIP checkpoints, prompts/taxonomy, crop/mask/fallback rules and determinism | GPU extraction about 599.547 seconds; five no-detection images | Fixed measurement pipeline, not calibrated attribute probabilities. |
| [y_gpu_smoke.json](../artifacts/y_gpu_smoke.json) | CUDA smoke/device record | Operational feature execution check | Not a preference-model or full-extraction accuracy result. |

All H/C/Y extraction records succeeded, and each covers **100% of the 3,742 strict-validation responses**, above the 95% operational gate. Y includes **five no-detection** and **21 dog-fallback** cases. No detection is a valid modeled measurement outcome, not automatically an extraction failure. Coat statistics are not fabricated from the entire background; unavailable measurements remain explicitly missing.

Feature extraction is outcome-blind and was permitted for all photos. Learned imputation/scaling/category pooling/RMS normalization is fitted on **unique fitting photos**, not repeated response-weighted endpoints.

### 4.2 Human QA, admissions and population freeze

| Item | What it is | Supports / limits |
|---|---|---|
| [feature_quality.csv](../tables/feature_quality.csv) | Three automated QA summaries | Historical `pending_human_review` fields describe the pre-review snapshot, not current final admission. |
| [feature_review_manifest.json](../artifacts/feature_review_manifest.json) | Immutable 64-image selection, two reviewers, fixed flags and hashes | 48 global hash-ranked fitting photos plus up to 16 flagged photos; seed 20261103. |
| [feature_review.csv](../artifacts/feature_review.csv) | 128 reviewer-image records | Outcome-blind review by Lu Yuqing and Yuen Lok Yan; no unresolved issue fields recorded. Small sample, not full annotation. |
| [review_panels](../artifacts/review_panels) | 64 per-image JPG panels with crop/detection/mask/attribute views | Every panel is individually listed in the inventory; complementary to review records, not independent evidence. |
| [feature_admission_decisions.csv](../artifacts/feature_admission_decisions.csv) | Final H/C/Y admission decisions | All admitted with “review found no problem”; use this for final status. |
| [common_population_manifest.parquet](../artifacts/common_population_manifest.parquet) | 24,896 structural membership rows | Original full/common fitting and held-out populations coincide. |
| [common_population_report.json](../artifacts/common_population_report.json) | Frozen counts, final decisions and historical QA/status fields | Historical adequacy-review wording remains intact; subsequent adequacy acceptance is separate. |
| [stage7_adequacy.json](../artifacts/stage7_adequacy.json) | Explicit common-population adequacy acceptance | Operational go decision with precision/power limitation, not a power calculation. |

The exact frozen common report hash is `4090cfd31c58c873849ad6df557c49776c9051f307efba6524495c65c491c32c`. Historical CRLF bytes and snapshot statuses are part of that record; they were not “cleaned up” for this catalog.

### 4.3 Fitting graph evidence

| Item | What it contains | C02-C06 support / nonclaim |
|---|---|---|
| [graph_summary.csv](../tables/graph_summary.csv) | Four full/common × photo/cat graph summaries | Establishes fitting connectivity and structure. |
| [graph_components.parquet](../artifacts/graph_components.parquet) | 3,066 full/common node records: component, opponents and support | Used to judge strict-G2 support and subgroup fitting-opponent counts. |
| [graph_edges.parquet](../artifacts/graph_edges.parquet) | 55,540 full/common weighted undirected/directed edge records | Fitting-only graph construction; not a real-data cycle test. |
| [g2_eligibility.parquet](../artifacts/g2_eligibility.parquet) | 24,896 row-level full/common strict-G2 masks | Frozen structural eligibility, not outcome-selected exclusions. |
| [split_coverage.csv](../tables/split_coverage.csv) | 12 full/common partition/type summaries | All 3,742 validation and 3,733 test rows structurally eligible. |
| [official_test_coverage_stage4.csv](../tables/official_test_coverage_stage4.csv) | Updated 6,224 official structural support records | Coverage/routing information, not official performance. |
| [graph_support.pdf](../figures/graph_support.pdf) | One-page graph/coverage visualization | Visual summary of the same support evidence. |
| [graph_provenance.json](../artifacts/graph_provenance.json) | Source/output hashes and graph gate | Graph integrity and fitting-only temporal provenance. |

Full and common fitting graphs are identical:

- **Photos:** 1,296 nodes; 5,634 undirected edges; one undirected component; 9,280 directed edges; two strongly connected components; largest SCC 1,295 photos.
- **Cats:** 237 nodes; 4,842 undirected edges; one undirected component and one directed SCC; 2,098 within-cat fitting responses do not create between-cat edges.

Directed strong connectivity concerns the ordinary **unpenalized** BT MLE. It is not the same as whether the ridge-penalized estimator yields finite accepted coefficients.

## 5. Stage 5-6: known-truth demonstrations and development description

### 5.1 Estimator demonstrations

| Item | Contents | Claim supported |
|---|---|---|
| [simulations.parquet](../artifacts/simulations.parquet) | 1,200 accepted synthetic fit results; 100 repetitions at 10/50/200 responses per pair/group | C07: recovery and model restriction under specified known truth. |
| [simulation_provenance.json](../artifacts/simulation_provenance.json) | Seed 20261029, design, objective and hashes | Known-truth reproducibility; zero failed fits. |
| [simulations_ABC.pdf](../figures/simulations_ABC.pdf) | One-page visual comparison of the three scenarios | Clarifies scalar-score and context model behavior. |
| [test_objective.py](../tests/test_objective.py), [test_recovery.py](../tests/test_recovery.py) | Stable objective/analytic gradient, recovery, aggregation and fallback tests | Methodological/software evidence, not evidence of real-data cycles. |

| Known-truth case | Construction/reference | Correct interpretation |
|---|---|---|
| Correct BT | Scores `(log 2, 0, -log 2)`; pair probabilities 2/3, 2/3, 0.8; expected oracle loss **0.591144** | Tests numerical/statistical recovery when the model is correctly specified. |
| Opposite groups | Equal groups with scores `u` and `-u`; pooled probabilities exactly 0.5; pooled reference **0.693147** versus contextual **0.591144** | Calibrated pooled 0.5 can conceal predictive context. Not proof that real pet-care groups behave this way. |
| Cycle | qAB = qBC = qCA = 0.8; pair oracle **0.500402**, best equal-weight scalar BT **0.693147** | Demonstrates a scalar-ranking restriction. No empirical cycle detection was performed. |

Mean expected losses from the saved synthetic fits, with 100 repetitions in each cell:

| Scenario / fitted model | 10 responses per pair/group | 50 | 200 | Relevant reference |
|---|---:|---:|---:|---:|
| A correct BT | 0.634695 | 0.597281 | 0.592705 | 0.591144 |
| B group-aware BT | 0.634647 | 0.597246 | 0.592688 | 0.591144 |
| B pooled BT | 0.707474 | 0.695616 | 0.693739 | 0.693147 |
| C scalar BT on cycle | 0.718545 | 0.696959 | 0.694427 | Best scalar BT 0.693147; pair oracle 0.500402 |

Recovery toward a known reference as simulated support grows is evidence about this specified estimator/data-generating setup, not an empirical learning curve of human preferences.

### 5.2 Development support, proportions and entropy

| Item | Rows | What it is / supporting claim |
|---|---:|---|
| [pair_summary.csv](../tables/pair_summary.csv) | 13,684 | Pair-by-development-subset canonical counts/proportions/Wilson intervals; C01. Overlapping fit/validation/development records are not 13,684 independent pairs. |
| [pair_support_summary.csv](../tables/pair_support_summary.csv) | 36 | Support-band summaries by partition/type; C01 and S02. |
| [photo_support.csv](../tables/photo_support.csv) | 3,705 | Photo response/opponent counts by development subset; C01/C06. |
| [context_support.csv](../tables/context_support.csv) | 15 | Context response/pair/photo/cat support and fitting eligibility; C05/S02. |
| [context_pair_support.csv](../tables/context_pair_support.csv) | 31,744 | Pair-group response cells and proportions; S02. |
| [context_overlap.csv](../tables/context_overlap.csv) | 30 | Shared pairs and the at-least-five-responses-in-each-group gate; S02. |
| [pair_entropy.csv](../tables/pair_entropy.csv) | 13,684 | Empirical within-pair entropy in nats on the same observed responses; C01, explicitly descriptive. |
| [disagreement_examples.csv](../tables/disagreement_examples.csv) | 2 | Two requested example slots recorded as unavailable because no pair has 20 development responses. |
| [support_distribution.pdf](../figures/support_distribution.pdf) | 1 page | Visualizes support bands, not a measure of individual consistency. |
| [disagreement.pdf](../figures/disagreement.pdf) | 1 page | Empirical disagreement/proportion description, with sparse-support qualification. |
| [stage6_provenance.json](../artifacts/stage6_provenance.json) | JSON | Development-only source/output hashes and working-model/support rules. |

**Key development facts:** 21,163 responses; 6,842 pairs; **370 singleton pairs**, **6,472 pairs with 2-4 responses**, none with five or more; maximum support four. **4,412 pairs** contain choices for both canonical endpoints. No pair meets the 20-response example threshold, and **zero shared context pairs** meet the five-responses-per-group rule.

Thus the evidence supports “observed disagreement under low support,” not “precisely balanced preferences,” a respondent-consistency result, or an irreducible information limit. Wilson intervals are binomial working summaries; respondent dependence cannot be fully checked.

## 6. Stage 7-8: selected models and predictive diagnosis

### 6.1 Fitting, preprocessing and selection artifacts

| Item | What it is | Supports / limitation |
|---|---|---|
| [all_fit_attempts.csv](../tables/all_fit_attempts.csv) | 160 recorded original candidates, including fallbacks; accepted numerical fits | Complete development search, not 160 independent empirical discoveries. |
| [validation_results.csv](../tables/validation_results.csv) | 16 selected-model results | C02-C06 selection evidence; validation is not untouched evaluation. |
| [validation_predictions.csv](../tables/validation_predictions.csv) | Saved selected-model validation predictions | Reconstruction, matched comparisons and case diagnosis. |
| [model_selection.json](../artifacts/model_selection.json) | Full selected coefficients/settings, representation alias, reference and interactions | Freezes M3*=C and matched M4*=C; M4 is not independently reselected across representations. |
| [H preprocessing](../artifacts/preprocessing/H.json), [C preprocessing](../artifacts/preprocessing/C.json), [Y preprocessing](../artifacts/preprocessing/Y.json) | Fit-only imputation, scaling, coding, retained columns and RMS normalization | Unique fitting-photo transforms; do not silently use a new split's photos to estimate them. |
| [stage7_verification.json](../artifacts/stage7_verification.json) | 16 selected models reconstructed on 3,742 validation responses; tolerance 1e-12 | Accepted software/population checks, not an independent test. |
| [model_resources.csv](../tables/model_resources.csv) | 16 stored-parameter/support/backbone/runtime records | Operational complexity; not effective degrees of freedom. |

The [models directory](../models) contains **16 frozen fitting-only model JSONs**:

- Full and common M0, M1, M2 and M2-no-position: eight files.
- M3-H/C/Y and matched M4-H/C/Y: six files.
- M3-side and M4-side: two secondary diagnostic files.

These filenames and their hashes are individually listed in the inventory. M3* and M4* are aliases for the selected C models, not additional fitted JSONs. H/Y pooled M4 records are not extra independent interaction fits.

| Model | Selected validation loss | Retained feature dimensions | Stored coefficients |
|---|---:|---:|---:|
| M0 | 0.692972 | 0 | 1 |
| M1 | 0.662130 | 0 | 238 |
| M2 | 0.669075 | 0 | 1,297 |
| M3-H | 0.682642 | 19 | 20 |
| M3-C | 0.665476 | 576 | 577 |
| M3-Y | 0.676747 | 100 | 101 |
| M4-C | 0.665303 | 576 | 2,881 |

M4-H and M4-Y selected exact pooled fallbacks. Original C used ridge 0.0001; original contextual C used multiplier 100. The original reference is `never`; four interaction categories are `current_regular`, `missing`, `occasionally`, `past_regular`.

Trainable preference coefficients must be separated from frozen extraction parameters: the resource table records **927,008 CNN feature-backbone parameters**, **2,876,848 YOLO parameters**, and **151,277,313 CLIP parameters**. Inference timings explicitly cover existing design matrix to probability and exclude extraction, preprocessing and file IO.

### 6.2 Development subgroup, confidence and calibration diagnostics

| Item | Scale | What it explains / claim supported |
|---|---|---|
| [diagnostic_results.csv](../tables/diagnostic_results.csv) | 176 rows | Development subgroup/support/quality loss, Brier, accuracy and probability/logit summaries; C06. |
| [diagnostic_calibration.csv](../tables/diagnostic_calibration.csv) | 1,430 rows | Development reliability bins and counts; C06. Empty/low-support bins are retained. |
| [diagnostic_calibration.pdf](../figures/diagnostic_calibration.pdf) | 1 page | Development reliability visualization, not the frozen-test reliability figure. |
| [diagnostic_confidence.csv](../tables/diagnostic_confidence.csv) | 3 rows | M2/M3*/M4* probability/logit/loss summaries; confidence errors may matter. |
| [diagnostic_interpretation.csv](../tables/diagnostic_interpretation.csv) | 4 rows | Result, alternatives, nonclaim and next-check statements for development diagnosis. |
| [worst_validation_losses.csv](../tables/worst_validation_losses.csv) | 60 records | Fixed worst-loss audits, 20 for each diagnostic model; explains model errors, not typical human preference. |

A key C06 pattern: M2 fitting loss **0.607925** is much lower than M1 **0.646183**, but M2 validation loss **0.669075** is worse than M1 **0.662130**. This is consistent with estimation variance/overfit or confidence mismatch, but does not isolate one mechanism. H/C/Y extraction, matched rows and numerical acceptance were checked before attributing ordinary underperformance.

Calibration uses ten equal-width probability bins. Support flags protect against treating sparse observed frequencies as precise calibration evidence. Even good calibration cannot prove complete information capture.

### 6.3 Penalty paths, controls and learning curves

| Item | Scale | Support / limitation |
|---|---|---|
| [penalty_path_refits.csv](../tables/penalty_path_refits.csv) | 84 refits | Original H/C/Y pooled/contextual path reproduction; fit/validation/norm behavior supports shrinkage/variance diagnosis. |
| [penalty_paths.pdf](../figures/penalty_paths.pdf) | 1 page | Plot of registered penalty paths; no post-test grid extension. |
| [synthetic_label_control.csv](../tables/synthetic_label_control.csv) | 60 records | 20 independent fair-coin fitting/validation repetitions for M2, M3*, M4*. |
| [stage8_core.json](../artifacts/stage8_core.json) | JSON | Operational control passed; zero failed control fits, no investigation trigger. |
| [learning_curves.csv](../tables/learning_curves.csv) | 24 selected results | Two seeds × four nested fitting fractions × M2/M3*/M4*. |
| [learning_curve_candidates.csv](../tables/learning_curve_candidates.csv) | 288 candidates | Penalty/rho/pooled tuning at each curve size. |
| [learning_curve_context_support.csv](../tables/learning_curve_context_support.csv) | 40 records | Reference/interaction support and fallback shifts at each seed/fraction. |
| [learning_curve_preprocessing.json](../artifacts/learning_curve_preprocessing.json) | Eight subset transform records | Unique subset-fitting-photo preprocessing; not reused full-fitting transforms. |
| [learning_curves.pdf](../figures/learning_curves.pdf) | 1 page | Curves against actual fitting-response counts on a common validation intersection. |

Fair-coin validation mean losses were **0.719592 (M2)**, **0.699655 (M3*)**, and **0.699773 (M4*)**, against constant 0.5 prediction loss **0.693147**. No repetition improved over the constant baseline. This supports “no operational fair-coin improvement trigger,” **not a universal absence-of-leakage certificate**.

| Fair-coin procedure | Mean validation loss | Minimum across 20 runs | Maximum | Runs improving on constant 0.5 |
|---|---:|---:|---:|---:|
| M2 | 0.719592 | 0.710321 | 0.727571 | 0/20 |
| M3* | 0.699655 | 0.696264 | 0.703668 | 0/20 |
| M4* | 0.699773 | 0.696513 | 0.704111 | 0/20 |

Learning-curve fractions are 25%, 50%, 75%, 100%, seeds 20261030 and 20261031. The common intersection is **1,836 validation responses / 589 pairs**, not the complete 3,742-row validation population. At 25%, missing context is unsupported and uses reference fallback in both seeds. The representation remains C, but preprocessing, context eligibility and penalties are refitted/tuned separately. Do not infer an information limit or a uniform saturation trend merely from a plotted endpoint.

| Curve seed | Fitting fraction | Fitting responses / pairs / photos | M2 validation loss | M3* loss | M4* loss | Selected M4 mode |
|---|---:|---|---:|---:|---:|---|
| 20261030 | 25% | 4,367 / 1,408 / 1,161 | 0.682937 | 0.675666 | 0.675633 | Interactions, rho 100 |
| 20261031 | 25% | 4,299 / 1,408 / 1,140 | 0.687363 | 0.676712 | 0.675571 | Interactions, rho 10 |
| 20261030 | 50% | 8,737 / 2,817 / 1,288 | 0.676994 | 0.671707 | 0.671707 | Exact pooled |
| 20261031 | 50% | 8,636 / 2,817 / 1,281 | 0.680579 | 0.674022 | 0.673771 | Interactions, rho 10 |
| 20261030 | 75% | 13,057 / 4,225 / 1,296 | 0.676786 | 0.670957 | 0.670957 | Exact pooled |
| 20261031 | 75% | 13,007 / 4,225 / 1,295 | 0.678850 | 0.671791 | 0.671697 | Interactions, rho 10 |
| 20261030 | 100% | 17,421 / 5,634 / 1,296 | 0.672514 | 0.666987 | 0.666987 | Exact pooled |
| 20261031 | 100% | 17,421 / 5,634 / 1,296 | 0.672514 | 0.666987 | 0.666987 | Exact pooled |

The point curves generally improve with more fitting data in this fixed intersection, but they do not establish a plateau or universal benefit of context. In particular, the full-size curve selects pooled M4 on this intersection, while original full-validation selection chose interactions. Those are different validation populations, not a contradiction.

Original full-fitting context support, also reproduced at each 100% curve size:

| Context | Fitting responses | Distinct fitting pairs |
|---|---:|---:|
| current_regular | 2,600 | 2,214 |
| missing | 360 | 353 |
| never | 4,969 | 3,592 |
| occasionally | 4,533 | 3,337 |
| past_regular | 4,959 | 3,577 |

### 6.4 Representation recovery, ablations, PCA, temperature and contributions

| Artifact | Result / purpose | Supported interpretation and nonclaim |
|---|---|---|
| [representation_recovery.csv](../tables/representation_recovery.csv) | H RMSE 0.489664 / correlation 0.326535; C 0.396799 / 0.657831; Y 0.461713 / 0.454021 | S05: C recovers estimated fitting-photo M2 scores more closely in this in-sample probe. Those targets are noisy fitted scores, not independent preference truth. |
| [grouped_ablations.csv](../tables/grouped_ablations.csv) | 49 H/Y feature-group deletion candidates | S06: conditional validation value after deletion/refit; not causal feature importance. |
| [cnn_pca_diagnostic.csv](../tables/cnn_pca_diagnostic.csv) | Seven candidate rows; PCA64 selected diagnostic loss 0.671440 versus original C 0.665476 | S07: this compression probe did not improve point validation loss; not promoted. |
| [temperature_diagnostic.csv](../tables/temperature_diagnostic.csv) | 21 model/temperature rows; selected M2 T=1.5, M3*/M4* T=1 | C06: validation-fixed confidence adjustment opportunity; secondary only. |
| [feature_contributions.csv](../tables/feature_contributions.csv) | 90 grouped contribution records for fixed first-five common-validation response examples across six procedures | Exact additive explanation of fitted logits, not human preference causes. |

Selected deletion-refit losses:

| Representation / dropped group | Validation loss | Full-block loss | Reading |
|---|---:|---:|---|
| H color/brightness | 0.685778 | 0.682642 | Worsened point prediction after deletion. |
| H contrast/sharpness/edges | 0.686180 | 0.682642 | Worsened point prediction after deletion. |
| H size | 0.685051 | 0.682642 | Worsened point prediction after deletion. |
| Y coat measurements | 0.686845 | 0.676747 | Largest tested Y deletion degradation. |
| Y framing/view | 0.676664 | 0.676747 | Slightly lower diagnostic point loss after deletion. |
| Y eyes/posture | 0.676707 | 0.676747 | Slightly lower diagnostic point loss after deletion. |
| Y inferred breed | 0.678180 | 0.676747 | Worsened point prediction after deletion. |

Those tiny favorable Y deletions were not promoted to the primary pipeline. Differences are development diagnostics without an independent paired-test superiority claim.

Related rules/status/hash records are [stage8_protocol.json](../artifacts/stage8_protocol.json), [stage8_probe_protocol.json](../artifacts/stage8_probe_protocol.json), [stage8_contribution_protocol.json](../artifacts/stage8_contribution_protocol.json), [development_decisions.csv](../artifacts/development_decisions.csv), [representation_diagnostic_status.json](../artifacts/representation_diagnostic_status.json), [learning_curve_status.json](../artifacts/learning_curve_status.json), [diagnostic_report_status.json](../artifacts/diagnostic_report_status.json), and [stage8_provenance.json](../artifacts/stage8_provenance.json). These document accepted completion and development-only access, not separate empirical replications.

## 7. Stage 9: primary frozen G2 results and uncertainty

### 7.1 Canonical evidence artifacts

| Item | Scale | What it supports |
|---|---|---|
| [model_freeze.json](../artifacts/model_freeze.json) and [seal](../artifacts/model_freeze.sha256) | Registry/settings/source masks and hashes before original test access | Temporal/integrity basis for C02-C05. |
| [test_predictions.parquet](../artifacts/test_predictions.parquet) | **70,927 prediction rows** = 19 model/population groups × 3,733 responses | Exact metric reconstruction and matched response/pair contrasts. Not 70,927 independent test observations. |
| [main_results.csv](../tables/main_results.csv) | 19 overall rows | Loss, Brier, accuracy, pair-weighted loss, conditional intervals, probability distribution, fit/validation, penalties and resource scope. |
| [primary_contrasts.csv](../tables/primary_contrasts.csv) | Four contrasts | Registered approximate 98.75% paired response/pair intervals. |
| [secondary_results.csv](../tables/secondary_results.csv) | Eight contrasts | H/C/Y context, position, group-side and temperature diagnostics; 95% pointwise unadjusted. |
| [subgroup_results.csv](../tables/subgroup_results.csv) | 304 rows; 228 supported and 76 low-support | Test subgroup loss/Brier/accuracy by comparison type, context, fitting support and feature quality. |
| [test_calibration.csv](../tables/test_calibration.csv) | 2,280 bin records | Overall/subgroup reliability frequencies, counts and flags; empty bins remain explicit. |
| [calibration.pdf](../figures/calibration.pdf) | 1 page | Frozen-test reliability visualization. |
| [evaluation_provenance.json](../artifacts/evaluation_provenance.json) | Single evaluated-once record and output hashes | Original test access, uncertainty rules and preservation. |

All primary comparisons use the same **3,733 responses / 1,207 unordered pairs**. Bootstrap uses 5,000 shared unordered-pair resamples, retaining every response and repeated sampled-pair multiplicity, with the response denominator recomputed in each draw. Pair-weighted loss is a separate estimand, not a replacement for the response-weighted primary metric.

### 7.2 Main numerical results

| Model | Response log loss | Brier | Accuracy | Pair-weighted log loss |
|---|---:|---:|---:|---:|
| M0 | 0.692644 | 0.249748 | 51.647% | 0.692289 |
| M1 | 0.663212 | 0.235367 | 59.175% | 0.666282 |
| M2 | 0.670741 | 0.238917 | 58.800% | 0.671434 |
| M3-H | 0.688112 | 0.247516 | 53.308% | 0.688751 |
| M3-C = M3* | 0.669202 | 0.238335 | 58.746% | 0.670074 |
| M3-Y | 0.679873 | 0.243482 | 56.443% | 0.682929 |
| M4-C = M4* | 0.668296 | 0.237885 | 58.853% | 0.668984 |

M4-H equals M3-H and M4-Y equals M3-Y **exactly by pooled fallback**. Full/common M0-M2 results also coincide here because the matched populations coincide. These equalities do not demonstrate independent replication.

### 7.3 Four original primary contrasts

| Claim | Contrast | Response difference | Approximate 98.75% interval | Pair-weighted difference |
|---|---|---:|---|---:|
| C02 | M0 minus M1 | +0.029432 | [0.016801, 0.041898] | +0.026007 |
| C03 | M1 minus M2 | -0.007529 | [-0.018227, 0.003146] | -0.005153 |
| C04 | M2 minus M3-C | +0.001539 | [-0.008144, 0.012219] | +0.001361 |
| C05 | M3-C minus M4-C | +0.000907 | [0.000149, 0.001654] | +0.001090 |

**Faithful summary:** cat identity improves over constant prediction under this holdout. Extra photo-specific detail and CNN superiority over M2 are unresolved by their primary intervals. Matched context gives a small conditional predictive gain for C. M1 has the lowest primary-ladder point loss, but the plan does not license converting every point ranking into a significant pairwise conclusion.

### 7.4 Secondary numerical evidence

| Secondary contrast | Difference | 95% pointwise interval | Permitted reading |
|---|---:|---|---|
| H pooled context | 0 | [0, 0] | No H contextual extension selected; not “context is irrelevant.” |
| Y pooled context | 0 | [0, 0] | Same pooled-fallback interpretation. |
| M3-side minus M4-side | +0.000895 | [0.000313, 0.001483] | Small context gain persists after allowing group-specific displayed-side tendencies; still noncausal. |
| M2-no-position minus M2 | +0.000786 | [-0.000724, 0.002275] | Position contribution direction unresolved in this secondary test. |
| M2 minus M2 temperature | +0.003642 | [0.000219, 0.007195] | Validation-fixed T=1.5 improves secondary test loss to 0.667100. |
| M3/M4 temperature | 0 | [0, 0] | T=1 was selected; these predictions are aliases of originals. |

The temperature-adjusted M2 was not made a new primary/deployment winner on internal-test results.

### 7.5 Supported comparison-type and low-support slices

| Model | Within-cat loss: 455 responses / 170 pairs | Between-cat loss: 3,278 responses / 1,037 pairs |
|---|---:|---:|
| M0 | 0.693563 | 0.692516 |
| M1 | 0.693712 | 0.658979 |
| M2 | 0.700564 | 0.666602 |
| M3-C | 0.701543 | 0.664713 |
| M4-C | 0.700987 | 0.663758 |

This supports S09: within-cat and between-cat tasks are not interchangeable. For M1 within a cat, cat-score differences cancel; only its displayed-side intercept remains. “M1 predicts between cats well” does not mean it separates photos of the same cat.

The missing-context slice has **78 responses / 76 pairs**, below the 100-response reporting threshold; M3-C loss **0.691267**, M4-C **0.694561**. It supports a warning against assuming improvement for every subgroup, not a strong group-specific negative conclusion. Other support slices are individually available in the subgroup table. Reported support thresholds are operational, not power guarantees.

## 8. Stage 10 and presentation coefficient evidence

### 8.1 Evidence-linked synthesis

| Item | What it is | Support / nonclaim |
|---|---|---|
| [claims_final.csv](../claims_final.csv) | 23 current result/question/interpretation/alternative/nonclaim/next-evidence rows | The authoritative recorded synthesis; not a new test. |
| [diagnostic_synthesis.csv](../tables/diagnostic_synthesis.csv) | Same current content as the claims ledger | Convenience table, not independent replication. |
| [stage10_provenance.json](../artifacts/stage10_provenance.json) | Original 17-row output provenance and 151 numerical/categorical source references | Original-prefix provenance, not the current whole-file hash. |
| [stage10_5_provenance.json](../artifacts/stage10_5_provenance.json) | Current hashes and preserved-prefix append lineage | Explains the six later rows without overwriting original synthesis. |

The original 17-row hash was `b1bbbc2f567459a23b7d56e6d6718d39c6fc2f4c92a2c036319f952098db49f5`. The current 23-row hash is `9cc8001c3cd61ed38b72a8d0890cb28c7edc2d96a599043c1ef1dfd49a7f808a`. Treating the historical Stage 10 hash as a current whole-file requirement would incorrectly diagnose corruption.

### 8.2 “Which features are strong?” presentation artifacts

| Item | What it is | Supports / limitation |
|---|---|---|
| [presentation_feature_coefficients.csv](../tables/presentation_feature_coefficients.csv) | Selected signed coefficients from original frozen M3-H/M3-Y | Describes model associations under their actual fitted coding and scale. |
| [presentation_feature_importance_H.pdf](../figures/presentation_feature_importance_H.pdf) | 1 page, five positive and five negative handcrafted coefficients | Which included H directions receive larger signed fitted weights; not standalone importance or causality. |
| [presentation_feature_importance_Y.pdf](../figures/presentation_feature_importance_Y.pdf) | 1 page, five positive and five negative encoded attribute coefficients | Model reliance under regularized categorical coding; not true breed or human causal preferences. |
| [presentation_context_interactions.csv](../tables/presentation_context_interactions.csv) | Selected C pooled/reference/deviation/group coefficients | How the fitted linear scorer changes across pet-care groups. |
| [presentation_context_interactions.pdf](../figures/presentation_context_interactions.pdf) | 1 page, grouped context weights for eight CNN coordinates | Predictive association, not causal context effect; coordinates have no invented semantic names. |
| [presentation_feature_provenance.json](../artifacts/presentation_feature_provenance.json) | Frozen-model inputs/output hashes and ranking rule | Presentation extraction, not model refitting or new preference evidence. |

The actual models are [M3-H.json](../models/M3-H.json), [M3-Y.json](../models/M3-Y.json), and [M4-C.json](../models/M4-C.json), not imagined `common_M3-H/Y` files.

For C, the selected eight coordinates are **cnn_415, cnn_149, cnn_199, cnn_385, cnn_174, cnn_336, cnn_089, cnn_083**, ranked by maximum absolute group deviation against `never`. These are unnamed embedding coordinates, not “eye,” “breed,” or “pose” dimensions.

Coefficient magnitude depends on scaling, collinearity, coding and shrinkage. The separate grouped-ablation refits are the evidence about conditional predictive value; coefficient bars cannot replace them.

The selected signed coefficient values are:

| H positive feature | Coefficient | H negative feature | Coefficient |
|---|---:|---|---:|
| log_pixels | +0.503464 | sharpness_laplacian_var | -0.376631 |
| edge_density | +0.479896 | contrast | -0.303714 |
| warm_pixel_ratio | +0.357677 | mean_r | -0.236415 |
| mean_b | +0.211965 | saturation_mean | -0.145964 |
| saturation_std | +0.199653 | entropy_bits | -0.105258 |

| Y positive encoded feature | Coefficient | Y negative encoded feature | Coefficient |
|---|---:|---|---:|
| breed=british_shorthair | +1.770284 | breed=&lt;other&gt; | -3.299679 |
| colour_pattern=&lt;other&gt; | +1.187634 | eyes=&lt;other&gt; | -1.025718 |
| colour_primary=white | +1.055211 | colour_pattern=solid | -0.878875 |
| eyes=both_open | +0.683554 | colour_secondary=gray | -0.411591 |
| colour_secondary=cream | +0.554981 | colour_primary=black | -0.395851 |

For one-hot attributes, consider contrasts between category coefficients within the same family and the centering/regularization convention. An `other` coefficient reflects fitting-category pooling, not a natural semantic property. These are not odds ratios per raw measurement unit because fitting preprocessing and block normalization matter.

## 9. Exploratory Stage 10.5: demographic semantic associations

| Item | What it is | Supports / limitation |
|---|---|---|
| [stage10_5_protocol.json](../artifacts/stage10_5_protocol.json) | Development-only scope, demographics, traits, denominators and thresholds | T2 observational description, not test-set modeling. |
| [exploratory_demographic_semantic_rates.csv](../tables/exploratory_demographic_semantic_rates.csv) | 1,456 demographic-group/semantic-trait rate records across 56 traits | X105-01-X105-06 raw comparisons with explicit counts. |
| [exploratory_demographic_semantic_support.csv](../tables/exploratory_demographic_semantic_support.csv) | 26 demographic-group support rows | Denominators and missing-category scope. |
| [exploratory_demographic_semantic_overlap.csv](../tables/exploratory_demographic_semantic_overlap.csv) | 2,576 shared discriminating-pair/group comparison records | Exposes different pair composition rather than treating raw group ranges as adjusted preferences. |
| [exploratory_demographic_semantics.pdf](../figures/exploratory_demographic_semantics.pdf) | 12 pages of support-gated categorical/numeric semantic rate heatmaps | Descriptive association only. |
| [exploratory_demographic_semantic_support.pdf](../figures/exploratory_demographic_semantic_support.pdf) | 1 page of support/coverage | Companion warning against comparing sparse demographic cells. |
| [stage10_5_provenance.json](../artifacts/stage10_5_provenance.json) | Input/output hashes, development-only access and claim-prefix append | Original G2 test outcomes not used in this analysis. |

Population: **21,163 development responses / 6,842 pairs**. All six demographics were examined, retaining missing groups separately and documented MBTI `unknown`.

- **Categorical trait rate:** only comparisons with exactly one endpoint carrying that category; rate is the fraction choosing that endpoint.
- **Numeric trait rate:** only finite, unequal endpoint values; rate is the fraction choosing the higher-value endpoint. Mean signed gap is a separate descriptive quantity.
- Unknown semantic labels/no-detection endpoints are excluded under the recorded trait rules; dog-fallback cases remain explicitly retained/counted.
- Plotted cells require **at least 100 responses and 30 pairs**.
- Wilson intervals are working-model summaries, not cluster-robust discoveries.
- **No matched pair/group comparison qualifies** under the five-responses-in-each-group rule.

Illustration, not an adjusted preference estimate: for CLIP-inferred British-shorthair category, `current_regular` selected the category endpoint **89/131 = 67.9%**, versus `occasionally` **115/225 = 51.1%**. These groups faced different discriminating pairs. The example is not pedigree validation or a causal pet-care effect.

The recorded gender example for `view=front` is woman **1,396/2,688 = 51.9345%** versus missing gender **48/122 = 39.3443%**. The latter is a missing-data group, not a substantively interpretable demographic category. Many large raw ranges involve missing demographics; do not label them broad group preference differences.

The six selected raw-rate comparisons actually recorded in the claim ledger are summarized below. The feature/family and selection rule are recorded with each source claim; these are post-development descriptive examples, not six prespecified discoveries.

| Recorded claim | Demographic | Selected higher-rate group: choices / discriminating responses | Comparator: choices / discriminating responses | Raw range | Shared discriminating pairs | Shared pairs meeting support rule |
|---|---|---|---|---:|---:|---:|
| X105-01 | gender | woman: 1,396/2,688, 51.9345% | missing: 48/122, 39.3443% | 0.125903 | 86 | 0 |
| X105-02 | pet_history | current: 1,192/2,269, 52.5342% | missing: 48/122, 39.3443% | 0.131899 | 73 | 0 |
| X105-03 | pet_care | current_regular: 89/131, 67.9389% | occasionally: 115/225, 51.1111% | 0.168278 | 57 | 0 |
| X105-04 | pet_contact | sometimes: 798/1,515, 52.6733% | missing: 48/122, 39.3443% | 0.133290 | 49 | 0 |
| X105-05 | future_pet | unsure: 647/1,237, 52.3040% | missing: 48/122, 39.3443% | 0.129597 | 54 | 0 |
| X105-06 | mbti_ie | documented unknown: 549/1,052, 52.1863% | missing: 48/122, 39.3443% | 0.128420 | 38 | 0 |

The raw range is a difference in proportions, not a causal effect size. “Unknown” MBTI and missing MBTI are different recorded states. Five examples compare a substantive/unknown group with missing data, and none has adequately supported matched-pair contrasts.

## 10. Frozen-test abstention/utility supplement

| Item | What it is | Supports / limitation |
|---|---|---|
| [abstention_utility_protocol.json](../artifacts/abstention_utility_protocol.json) | User-confirmed payoffs, models, fixed confidence grid and non-selection rules | Explicit post-test exploratory analysis. |
| [exploratory_abstention_utility.csv](../tables/exploratory_abstention_utility.csv) | **105 rows** = five frozen models × 21 thresholds | Coverage, retained counts/pairs, accuracy, loss, utility and pointwise uncertainty. |
| [presentation_abstention_utility.pdf](../figures/presentation_abstention_utility.pdf) | **4 pages**: coverage/accuracy, realized/implied utility, fixed-cutoff counts, safeguards | Selective prediction trade-offs, not a validated abstention policy. |
| [abstention_utility_provenance.json](../artifacts/abstention_utility_provenance.json) | Saved-input/output hashes, layout and protected-artifact records | No refit, raw-outcome reread or original evaluation rerun. |
| [test_abstention_utility.py](../tests/test_abstention_utility.py) | Nine focused cases recorded in the extension workflow | Denominators, empty selections, ties, extremes, paired multiplicity and PDF/output safeguards. Not a new test run in this catalog. |

Models: M0, M1, M2, original frozen M3*=C and matched M4*. Confidence is `max(p, 1-p)`; thresholds **0.50 to 1.00 in 0.025 increments**; ties retained together. Correct prediction earns **+1**, error **-1**, abstention **0**. Total utility is divided by **all original 3,733 responses**, including abstained cases. Selective accuracy is divided only by retained responses and is undefined when none are retained.

At the fixed illustrative M4* cutoff **0.60**:

- **1,575 responses / 619 pairs** retained.
- Coverage **42.1913%**; retained accuracy **65.5238%**.
- Utility per original response **0.130994**, versus **0.177069** for always predicting.

Thus higher selective accuracy does **not** necessarily improve total utility. Under the stated symmetric +1/-1/0 payoffs and calibrated probabilities, every retained prediction has nonnegative expected reward at confidence >=0.5; always predicting is weakly optimal in model-implied total utility. This does not prove real deployment calibration.

| Fixed M4* confidence cutoff | Retained / abstained responses | Correct / incorrect retained | Coverage | Selective accuracy | Utility per original response | 95% utility interval |
|---|---|---|---:|---:|---:|---|
| 0.50 | 3,733 / 0 | 2,197 / 1,536 | 100% | 58.8535% | 0.177069 | [0.143010, 0.211073] |
| 0.60 | 1,575 / 2,158 | 1,032 / 543 | 42.1913% | 65.5238% | 0.130994 | [0.108267, 0.153478] |
| 0.70 | 437 / 3,296 | 307 / 130 | 11.7064% | 70.2517% | 0.047415 | [0.034500, 0.060639] |

Uncertainty uses 5,000 shared unordered-pair draws and exploratory 95% pointwise intervals. Accuracy intervals require sufficient retained response/pair support and no undefined empty-retention draws. No test-optimal cutoff or deployment threshold was selected.

## 11. Quarantined retrospective G3, within-cat G3, and G4

These experiments use separate fitting/validation/model/freeze/output directories. They preserve the original G2 pipeline, but **cannot erase prior outcome exposure**. A fresh directory or different split is not a new independent test.

### 11.1 G3: new photos of known cats

Directory: [g3_experiment](../g3_experiment).

| File/group | What it is | Evidence supported |
|---|---|---|
| [protocol.json](../g3_experiment/protocol.json), [README.md](../g3_experiment/README.md) | Authorized retrospective scope and nonclaims | Explains why this is not original Stage 0 G3. |
| [photo_reservation.csv](../g3_experiment/photo_reservation.csv) | Duplicate-content-aware photo groups hash sorted before outcomes | 64 validation and 64 test photo groups among 1,296 training photos. |
| [core_split.csv](../g3_experiment/core_split.csv), [row_allocation.csv](../g3_experiment/row_allocation.csv) | Separate nonreserved pair split and complete response assignments | All comparisons involving both reserved endpoints quarantined; fitting does not include reserved photos. |
| [coverage.json](../g3_experiment/coverage.json) | Outcome-blind eligibility/support | Fit 14,218 responses / 4,612 pairs; validation 2,063 / 660 / 64 held-out photos; test 2,250 / 718 / 64. |
| [development_access.json](../g3_experiment/development_access.json) | Fitting/validation access and original-partition overlap | Separate fitting includes 2,054 original G2 test responses. |
| [all_fit_attempts.csv](../g3_experiment/all_fit_attempts.csv), [validation_results.csv](../g3_experiment/validation_results.csv) | 94 recorded candidates and eight selected models | C selected on G3 validation; matched M4-C validation loss 0.669603 versus M3-C 0.671638. |
| [selected_models.json](../g3_experiment/selected_models.json) | Separate coefficients, unique-fitting-photo preprocessing and context mapping | Reference `past_regular`, C ridge 0.001, context multiplier 1. |
| [freeze.json](../g3_experiment/freeze.json), [seal](../g3_experiment/freeze.sha256) | Separate choices sealed before G3 test access | New within-procedure ordering, not historical outcome blindness. |
| [evaluation_started.json](../g3_experiment/evaluation_started.json) | Write-once test-access/overlap record | G3 test includes 361 original G2 test responses; other outcomes also appeared in development. |
| [test_predictions.parquet](../g3_experiment/test_predictions.parquet) | 18,000 saved predictions = eight models × 2,250 responses | Metric and pairwise reconstruction. |
| [test_results.csv](../g3_experiment/test_results.csv), [test_contrasts.csv](../g3_experiment/test_contrasts.csv) | Matched test metrics and three paired photo-cluster contrasts | Retrospective G3 performance, not original G2 claims. |
| [presentation_g3_results.pdf](../g3_experiment/presentation_g3_results.pdf) | 1-page model/contrast figure | Presentation of the same results. |
| [evaluation_provenance.json](../g3_experiment/evaluation_provenance.json), [quarantine_verification.json](../g3_experiment/quarantine_verification.json) | Test/freeze/output hashes and original preservation checks | Single evaluation and quarantine integrity. |
| `root_hashes_before.json`, `evidence_hashes_before.json` in this directory | Pre-experiment preservation snapshots | Historical scope, not a ban on legitimate subsequent new files. |
| `common.py`, `prepare.py`, `develop.py`, `evaluate.py`, `tests.py`, `verify_saved.py`, `publish_evidence.py` | Local methods, focused tests, replay and copy-only publication source | Operational reproducibility, not additional empirical observations. Individually linked in the inventory. |

No M2 unseen-photo free score was assigned. Eligible rows have exactly one reserved photo, another nonreserved fitting photo, and a fitting photo of the reserved photo's cat. **247 both-reserved responses** were quarantined.

| Model | G3 test loss | Accuracy |
|---|---:|---:|
| M0 | 0.692926 | 51.289% |
| M1 | 0.662725 | 59.556% |
| M3-H | 0.688707 | 53.689% |
| M3-C | 0.672169 | 57.111% |
| M3-Y | 0.682825 | 55.111% |
| M4-H | 0.688574 | 53.956% |
| M4-C | 0.669966 | 58.578% |
| M4-Y | 0.683571 | 54.578% |

| G3 contrast | Difference | 95% held-out-photo interval |
|---|---:|---|
| M0 minus M1 | +0.030201 | [0.019958, 0.041114] |
| M1 minus M3-C | -0.009444 | [-0.016069, -0.003568] |
| M3-C minus M4-C | +0.002203 | [-0.001249, 0.005418] |

On this retrospective sample, known-cat identity predicts better than the selected shared CNN by point loss and its stated descriptive paired interval. The contextual increment's direction remains uncertain. This is not evidence that new-photo visual transfer is universally useless, nor an independent replication of G2.

### 11.2 G3 within-cat saved-prediction subgroup

Directory: [within_cat_subgroup](../g3_experiment/within_cat_subgroup).

| Item | What it is | Support / nonclaim |
|---|---|---|
| [protocol.json](../g3_experiment/within_cat_subgroup/protocol.json) | Post-test subgroup/support/uncertainty rules | No new fit, model selection or raw test parsing. |
| [within_cat_summary.json](../g3_experiment/within_cat_subgroup/within_cat_summary.json) | 286 responses; 106 pairs; 60 held-out photos; 53 cats; 43 original G2 test responses | Supported descriptive slice, not a new independent test. |
| [within_cat_test_results.csv](../g3_experiment/within_cat_subgroup/within_cat_test_results.csv) | Eight frozen-model subgroup metrics | Same matched responses for every model. |
| [within_cat_contrasts.csv](../g3_experiment/within_cat_subgroup/within_cat_contrasts.csv) | Five contrasts with 5,000 shared held-out-photo draws | Post-test 95% pointwise unadjusted comparisons. |
| [presentation_within_cat_g3.pdf](../g3_experiment/within_cat_subgroup/presentation_within_cat_g3.pdf) | 1-page within-cat results and paired intervals | Explicit M1 score-cancellation and retrospective warnings. |
| [analysis_provenance.json](../g3_experiment/within_cat_subgroup/analysis_provenance.json), [publication_verification.json](../g3_experiment/within_cat_subgroup/publication_verification.json) | Inputs/output hashes and copy-only preservation | Original G2/G3 records unchanged. |
| Local original/evidence snapshots, README and analyze/tests/verify/publish source | Method/provenance/verification | Each individually inventoried; no separate new sample. |

| Model | Within-cat G3 loss |
|---|---:|
| M0 | 0.696249 |
| M1 | 0.696475 |
| M3-H | 0.711427 |
| M3-C | 0.699886 |
| M3-Y | 0.699344 |
| M4-H | 0.712111 |
| M4-C | 0.693665 |
| M4-Y | 0.697761 |

M3-C minus M4-C = **+0.006221**, interval **[0.000188, 0.011935]**; the other four recorded contrasts span zero. This interval is exploratory and unadjusted across the follow-up comparisons. M4-C accuracy is **50.350%**: a paired improvement over its selected counterpart does not establish broadly strong within-cat prediction.

M1 is constant within a cat, with logit **0.085969**; it cannot discriminate the two photos using a cat-score difference. This subgroup explains why aggregate known-cat results must not be substituted for same-cat photo prediction.

### 11.3 G4: one unseen cat versus one known cat

Directory: [g4_experiment](../g4_experiment). This is a **new user-authorized target**, not Section 22 or a original pre-registered G4.

| Item/group | Contents / scale | Supports / limitation |
|---|---|---|
| [protocol.json](../g4_experiment/protocol.json), [README.md](../g4_experiment/README.md) | Exactly one unseen cat versus one known cat; duplicate-aware 10%/10% cat reservation and fixed rules | Ten percent chosen before reservation to allow 20+ held-out cats per partition. |
| [cat_reservation.csv](../g4_experiment/cat_reservation.csv) | 237 cat assignments: 23 validation, 23 test, 191 core | Cat-disjoint separate fitting within this experiment. |
| [core_split.csv](../g4_experiment/core_split.csv), [row_allocation.csv](../g4_experiment/row_allocation.csv) | Separate core pair split and response eligibility/quarantine | No reserved cat in fitting; 1,424 both-reserved/same-reserved-cat responses quarantined. |
| [coverage.json](../g4_experiment/coverage.json) | Fit 11,543 responses / 3,751 pairs / 191 cats / 1,041 photos; validation 3,503 / 1,106 / 23 held-out cats; test 3,479 / 1,101 / 23 | Both operational gates passed; only 23 test clusters, not a precision guarantee. |
| [development_access.json](../g4_experiment/development_access.json) | Historical G2/G3 overlap and connected fitting-cat graph | Fitting includes 1,678 original G2 test responses and 1,065 original G3 test responses. |
| [all_fit_attempts.csv](../g4_experiment/all_fit_attempts.csv), [validation_results.csv](../g4_experiment/validation_results.csv) | 95 recorded candidates; eight selected models | All accepted; representation C selected on validation. |
| [selected_models.json](../g4_experiment/selected_models.json) | Separate coefficients/preprocessing/context; M1-zero convention | C ridge 0.001; matched context multiplier 10; H/Y M4 pooled. |
| [freeze.json](../g4_experiment/freeze.json), [seal](../g4_experiment/freeze.sha256), [evaluation_started.json](../g4_experiment/evaluation_started.json) | Separate pre-test freeze and one-access guard | Test includes 536 original G2 test responses and 324 original G3 test responses; not historically unseen. |
| [test_predictions.parquet](../g4_experiment/test_predictions.parquet) | **27,832 saved predictions** = eight models × 3,479 responses | Reconstruction, not 27,832 independent observations. |
| [test_results.csv](../g4_experiment/test_results.csv), [test_contrasts.csv](../g4_experiment/test_contrasts.csv) | Matched metrics and three held-out-cat paired contrasts | Retrospective one-unseen-cat transfer. |
| [presentation_g4_results.pdf](../g4_experiment/presentation_g4_results.pdf) | 1-page model/contrast figure | Same evidence, explicit cold-start and retrospective warnings. |
| [evaluation_provenance.json](../g4_experiment/evaluation_provenance.json), [publication_verification.json](../g4_experiment/publication_verification.json) | Output/freeze/access hashes and original preservation | No G2/G3 artifact modification. |
| Local source/tests/replay/publish files and original/evidence snapshots | Implementation and integrity | Individually inventoried; not new outcomes. |

M1-zero assigns the unseen cat a **fixed zero score**, retaining the known fitting-cat score. Zero is a cold-start convention, not an estimated score or evidence that the unseen cat is average. M2 is excluded because unseen-cat photos lack fitted free scores.

| Model | G4 test loss | Accuracy |
|---|---:|---:|
| M0 | 0.692195 | 52.285% |
| M1-zero | 0.681203 | 56.194% |
| M3-H / pooled M4-H | 0.684656 | 55.361% |
| M3-C | 0.671809 | 58.005% |
| M3-Y / pooled M4-Y | 0.676295 | 56.970% |
| M4-C | 0.670039 | 58.638% |

| G4 contrast | Difference | 95% held-out-cat interval |
|---|---:|---|
| M0 minus M1-zero | +0.010992 | [0.005621, 0.016576] |
| M1-zero minus M3-C | +0.009394 | [0.000419, 0.018464] |
| M3-C minus M4-C | +0.001770 | [0.000971, 0.002678] |

These results support predictive transfer of the tested shared scorer on this retrospective one-unseen-cat sample, subject to fixed-zero baseline and uncertainty qualifications. They do **not** establish performance for two unseen cats, within-unseen-cat photo choice, respondent-disjoint generalization, or a causal context effect.

## 12. Section 23: official prediction is deployment, not evaluation

Directory: [official_prediction](../official_prediction).

| Item | What it is | Supports / limitation |
|---|---|---|
| [deployment_policy.json](../official_prediction/deployment_policy.json) and [seal](../official_prediction/deployment_policy.sha256) | Validation-only routing and selected settings sealed before deployment refit | No internal-test performance used to choose deployment models. |
| [deployment_started.json](../official_prediction/deployment_started.json) | All-labeled-refit access record | Training now deliberately includes former holdout labels, as Section 23 requires. |
| [deployment_models.json](../official_prediction/deployment_models.json) | Separate M0/M1/M4-C refits and CNN preprocessing on all 24,896 labeled responses / 1,296 photos / 237 cats | Not the fitting-only models used in original test tables. |
| [official_predictions.csv](../official_prediction/official_predictions.csv) | **6,224 original official rows**, same columns/order; only choice/prob_a filled | Deliverable completeness and valid probability format, **not accuracy**. |
| [routing.csv](../official_prediction/routing.csv) | 6,224 route/reason/novelty/logit/probability records | Explains exactly which rule produced each row. |
| [deployment_provenance.json](../official_prediction/deployment_provenance.json) | Source/output hashes, route counts, scope and no-upload status | Original raw/frozen artifacts unchanged. |
| [verification.json](../official_prediction/verification.json) | Independent reconstruction of all 6,224 probabilities, output fields, fit-only parameter sources, all-labeled transforms and gradients | Numerical/output correctness, not measured official performance. |
| Local original snapshot, README, common/policy/run/verify/tests source | Methods and safeguards | Separate from G2/G3/G4 evaluation. |

### Routing and numerical checks

- **3,112 supported known-photo rows:** refitted M1, selected from original common G2 validation loss **0.662130**.
- **3,112 new-photo rows:** refitted M4-C, selected from retrospective G3 validation loss **0.669603**, versus its matched M3-C **0.671638**.
- M1 fallback is defined for unavailable visual features with supported known cats; M0 fallback for unsupported identities/features. No such fallback rows were needed in this file.
- Official novelty: 3,112 rows have no unseen cat; 1,556 have one unseen cat; **1,556 have two unseen cats**.
- Choice counts: **3,737 A / 2,487 B**. These are generated outputs, not observed official outcomes.
- Probabilities are finite in [0,1], and choice is A iff serialized probability >=0.5.
- M1 maximum gradient **4.953665e-7**, M4-C **7.507974e-7**, both below the 1e-5 acceptance gate.
- Prediction SHA-256: `d005ff0de0545b18e1fe3095ad15f38a27d361bfdff1137c3535a70d6e44922f`.

G3 is new-photo/known-cat evidence. Applying its shared scorer to unseen cats is an extrapolation beyond that validation target. Retrospective G4 supports a separate one-unseen-cat target, but was not used to retune this Section 23 policy and does not validate two-unseen-cat comparisons.

No official labels are available. No official accuracy, log loss, confidence interval for official performance, or upload/submission success is established. The original official CSV remains blank and unchanged.

## 13. Stage 11: report, archive, reproduction and administrative evidence

| Item | Contents / verified facts | Supports / limitation |
|---|---|---|
| [Group_19_Report.pdf](../Group_19_Report.pdf) | **42 pages total, 18 main pages**; core argument and diagnostic appendices | Presentation synthesis of existing evidence, not another empirical analysis. |
| [stage11_report_provenance.json](../artifacts/stage11_report_provenance.json) | Report source/output hashes, layout/page records and draft status | Report is still a working draft with human actions pending. |
| [Group_19_Code.zip](../Group_19_Code.zip) | 232 payload files plus PACKAGE_MANIFEST; 13,050,042 bytes | Course-only reproducible snapshot of the original Stage 11 package. Later supplements are not automatically included. |
| [stage11_package_provenance.json](../artifacts/stage11_package_provenance.json) | Archive hash, payload entries and inclusion/exclusion rules | Local draft package created, not submitted. |
| [archive_reproduction.json](../artifacts/archive_reproduction.json) | Exact ZIP extracted to a different path; clean CPU environment; all archived tests passed; statistical replay passed | Historical **116 archived tests**, not the current expanded workspace's test count or a newly rerun suite. |
| [reproduction_statistical.json](../artifacts/reproduction_statistical.json) | Fresh-process 16 selected-model frozen-setting reconstruction | Maximum saved-test metric and probability differences **0.0**; preprocessing and pair bootstrap reproduced. |
| [feature_regeneration_H.json](../artifacts/feature_regeneration_H.json) | Fixed 16-photo H extraction check; 320 numeric comparisons | Maximum observed difference 0.0; subset only. |
| [feature_regeneration_C.json](../artifacts/feature_regeneration_C.json) | Fixed 16-photo C check; 9,216 coordinates | Maximum observed difference 0.0; subset only. |
| [feature_regeneration_Y.json](../artifacts/feature_regeneration_Y.json) | Fixed eight-photo CUDA Y check; 63 numeric and 80 categorical comparisons | Maximum numeric difference **2.517273e-8**; hard labels/masks matched under the recorded check; tolerance 1e-5. |
| [REPRODUCTION.md](../REPRODUCTION.md) | Portable cached route, environment roles, feature subset/full-extraction instructions and restrictions | The entire raw-image/model-selection-to-report pipeline was **not** re-executed in a fresh installation. |
| [CPU lock](../requirements-lock-cpu.txt), [GPU lock](../requirements-lock-y-gpu.txt) | Exact tested environment snapshots | Reproduction methods, not empirical preference evidence. Stage-specific requirement files are separately inventoried. |
| [stage11_decisions.json](../artifacts/stage11_decisions.json) | Five presenter allocation, draft flags, deadline, local-only packaging decisions | Proposed assignments, not completed individual contributions. |
| [stage11_contributions_draft.csv](../tables/stage11_contributions_draft.csv) | Five user-authorized placeholder rows | Team must verify actual completed work. |
| [references.csv](../references.csv) | 12 dataset/method/software/course references, including Chapter 7 placeholder | Bibliographic record, not independent empirical proof. |
| [third_party_resources.csv](../tables/third_party_resources.csv) | Eight resource/license/distribution summaries | Research/education restrictions and license review; not blanket clearance. |
| [Midterm_Report_Guidelines_2026.pdf](../Midterm_Report_Guidelines_2026.pdf) | Three official guideline pages | Report/submission requirement source, not statistical evidence. |

The contribution placeholder is explicitly on **report page 15**. The Chapter 7 reference is not completed. The submission channel is unverified and no upload is recorded. Course deadline: **2 November 2026, noon HKT**; internal target 11 a.m.

The code ZIP excludes raw photographs, pretrained weights, virtual environments, Git and download caches. Course CSVs and cached derived features are restricted to education/research use; do not interpret packaging as public redistribution approval.

## 14. Presentation evidence: curated copies and lineage

Directory: [Presentation_Evidence](../Presentation_Evidence). Its [folder tree](../Presentation_Evidence/folder_tree.txt) shows the two navigation structures:

- `By_Stage`: pipeline order, then isolated retrospective experiments.
- `By_Presentation_Flow`: Data, Models, Diagnostics, Claims.

There are **120 copied assets**, from **60 selected original source files**, plus **13 root metadata files**. The copies add no observations or independent tests.

| Copy batch | Unique sources | Copies | Metadata |
|---|---:|---:|---|
| Original Stage 11.5 selection | 36 | 72 | [selection_manifest.csv](../Presentation_Evidence/selection_manifest.csv), [verification.json](../Presentation_Evidence/verification.json), original before/after snapshots |
| Abstention utility | 4 | 8 | [extension_selection_manifest.csv](../Presentation_Evidence/extension_selection_manifest.csv), [extension_verification.json](../Presentation_Evidence/extension_verification.json) |
| G3 | 7 | 14 | [g3_selection_manifest.csv](../Presentation_Evidence/g3_selection_manifest.csv), [g3_verification.json](../Presentation_Evidence/g3_verification.json) |
| Within-cat G3 | 6 | 12 | [g3_within_cat_selection_manifest.csv](../Presentation_Evidence/g3_within_cat_selection_manifest.csv), [g3_within_cat_verification.json](../Presentation_Evidence/g3_within_cat_verification.json) |
| G4 | 7 | 14 | [g4_selection_manifest.csv](../Presentation_Evidence/g4_selection_manifest.csv), [g4_verification.json](../Presentation_Evidence/g4_verification.json) |

Earlier manifests remain historical. Their smaller original-file counts reflect creation time and scope, not missing current files. The folder tree is intentionally refreshed when a new evidence batch is added. **Official prediction artifacts are not automatically part of the presentation copy selection or original report/ZIP.**

Material outside this folder includes row-level predictions, complete candidate logs, full context/pair tables, feature caches and masks, all original models, individual review panels, provenance/access history, reproduction logs, deployment models/predictions, and methods/tests. The file inventory covers them all.

## 15. What is not a completed evidence claim

The following are absent, disabled, or unestablished rather than silently omitted:

| Item/claim | Current status |
|---|---|
| Pristine historical G2/G3/G4 holdout | Not established; earlier analyses and explicit cross-experiment outcome overlaps are documented. |
| Respondent-disjoint evaluation / individual consistency | Not supported; no respondent identifier in the dataset. |
| High-support disagreement examples | None qualify at 20 development responses; no substitute selected by lowering threshold. |
| Matched demographic/context pair-level preference effects | No adequate >=5-response cells per compared group; raw different-pair rates only. |
| Information-limit / irreducible human-randomness estimate | Not established; in-sample pair entropy is descriptive. |
| Objective attractiveness or causal pose/breed/context effect | Outside supported scope. |
| Empirical real-data preference cycles | Not tested in the core; cycle demonstration is synthetic. |
| True pedigree/semantic annotation validity | Not established by CLIP or limited visual QA. |
| All-block fusion, photo-residual hybrid, grouped permutation, fine-tuning | Disabled original options; no accepted core result. Retained legacy hybrid code is not a completed experiment. |
| Primary PCA or temperature-promoted model | Not promoted; PCA/temperature remain registered secondary diagnostics. |
| Independently reserved Stage 0 G3 | Not performed; later G3 is explicitly retrospective. |
| G4 two-unseen-cat or within-unseen-cat performance | Not evaluated by the authorized one-unseen-cat target. |
| Validated test-selected abstention deployment policy | Not selected; utility curves are exploratory. |
| Measured official-test accuracy/loss | Impossible from current blank official labels. Generated choices are not observed outcomes. |
| Complete human contribution statement / Chapter 7 bibliography / submission | Still pending; package and predictions are local deliverables, not an upload. |
| Fresh complete raw-image-to-model-selection-to-report reproduction | Not run; recorded reproduction is cached-statistical plus fixed extraction subsets. |

The [outputs](../outputs), [processed](../processed) and [notebooks](../notebooks) working directories contain no additional inventoried files at collection time. Retained [legacy](../legacy) source is implementation reference only. Git/environment/download caches are not additional scientific evidence and are outside this catalog's inventory scope.

## 16. Recommended presentation use without overstating results

1. **Data and target:** raw audit, original partition, graph/coverage figure, sparse development support.
2. **Core models:** original matched validation and frozen G2 results; show all four primary contrasts, including both unresolved intervals.
3. **Diagnosis:** fit/validation gap, preselected temperature, recovery/ablations and learning-curve common-population caveat; no “human randomness” shortcut.
4. **Feature associations:** H/Y signed bars and unnamed C context coordinates; explicitly say “Predictive Association, Not Causal Effect.”
5. **Exploratory appendix:** demographic rates with denominators/composition, utility with total-denominator payoff rule, retrospective G3/G4 and within-cat results with prior-exposure warnings.
6. **Deployment appendix:** explain validation-selected routing and official file completeness; do not state official accuracy.
7. **Reproducibility:** exact cached reconstruction, recorded subset tolerances, historical test scope, unchanged freezes, and remaining contribution/reference/submission tasks.

Every number should point to its canonical table/record rather than a presentation copy alone. Prefer “supports,” “consistent with,” and “not established” according to the relevant evidence, rather than converting predictive differences into causal preference stories.
