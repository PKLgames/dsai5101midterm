# Cat-by-demographic data support

## Question and scope

How many comparison **response records** involve each cat within each recorded demographic category? This is a retrospective, outcome-blind description of the original G2 data structure, not an analysis of preferences. It uses no test rows or outcome columns and fits no models. The fitting set and the combined fitting-plus-validation development set are shown separately; development includes fitting and is not an independent sample. The original common visual fitting mask and strict G2 validation mask were used. All 237 fitting cats are included for every category, even if a cell has zero records.

The fitting set contains 17,421 responses on 5,634 unordered photo pairs; development contains 21,163 responses on 6,842 pairs. A between-cat comparison contributes one appearance to each of its two cats; a within-cat comparison contributes one appearance to the one cat, not two. Thus the corresponding totals are 32,744 and 39,778 cat appearances **per demographic variable**. Do not add these totals across variables: they describe the same responses.

Response records are not necessarily unique people; respondent linkage and independence are not established. The counts do not measure power, preference, effect size, or causation. The per-cat cutoffs below are descriptive, not preregistered power thresholds. Quartiles in the accompanying CSV use NumPy's default linear interpolation.

## Per-cat group coverage

Each median is across all 237 cats. The last two columns count cats with **fewer than 10** response appearances in the indicated group. These cat-level counts pool different opponents and photo pairs; they do not mean the same pair was rated repeatedly.

| Variable | Category | Fit median | Fit cats <10 | Development median | Development cats <10 |
|---|---|---:|---:|---:|---:|
| gender | man | 78 | 0 | 95 | 0 |
| gender | woman | 57 | 0 | 70 | 0 |
| pet_history | current | 47 | 0 | 58 | 0 |
| pet_history | never | 52 | 0 | 63 | 0 |
| pet_history | past | 35 | 0 | 43 | 0 |
| pet_care | current_regular | 21 | 4 | 25 | 0 |
| pet_care | never | 39 | 0 | 48 | 0 |
| pet_care | occasionally | 36 | 0 | 44 | 0 |
| pet_care | past_regular | 39 | 0 | 47 | 0 |
| pet_contact | daily | 28 | 0 | 35 | 0 |
| pet_contact | never | 6 | 225 | 7 | 193 |
| pet_contact | often | 27 | 0 | 33 | 0 |
| pet_contact | rarely | 42 | 0 | 51 | 0 |
| pet_contact | sometimes | 32 | 0 | 39 | 0 |
| future_pet | no | 13 | 51 | 16 | 11 |
| future_pet | unsure | 27 | 0 | 33 | 0 |
| future_pet | yes | 95 | 0 | 116 | 0 |
| mbti_ie | E | 35 | 0 | 43 | 0 |
| mbti_ie | I | 77 | 0 | 94 | 0 |
| mbti_ie | unknown | 23 | 0 | 27 | 0 |
| all six variables | &lt;missing&gt; | 3 | 237 | 3 | 236 |

The six missing-category rows coincide in the source; they are repeated descriptions of the same missing responses, not six independent missing groups. In fitting, 14 cats have no missing-category appearances and 198 have fewer than five; in development, four have none and 174 have fewer than five. The recorded MBTI category `unknown` is distinct from a missing response.

The granular [per-cat table](per_cat_demographic_counts.csv) includes response appearances, distinct photo pairs, distinct photos of the cat, distinct opponent cats, and within-/between-cat counts. The [group summaries](group_support_summary.csv) include response totals, minima, quartiles, maxima, zeros, and the number of cats below 5, 10 and 20 appearances. The [matched-group table](matched_group_support.csv) contains all observed within-variable group pairs.

## Where support is actually thin

- **Not all cat-by-group cells are thin.** For both men and women, every cat has at least 10 total appearances in fitting and development. Median fitting counts are 78 and 57, respectively, and median distinct pair counts are 39 and 35. These are counts of records, not independent respondents.
- **Rare recorded categories are much thinner.** For `pet_contact=never`, the fitting median is six appearances and 225/237 cats have fewer than 10; in development the median is seven and 193/237 have fewer than 10. For `future_pet=no`, the corresponding medians are 13 and 16, with 51 and 11 cats below 10. These are descriptive contrasts, not evidence that a specific model cannot be estimated.
- **Within-cat comparisons are a smaller subset.** In fitting, the median number of within-cat response appearances is five per cat for men and four for women; six cats have none for men and eight have none for women. In development the medians are six and four, and two and four cats respectively have none. For `pet_contact=never`, 165 cats in fitting and 149 in development have zero within-cat appearances. A within-cat response compares two photos of the same cat, and is counted once for that cat.
- **Overlap on the same pair is more restrictive than overlap on the same cat.** All 237 cats appear in both gender groups at least 10 times in each scope. However, only 3,883 fitting photo pairs and 4,713 development pairs have at least one response in each gender group; only 735 and 907 pairs respectively have at least two records in each group. Those pair counts are across the entire pair universe, not counts per cat.
- **No development pair meets the original five-per-group pair-support rule.** A development pair has at most four records in total, so zero pairs can contain at least five responses in each of two groups. The earlier threshold was an operational group-contrast rule, not a universal statistical-power theorem. This calculation covers every pair, not only pairs with a proposed semantic difference.

The aggregate numbers do not establish which demographic likes which cat or feature. In particular, they do not adjust for correlated respondent attributes, repeated responses, opponent/pair composition, multiple testing, or sparse semantic contrasts. Even substantial cat-level coverage does not by itself identify a gender-by-feature effect. A defensible follow-up would require an explicitly designed, sufficiently supported pair/semantic contrast and an analysis accounting for participant and pair structure; these counts alone do not prove that such an effect is impossible or determine a required sample size.

## Provenance and verification

The [analysis provenance](analysis_provenance.json) records structural input and output hashes and the original-project snapshot check. [Independent verification](verification.json) reconciles all 12,324 cat/group/scope cells, all 52 group-summary rows and all 92 matched-group rows using a separate pandas endpoint-expansion calculation; focused synthetic tests cover within-cat deduplication, both photos, zero cells, shared pairs and outcome independence. The [presentation PDF](presentation_cat_demographic_support.pdf) is a visualization of these same records, not additional data. Original frozen model and evaluation artifacts were not modified by the counting analysis.
