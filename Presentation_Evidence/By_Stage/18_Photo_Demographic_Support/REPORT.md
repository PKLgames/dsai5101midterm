# Photo-by-demographic data support

## Question and scope

How many comparison **response records** involve each photo in each recorded demographic category? This is a retrospective, outcome-blind structural description, not a preference analysis. It uses no test rows or outcome columns and fits no models. It matches the earlier cat-support study: the original common visual fitting mask and strict G2 validation mask are used; fitting and fitting-plus-validation development are reported separately. Development includes fitting and is not an independent sample.

The fixed universe is all **1,296 distinct fitting photos**, including photo/group cells with zero observations. The strict validation photos are in this same fitting universe. Fitting contains 17,421 responses on 5,634 unordered photo pairs; development contains 21,163 responses on 6,842 pairs. Every response contains two distinct photos, and contributes one appearance to each photo, **including within-cat comparisons**. Thus there are 34,842 fitting and 42,326 development photo appearances per demographic variable. In contrast, the companion cat-level analysis counts a within-cat response once for its cat. Do not add appearances across variables: they describe the same responses.

Responses are not necessarily unique people, and respondent linkage and independence are not established. Counts do not measure preference, causation, effect size, or statistical power. The cutoffs of 5, 10 and 20 below are illustrative support descriptions, not prespecified power thresholds. Quartiles use NumPy's default linear interpolation.

## Per-photo group coverage

Each median is across all 1,296 photos, including zero cells. The last two columns show how many photos have **fewer than 10** response appearances in that group. Appearances pool different photo pairs and opponents; they are not repeated ratings of the same pair.

| Variable | Category | Fit median | Fit photos <10 | Development median | Development photos <10 |
|---|---|---:|---:|---:|---:|
| gender | man | 15 | 197 | 18 | 83 |
| gender | woman | 11 | 482 | 13 | 240 |
| pet_history | current | 9 | 717 | 11 | 443 |
| pet_history | never | 10 | 614 | 12 | 365 |
| pet_history | past | 7 | 1,051 | 8 | 819 |
| pet_care | current_regular | 4 | 1,268 | 5 | 1,241 |
| pet_care | never | 7 | 921 | 9 | 682 |
| pet_care | occasionally | 7 | 1,035 | 8 | 832 |
| pet_care | past_regular | 7 | 936 | 9 | 682 |
| pet_contact | daily | 5 | 1,186 | 6 | 1,075 |
| pet_contact | never | 1 | 1,296 | 1 | 1,296 |
| pet_contact | often | 5 | 1,216 | 6 | 1,083 |
| pet_contact | rarely | 8 | 879 | 10 | 624 |
| pet_contact | sometimes | 6 | 1,124 | 7 | 960 |
| future_pet | no | 2 | 1,295 | 3 | 1,294 |
| future_pet | unsure | 5 | 1,211 | 6 | 1,126 |
| future_pet | yes | 18 | 89 | 22 | 35 |
| mbti_ie | E | 7 | 1,037 | 8 | 826 |
| mbti_ie | I | 15 | 200 | 18 | 79 |
| mbti_ie | unknown | 4 | 1,272 | 5 | 1,215 |
| all six variables | &lt;missing&gt; | 0 | 1,296 | 0 | 1,296 |

The six missing-category rows coincide in the underlying source: these are repeated descriptions of the same missing responses, not six independent missing groups. In fitting, 739 photos have zero missing-category appearances; in development, 665 have none. The recorded MBTI category `unknown` is distinct from missing.

The [per-photo table](per_photo_demographic_counts.csv) records response appearances, distinct photo pairs, distinct opponent photos, and within-/between-cat counts. The [group summaries](group_support_summary.csv) include response totals, minima, quartiles, maxima, zero and under-5/10/20 counts, and within-cat support. The [matched-group table](matched_group_support.csv) describes all observed within-variable group pairs.

## Interpretation and comparison with cat-level counts

- **The granularity matters.** Every cat had at least ten fitting appearances in each gender group in the companion analysis, but **197/1,296 photos for men and 482/1,296 for women** have fewer than ten. Fitting medians are 15 for men and 11 for women, versus cat-level medians of 78 and 57. This is not a like-for-like independence comparison: cats have multiple photos and the counting rule differs for within-cat responses.
- **Coverage varies by group.** In fitting, `pet_contact=never` has median one appearance per photo and 439 photos have none; even in development, its median is one and 341 have none. For `pet_care=current_regular`, fitting median is four and 815 photos have fewer than five; development median is five and 600 have fewer than five. For `future_pet=yes`, the fitting median is 18 and 89 photos have fewer than ten, so not every group is equally sparse.
- **Within-cat photo comparisons are a smaller subset.** In fitting, 328 photos have no within-cat appearance in the men's group and 399 have none in the women's group; development has 213 and 278 respectively. An appearance from a within-cat response is counted for **both** photos.
- **A shared photo is not necessarily a matched pair.** In fitting, 1,293 photos appear in both gender groups and 746 have at least ten responses in each; development has 1,295 and 1,026 respectively. The two groups may have seen that photo with different opponents. Only 3,883 fitting and 4,713 development pairs have at least one response in each gender group; 735 and 907 pairs respectively have at least two in each.
- **The earlier five-per-group pair rule is unmet.** Development has at most four responses total for any pair, so zero pairs meet the prior operational rule of at least five responses in each group. This is not a universal power theorem and does not prove that every possible pooled model or alternative design is impossible.

These are structural counts, not group preference estimates. They do not establish which demographic likes a particular photo or semantic feature, account for respondent dependence, or adjust for opponent composition, other demographics or multiple testing. A defensible semantic comparison needs a specified design, adequate support for its actual contrasts, and analysis of the dependence structure. These counts alone neither establish an effect nor determine a required sample size.

## Provenance and verification

[Analysis provenance](analysis_provenance.json) records structural input/output hashes and the pre-existing project snapshot. [Independent verification](verification.json) reconciles all **67,392 photo/group/scope cells**, **52 group summaries**, and **92 matched-group comparisons** using a separate pandas endpoint expansion; all 92 pair-overlap results agree with the prior cat-support table. Five synthetic tests cover within-cat endpoints, zero cells, repeated pairs, invalid identities and outcome independence. The [presentation PDF](presentation_photo_demographic_support.pdf) displays the same data, not new observations. Original frozen model and evaluation artifacts were not modified by the counting analysis.
