# Official prediction file

`official_predictions.csv` is a new local file with the same columns and row
order as `dataset/test.csv`. Only `choice` and `prob_a` are filled. The raw
test file is unchanged, and no submission or upload is performed.

The sealed `deployment_policy.json` uses validation-selected models, not
internal-test metrics: the G2 common-validation winner M1 for supported known
photos, and the G3 validation-selected matched M4-C for new photos. Missing
visual features with supported known cats use M1; unsupported identities and
features use M0. `routing.csv` records the route and reason for every row.

The selected penalties and G3 context mapping remain fixed. Separate deployment
models and CNN preprocessing are refitted on all audited labeled responses and
unique labeled photos. The saved evaluation models, splits and G2/G3/G4
artifacts are not changed.

The deployment policy was documented after prior outcome analyses. G3 is a
retrospective known-cat, new-photo experiment; using its visual scorer on
unseen cats is extrapolation beyond that target. G4 examined one unseen cat
against a known cat but did not directly evaluate two unseen cats. The official
test has no available labels, so this file has no measured official accuracy.

From the project root using the existing environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe official_prediction\tests.py
```

Policy sealing and deployment refitting are write-once and refuse to
overwrite existing results. The final provenance records source hashes,
selection evidence, numerical acceptance, route counts and output hashes.
`verification.json` records independent reconstruction of the saved predictions,
refit gradients, preprocessing and required output shape.
