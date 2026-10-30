"""Seal validation-only official-prediction routing before deployment refitting."""

from __future__ import annotations

import json
import os

from common import EXCLUDED, HERE, ROOT, digest, snapshot, verify_history, write_json


def run() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    names = ("original_hashes_before.json", "deployment_policy.json", "deployment_policy.sha256")
    if any((HERE / name).exists() for name in names):
        raise FileExistsError("Deployment policy already exists; refusing to overwrite.")
    original = snapshot(ROOT, EXCLUDED)
    write_json(HERE / "original_hashes_before.json", original)
    frozen = verify_history()
    g2 = json.loads((ROOT / "artifacts" / "model_selection.json").read_text(encoding="utf-8"))["selected"]
    choices = ("common:M0", "common:M1", "common:M2", "M3*", "M4*")
    if any(not g2[key]["accepted"] or g2[key]["validation_loss"] is None for key in choices):
        raise ValueError("A registered G2 selected model has no accepted validation result.")
    best = min(float(g2[key]["validation_loss"]) for key in choices)
    known = next(key for key in choices if float(g2[key]["validation_loss"]) <= best + 1e-6)
    if known != "common:M1":
        raise ValueError(f"Unexpected validation-selected supported-photo model: {known}")
    g3 = json.loads((ROOT / "g3_experiment" / "selected_models.json").read_text(encoding="utf-8"))
    block = g3["selected_representation"]
    if block != frozen["g3"]["selected_representation"] or block not in {"H", "C", "Y"}:
        raise ValueError("G3 representation does not match its validation freeze.")
    shared = [g3["selected"][f"M3-{block}"], g3["selected"][f"M4-{block}"]]
    if any(not row["accepted"] or row["validation_loss"] is None for row in shared):
        raise ValueError("The matched G3 visual candidates lack accepted validation results.")
    low = min(row["validation_loss"] for row in shared)
    winner = next(row for row in shared if row["validation_loss"] <= low + 1e-6)
    if winner["model"] != "M4-C" or winner["pooled"]:
        raise ValueError(f"Unexpected validation-selected shared scorer: {winner['model']}")
    g3_context = g3["context"]
    if not g3_context["reference"] or len(set(g3_context["interactions"])) != len(g3_context["interactions"]):
        raise ValueError("Invalid frozen G3 context mapping.")
    policy = {
        "status": "validation_only_deployment_rule_sealed_before_refit",
        "known_photo_model": {
            "model": known, "ridge": g2[known]["ridge"], "validation_loss": g2[known]["validation_loss"],
            "candidate_order": list(choices),
            "validation_losses": {key: g2[key]["validation_loss"] for key in choices},
            "tie_rule": "within 1e-6 prefer first in the stated order",
            "source": "frozen G2 common strict-validation selections",
        },
        "new_or_unsupported_photo_model": {
            "model": winner["model"], "block": block, "ridge": winner["ridge"], "rho": winner["rho"],
            "context_reference": g3_context["reference"], "context_interactions": g3_context["interactions"],
            "validation_losses": {row["model"]: row["validation_loss"] for row in shared},
            "tie_rule": "within 1e-6 prefer matched pooled M3",
            "source": "frozen retrospective known-cat G3 validation selections",
        },
        "fallbacks": "If C is unavailable, use M1 for two supported known cats; otherwise use smoothed M0.",
        "refit": "Refit selected models and preprocessing on all audited labeled responses and unique labeled photos; retain selected penalty and G3 context mapping.",
        "test_selection": "No G2, G3, or G4 internal-test metric is used by this routing or parameter selection.",
        "limitations": [
            "The deployment routing was documented after prior outcome analyses; it was not independently preregistered.",
            "The G3 validation used to choose the shared scorer is retrospective, not a pristine holdout.",
            "G3 tests new photos of known cats, not new cats. Applying its shared scorer to new cats is transfer beyond the G3 target.",
            "G4 covered one unseen cat versus one known cat, but is retrospective and does not directly validate comparisons between two unseen cats.",
            "No official test labels are available, and no official-test performance is claimed."
        ],
        "source_sha256": {
            "g2_freeze": digest(ROOT / "artifacts" / "model_freeze.json"),
            "g2_selection": digest(ROOT / "artifacts" / "model_selection.json"),
            "g3_freeze": digest(ROOT / "g3_experiment" / "freeze.json"),
            "g3_selection": digest(ROOT / "g3_experiment" / "selected_models.json"),
            "g4_freeze": digest(ROOT / "g4_experiment" / "freeze.json"),
            "raw_training": digest(ROOT / "dataset" / "train.csv"),
            "official_test": digest(ROOT / "dataset" / "test.csv"),
        },
    }
    if snapshot(ROOT, EXCLUDED) != original:
        raise RuntimeError("Project originals changed while sealing deployment policy.")
    write_json(HERE / "deployment_policy.json", policy)
    (HERE / "deployment_policy.sha256").write_text(digest(HERE / "deployment_policy.json") + "\n", encoding="ascii")
    print(json.dumps({"known_photo": policy["known_photo_model"], "new_photo": policy["new_or_unsupported_photo_model"]}, indent=2))


if __name__ == "__main__":
    run()
