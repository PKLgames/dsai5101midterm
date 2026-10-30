"""Freeze all development choices before loading internal-test outcomes."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from .development import _hash


def _read_json(root: Path, name: str) -> dict[str, Any]:
    return json.loads(root.joinpath(*name.split("/")).read_text(encoding="utf-8"))


def _verify_hashes(root: Path, report: dict[str, Any], groups: tuple[str, ...]) -> None:
    for group in groups:
        for name, digest in report.get(group, {}).items():
            path = root.joinpath(*name.split("/"))
            if not path.is_file() or _hash(path) != digest:
                raise ValueError(f"Provenance mismatch: {name}")


def _git(root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True,
    )
    return process.stdout.strip()


def freeze_model(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = root / "artifacts/model_freeze.json"
    seal = root / "artifacts/model_freeze.sha256"
    if path.exists() or seal.exists():
        raise FileExistsError("A model freeze already exists; refusing to overwrite it.")
    if not config.get("real_data_modeling_allowed"):
        raise ValueError("The Stage 0 governance gate is closed.")
    for name in ("artifacts/graph_provenance.json", "artifacts/stage6_provenance.json",
                 "artifacts/stage7_verification.json", "artifacts/stage8_core.json",
                 "artifacts/stage8_provenance.json"):
        report = _read_json(root, name)
        _verify_hashes(root, report, ("input_sha256", "inputs_sha256", "source_sha256",
                                      "output_sha256", "outputs_sha256"))
    stage8 = _read_json(root, "artifacts/stage8_provenance.json")
    core = _read_json(root, "artifacts/stage8_core.json")
    stage7 = _read_json(root, "artifacts/stage7_verification.json")
    if stage8.get("status") != "completed" or core.get("status") != "passed_operational_control":
        raise ValueError("Stage 8 diagnostics or synthetic-label control are incomplete.")
    if any((stage8.get("internal_test_outcomes_read"), core.get("test_outcomes_read"),
            stage7.get("internal_test_outcomes_read"))):
        raise ValueError("A preceding stage reports internal-test outcome access.")
    if _hash(root / "src/verify_development.py") != stage7["verification_source_sha256"]:
        raise ValueError("Stage 7 verification source changed.")
    selection = _read_json(root, "artifacts/model_selection.json")
    _verify_hashes(root, selection, ("input_sha256",))
    block = selection["selected_representation"]
    if block != stage8["selected_representation"] or block not in {"H", "C", "Y"}:
        raise ValueError("Selected visual representation does not match Stage 8.")
    selected = selection["selected"]
    m3, m4 = selected["M3*"], selected["M4*"]
    if m3 != selected[f"M3-{block}"] or m4 != selected[f"M4-{block}"]:
        raise ValueError("M3* and M4* are not the matching selected representation.")
    if not m3.get("accepted") or not m4.get("accepted"):
        raise ValueError("Selected visual models are not accepted fits.")
    registry = config["candidate_registry"]
    if registry["primary_contrasts"] != ["M0-M1", "M1-M2", "M2-M3*", "M3*-M4*"]:
        raise ValueError("Primary-contrast registry changed.")
    if config["evaluation"]["bootstrap_replicates"] != 5000:
        raise ValueError("Bootstrap count differs from the registered plan.")
    expected_models = {
        f"{population}:{model}" for population in ("full", "common")
        for model in ("M0", "M1", "M2", "M2-no-position")
    } | {f"M{level}-{representation}" for level in (3, 4) for representation in ("H", "C", "Y")} \
      | {"M3-side", "M4-side"}
    if any(key not in selected or not selected[key].get("accepted") for key in expected_models):
        raise ValueError("A required selected model is missing or unaccepted.")
    with (root / "tables/temperature_diagnostic.csv").open(encoding="utf-8", newline="") as stream:
        temperatures = [row for row in csv.DictReader(stream) if row["selected"] == "True"]
    if {row["model"] for row in temperatures} != {"M2", "M3*", "M4*"} or len(temperatures) != 3:
        raise ValueError("Temperature diagnostics lack unique pre-test choices.")
    raw_hashes = _read_json(root, "artifacts/raw_hashes.json")
    train = root / "dataset/train.csv"
    if _hash(train) != raw_hashes["files"]["dataset/train.csv"]["sha256"]:
        raise ValueError("Raw labeled data changed since the Stage 1 audit.")
    names = {
        "decisions.yaml", "protocol.md", "project-plan-source.txt", "artifacts/raw_hashes.json",
        "artifacts/rows_canonical.parquet", "artifacts/g2_eligibility.parquet",
        "artifacts/split_manifest.parquet", "artifacts/graph_components.parquet",
        "artifacts/common_population_manifest.parquet", "artifacts/common_population_report.json",
        "artifacts/model_selection.json", "artifacts/stage7_verification.json",
        "artifacts/stage8_provenance.json", "artifacts/stage8_core.json",
        "artifacts/development_decisions.csv", "tables/temperature_diagnostic.csv", "tables/model_resources.csv",
        "src/freeze.py", "src/evaluate.py", "src/run.py", "src/development.py", "src/preprocessing.py",
        "src/estimator.py", "tests/test_evaluation.py",
    }
    for name in stage7["output_sha256"]:
        names.add(name)
    for name in stage8["input_sha256"]:
        names.add(name)
    for name in stage8["output_sha256"]:
        names.add(name)
    for block_name in ("H", "C", "Y"):
        names.add(f"features/{block_name}.parquet")
        names.add(f"features/manifests/{block_name}.json")
    hashes = {name: _hash(root.joinpath(*name.split("/"))) for name in sorted(names)}
    status = _git(root, "status", "--short", "--untracked-files=normal")
    freeze = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_branch": _git(root, "branch", "--show-current"),
        "git_worktree_status_before_freeze": status,
        "candidate_registry": registry,
        "selected_representation": block,
        "selected_model_keys": [key for key, value in selected.items()
                                if isinstance(value, dict) and "coefficients" in value and key not in {"M3*", "M4*"}],
        "selected_parameters": {
            key: {name: value[name] for name in ("model", "ridge", "rho", "pooled", "pooled_from")}
            for key, value in selected.items()
            if isinstance(value, dict) and "coefficients" in value and key not in {"M3*", "M4*"}
        },
        "context_reference": selection["reference_context"],
        "context_interactions": selection["interaction_contexts"],
        "primary_contrasts": [
            {"label": "cat_identity", "reference": "common:M0", "candidate": "common:M1"},
            {"label": "photo_detail", "reference": "common:M1", "candidate": "common:M2"},
            {"label": "visual_sharing", "reference": "common:M2", "candidate": f"M3-{block}"},
            {"label": "recorded_context", "reference": f"M3-{block}", "candidate": f"M4-{block}"},
        ],
        "temperature_diagnostics": {row["model"]: float(row["temperature"]) for row in temperatures},
        "diagnostics": ["M2-no-position", "M3-side", "M4-side", "H/C/Y matched comparisons",
                        "registered temperature scaling", "calibration", "support and feature-quality subgroups"],
        "optional_flags": config["options"],
        "bootstrap": {"seed": config["seeds"]["evaluation_bootstrap"], "replicates": 5000,
                      "cluster": "unordered_test_pair", "primary_confidence": 0.9875,
                      "pointwise_confidence": 0.95},
        "raw_training_sha256": _hash(train),
        "library_versions": {name: metadata.version(name) for name in ("numpy", "scipy", "pyarrow", "matplotlib")},
        "artifacts_sha256": hashes,
        "test_outcomes_loaded_at_freeze": False,
        "limitations": [
            "Internal holdout is not certified pristine because of earlier analyses.",
            "Bootstrap is conditional on fitted models and development choices; it omits training variability, respondent linkage and shared-image dependence.",
            "Secondary intervals are descriptive and subject to multiplicity.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(freeze, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with seal.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(_hash(path) + "\n")
    return freeze


def validate_freeze(root: Path) -> dict[str, Any]:
    path = root / "artifacts/model_freeze.json"
    if not path.is_file():
        raise FileNotFoundError("Evaluation requires a saved model freeze.")
    seal = root / "artifacts/model_freeze.sha256"
    if not seal.is_file() or seal.read_text(encoding="ascii").strip() != _hash(path):
        raise ValueError("The model-freeze integrity seal is missing or does not match.")
    frozen = _read_json(root, "artifacts/model_freeze.json")
    if frozen.get("test_outcomes_loaded_at_freeze") is not False:
        raise ValueError("Freeze record does not attest that test outcomes were unopened.")
    for name, digest in frozen["artifacts_sha256"].items():
        candidate = root.joinpath(*name.split("/"))
        if not candidate.is_file() or _hash(candidate) != digest:
            raise ValueError(f"Frozen artifact changed: {name}")
    if _hash(root / "dataset/train.csv") != frozen["raw_training_sha256"]:
        raise ValueError("Raw training data changed after freeze.")
    return frozen
