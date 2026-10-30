"""Known-truth synthetic BT and scalar-ranking demonstrations."""

from __future__ import annotations

import hashlib
import json
import os
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import expit

from .estimator import fit_logistic


def expected_loss(true_probabilities: Any, logits: Any) -> float:
    q, eta = np.asarray(true_probabilities, dtype=np.float64), np.asarray(logits, dtype=np.float64)
    if q.shape != eta.shape or np.any((q < 0) | (q > 1)) or not np.isfinite(eta).all():
        raise ValueError("Invalid expected-risk probabilities or logits.")
    return float(np.mean(q * np.logaddexp(0.0, -eta) + (1 - q) * np.logaddexp(0.0, eta)))


def generate_simulations(seed: int, repetitions: int = 100) -> list[dict[str, Any]]:
    if repetitions < 1:
        raise ValueError("Simulations require at least one repetition.")
    rng = np.random.default_rng(seed)
    u = np.array([np.log(2.0), 0.0, -np.log(2.0)])
    ab_bc_ac = np.array([[1., -1., 0.], [0., 1., -1.], [1., 0., -1.]])
    cycle = np.array([[1., -1., 0.], [0., 1., -1.], [-1., 0., 1.]])
    q_a = expit(ab_bc_ac @ u)
    q_b = np.concatenate((q_a, 1 - q_a))
    q_c = np.full(3, 0.8)
    design_b = np.block([[ab_bc_ac, np.zeros_like(ab_bc_ac)],
                         [np.zeros_like(ab_bc_ac), ab_bc_ac]])
    records: list[dict[str, Any]] = []
    for simulation in ("A", "B", "C"):
        for n in (10, 50, 200):
            for repetition in range(repetitions):
                if simulation == "A":
                    draws = rng.binomial(n, q_a)
                    specs = [("BT", ab_bc_ac, draws / n, q_a, ab_bc_ac,
                              expected_loss(q_a, ab_bc_ac @ u))]
                elif simulation == "B":
                    draws = rng.binomial(n, q_b)
                    specs = [
                        ("pooled", ab_bc_ac, (draws[:3] + draws[3:]) / (2 * n),
                         q_b, np.vstack((ab_bc_ac, ab_bc_ac)), float(np.log(2.0))),
                        ("group_aware", design_b, draws / n, q_b, design_b,
                         expected_loss(q_b, np.concatenate((ab_bc_ac @ u, -(ab_bc_ac @ u))))),
                    ]
                else:
                    draws = rng.binomial(n, q_c)
                    specs = [("BT", cycle, draws / n, q_c, cycle, float(np.log(2.0)))]
                for model, x, target, truth, evaluation_design, reference in specs:
                    fit = fit_logistic(x, target, np.full(x.shape[1], 1e-6))
                    risk = expected_loss(truth, evaluation_design @ fit.coefficients) if fit.accepted else None
                    records.append({
                        "simulation": simulation, "model": model, "responses_per_pair_per_group": n,
                        "repetition": repetition, "seed": seed, "accepted": fit.accepted,
                        "expected_log_loss": risk, "reference_log_loss": reference,
                        "pair_oracle_log_loss": expected_loss(q_c, np.log(q_c / (1 - q_c)))
                        if simulation == "C" else None,
                        "gradient_max": fit.gradient_max, "objective": fit.objective,
                        "solver_status": fit.status, "solver_message": fit.message,
                        "iterations": fit.iterations, "attempts_json": json.dumps(fit.attempts),
                        "coefficients_json": json.dumps(fit.coefficients.tolist()),
                    })
    return records


def _figure(path: Path, records: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, simulation, title in zip(axes, ("A", "B", "C"), ("Correct BT", "Opposite groups", "Cycle")):
        data = [r for r in records if r["simulation"] == simulation and r["accepted"]]
        for model in sorted({r["model"] for r in data}):
            means = [np.mean([r["expected_log_loss"] for r in data
                              if r["model"] == model and r["responses_per_pair_per_group"] == n])
                     for n in (10, 50, 200)]
            ax.plot((10, 50, 200), means, marker="o", label=model)
        if simulation == "C":
            ax.axhline(float(np.log(2)), linestyle="--", color="black", label="best BT")
            ax.axhline(expected_loss(np.full(3, 0.8), np.full(3, np.log(4))),
                       linestyle=":", color="gray", label="pair oracle")
        else:
            ax.axhline(0.5911, linestyle="--", color="black", label="contextual oracle")
        ax.set_xscale("log")
        ax.set_xticks([10, 50, 200], ["10", "50", "200"])
        ax.set_title(title)
        ax.set_xlabel("Responses per pair per group")
        ax.legend(fontsize="small")
    axes[0].set_ylabel("Exact expected log loss (nats)")
    fig.suptitle("Known-truth simulations; not real-data results")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_simulations(root: Path, seed: int) -> dict[str, Any]:
    outputs = [root / "artifacts" / "simulations.parquet", root / "figures" / "simulations_ABC.pdf",
               root / "artifacts" / "simulation_provenance.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Stage 5 simulation outputs already exist; refusing to overwrite.")
    records = generate_simulations(seed)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), outputs[0])
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
    _figure(outputs[1], records)
    failures = sum(not record["accepted"] for record in records)
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {
        "seed": seed, "repetitions": 100, "response_counts": [10, 50, 200],
        "simulation_ridge": 1e-6, "side_intercept": False,
        "side_intercept_explanation": "No position effect is part of the known-truth simulations; cycle orientation must not become a side predictor.",
        "risk": "exact equally weighted pair/group expected logistic loss from logits",
        "failed_fits": failures, "fit_count": len(records),
        "synthetic_label_control": "deferred: requires Stage 7 selected visual/contextual procedure and fixed development settings",
        "source_sha256": {name: digest(root / name) for name in
                          ("src/estimator.py", "src/simulations.py", "requirements-stage5.txt")},
        "library_versions": {name: metadata.version(name) for name in ("numpy", "scipy", "pyarrow", "matplotlib")},
        "outputs_sha256": {path.relative_to(root).as_posix(): digest(path) for path in outputs[:2]},
    }
    outputs[2].write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance
