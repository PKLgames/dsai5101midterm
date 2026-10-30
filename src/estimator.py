"""Stable penalized logistic estimator shared by identity and visual models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class FitResult:
    coefficients: np.ndarray
    accepted: bool
    objective: float
    gradient_max: float
    status: int
    message: str
    iterations: int
    attempts: tuple[dict[str, Any], ...]


def _validate(
    design: Any, targets: Any, weights: Any, penalties: Any,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    x = sparse.csr_matrix(design, dtype=np.float64) if sparse.issparse(design) else np.asarray(design, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    p = np.asarray(penalties, dtype=np.float64)
    if len(x.shape) != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("Design must be a nonempty two-dimensional matrix.")
    if y.shape != (x.shape[0],) or w.shape != y.shape or p.shape != (x.shape[1],):
        raise ValueError("Targets, weights, or penalties have incompatible shapes.")
    values = x.data if sparse.issparse(x) else x
    if not all(np.isfinite(v).all() for v in (values, y, w, p)):
        raise ValueError("Inputs must be finite.")
    if np.any((y < 0) | (y > 1)) or np.any(w <= 0) or np.any(p < 0):
        raise ValueError("Targets must be in [0,1], weights positive, and penalties nonnegative.")
    return x, y, w, p


def objective_gradient(
    coefficients: Any, design: Any, targets: Any, weights: Any, penalties: Any,
) -> tuple[float, np.ndarray]:
    x, y, w, p = _validate(design, targets, weights, penalties)
    theta = np.asarray(coefficients, dtype=np.float64)
    if theta.shape != (x.shape[1],) or not np.isfinite(theta).all():
        raise ValueError("Coefficients must be finite and match the design width.")
    logits = np.asarray(x @ theta).reshape(-1)
    loss = y * np.logaddexp(0.0, -logits) + (1.0 - y) * np.logaddexp(0.0, logits)
    obj = float(np.dot(w, loss) / w.sum() + np.dot(p, theta * theta) / 2)
    gradient = np.asarray(x.T @ (w * (expit(logits) - y))).reshape(-1) / w.sum() + p * theta
    return obj, gradient


def fit_logistic(design: Any, targets: Any, penalties: Any, weights: Any = None) -> FitResult:
    n = design.shape[0]
    x, y, w, p = _validate(design, targets, np.ones(n) if weights is None else weights, penalties)
    attempts = []
    for maxiter, ftol in ((5000, 1e-12), (10000, 1e-14)):
        result = minimize(
            objective_gradient, np.zeros(x.shape[1], dtype=np.float64),
            args=(x, y, w, p), method="L-BFGS-B", jac=True,
            options={"maxiter": maxiter, "gtol": 1e-6, "ftol": ftol, "maxls": 50},
        )
        theta = np.asarray(result.x, dtype=np.float64)
        if theta.shape == (x.shape[1],) and np.isfinite(theta).all():
            obj, grad = objective_gradient(theta, x, y, w, p)
            logits = np.asarray(x @ theta).reshape(-1)
            maxgrad = float(np.max(np.abs(grad)))
            accepted = bool(
                np.isfinite(obj) and np.isfinite(logits).all() and np.isfinite(expit(logits)).all()
                and np.isfinite(maxgrad) and maxgrad < 1e-5
            )
        else:
            obj, maxgrad, accepted = float("nan"), float("nan"), False
        attempts.append({
            "maxiter": maxiter, "ftol": ftol, "solver_success": bool(result.success),
            "status": int(result.status), "message": str(result.message),
            "iterations": int(result.nit), "objective": obj, "gradient_max": maxgrad,
            "accepted": accepted,
        })
        if accepted:
            break
    return FitResult(theta, accepted, obj, maxgrad, int(result.status), str(result.message),
                     int(result.nit), tuple(attempts))


def context_mapping(groups: list[str], pair_ids: list[str]) -> tuple[str | None, tuple[str, ...]]:
    if len(groups) != len(pair_ids):
        raise ValueError("Each context must have a pair ID.")
    counts: dict[str, int] = {}
    pairs: dict[str, set[str]] = {}
    for group, pair in zip(groups, pair_ids):
        counts[group] = counts.get(group, 0) + 1
        pairs.setdefault(group, set()).add(pair)
    eligible = [g for g in counts if counts[g] >= 100 and len(pairs[g]) >= 30]
    if not eligible:
        return None, ()
    reference = min(eligible, key=lambda group: (-counts[group], group))
    return reference, tuple(sorted(g for g in eligible if g != reference))


def contextual_design(differences: Any, groups: list[str], interactions: tuple[str, ...]) -> np.ndarray:
    x = np.asarray(differences, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != len(groups) or len(set(interactions)) != len(interactions):
        raise ValueError("Invalid feature differences, contexts, or interaction categories.")
    blocks = [np.ones((len(groups), 1)), x]
    g = np.asarray(groups)
    blocks.extend(x * (g == category)[:, None] for category in interactions)
    return np.hstack(blocks)


def contextual_penalties(dimensions: int, interactions: int, ridge: float, multiplier: float) -> np.ndarray:
    if dimensions < 1 or interactions < 0 or ridge <= 0 or multiplier <= 0:
        raise ValueError("Invalid contextual penalty configuration.")
    return np.concatenate(([1e-6], np.full(dimensions, ridge),
                           np.full(dimensions * interactions, ridge * multiplier)))


def pooled_fallback(pooled_predictions: Any) -> np.ndarray:
    predictions = np.asarray(pooled_predictions, dtype=np.float64)
    if predictions.ndim != 1 or not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
        raise ValueError("Pooled predictions must be a finite probability vector.")
    return predictions.copy()
