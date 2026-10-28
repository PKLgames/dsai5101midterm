"""Scoring helpers so log loss, Brier score, and accuracy stay comparable.

The clipping bound (1e-6) and the accuracy rule (``p >= 0.5`` -> A) match
the cat-strength fit.
"""

from __future__ import annotations

import numpy as np


def log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def accuracy(y, p) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean((np.asarray(p) >= 0.5) == (y == 1)))


def metrics(y, p) -> dict:
    return {
        "n": int(len(y)),
        "logloss": round(log_loss(y, p), 5),
        "brier": round(brier(y, p), 5),
        "accuracy": round(accuracy(y, p), 5),
    }
