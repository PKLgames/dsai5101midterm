"""Per-item strengths, photo-label encoders, and scoring helpers.

Provenance
----------
These helpers are the cat-strength fit and the photo-label encoders.
Labels come from ``photo-labelling/``.  The
subpackage stays separate so the preference-model
boundary stays visible.  It contains only the reusable pieces:

    labels/strength.py   per-item strengths (cat / photo) + side term,
                         adapted from ``fit_bt`` / ``predict_bt``
    labels/labels.py     readers + encoders for the YOLO/CLIP photo-label
                         table (``artifacts/labels/photo_labels.csv``)
    labels/metrics.py    log_loss / brier / accuracy definitions

The label values themselves come from ``photo-labelling/*.py`` (YOLO11n-seg
masks + CLIP zero-shot labels).  This subpackage only reads the resulting table.

Everything under ``model/`` outside this subpackage — backbone choice,
preprocessing, 576-d embedding, 20 hand features, pair encodings, the hybrid
model, the submission pipeline — is the feature-pipeline code.
"""

from . import labels, metrics, strength  # noqa: F401

__all__ = ["labels", "metrics", "strength"]
