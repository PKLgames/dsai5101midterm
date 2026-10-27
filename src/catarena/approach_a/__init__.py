"""Components copied or adapted from the comparison project (approach A).

Provenance
----------
Everything in this subpackage comes from ``<workspace>/dsai-5101-sdm-proj`` (the
project described by ``README_allie_chan.md``).  It is kept in its own
subpackage so the boundary stays visible, and it contains **only reusable
pieces** — no code that re-runs that project's analysis:

    approach_a/strength.py   per-item strengths (cat / photo) + side term,
                             adapted from its ``fit_bt`` / ``predict_bt``
    approach_a/labels.py     readers + encoders for its YOLO/CLIP photo-label
                             table (``prepared/photo_labels.csv``)
    approach_a/metrics.py    its log_loss / brier / accuracy definitions

Attribution for the label values themselves: they are produced by that
project's ``photo-labelling/*.py`` (YOLO11n-seg masks + CLIP zero-shot labels).
This project only reads the resulting table.

Nothing is imported at run time except that label table, and nothing here
writes to the other repository.  Everything under ``catarena/`` outside this
subpackage — backbone choice, preprocessing, 576-d embedding, 20 hand features,
pair encodings, the hybrid model, the submission pipeline — is original to this
project.
"""

from . import labels, metrics, strength  # noqa: F401

__all__ = ["labels", "metrics", "strength"]
