"""Project-wide relative paths.

Every path is derived from this file's location, so the package can be moved
and the scripts can be launched from any working directory.
"""

from __future__ import annotations

from pathlib import Path

# dsai5101midterm/src/model/paths.py -> dsai5101midterm/
ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "course"
TRAIN_CSV = DATASET_DIR / "train.csv"
TEST_CSV = DATASET_DIR / "test.csv"
TEST_EXAMPLE_CSV = DATASET_DIR / "test_example.csv"

DATA_DIR = ROOT / "artifacts"
IMAGES_DIR = DATA_DIR / "images"
FEATURES_DIR = DATA_DIR / "features"
SUBMISSIONS_DIR = DATA_DIR / "submissions"
DOCS_DIR = ROOT / "docs"

# --------------------------------------------------------------------------- #
# YOLO11n-seg + CLIP photo-label table.
# Written by ``photo-labelling/label_photos.py``.
# Everything here is read-only for the preference model.
# --------------------------------------------------------------------------- #
LABELS_DIR = DATA_DIR / "labels"
PHOTO_LABELS_CSV = LABELS_DIR / "photo_labels.csv"
CAT_LABELS_CSV = LABELS_DIR / "cat_labels.csv"

MANIFEST_CSV = DATA_DIR / "manifest.csv"
IMAGE_INDEX_CSV = DATA_DIR / "image_index.csv"
DOWNLOAD_ERRORS_CSV = DATA_DIR / "download_errors.csv"

# Hugging Face dataset that hosts the images (see course/README.md).
HF_BASE = (
    "https://huggingface.co/datasets/missing12/cat-image-arena-images/resolve/main"
)

# Demographic columns of the annotator, in the order used everywhere.
DEMO_COLS = [
    "gender",
    "pet_history",
    "pet_care",
    "pet_contact",
    "future_pet",
    "mbti_ie",
]

CATEGORICAL_LEVELS = {
    "comparison_type": ["within_cat", "between_cats"],
    "gender": ["man", "woman", "missing"],
    "pet_history": ["never", "past", "current", "missing"],
    "pet_care": ["never", "occasionally", "past_regular", "current_regular", "missing"],
    "pet_contact": ["never", "rarely", "sometimes", "often", "daily", "missing"],
    "future_pet": ["no", "unsure", "yes", "missing"],
    "mbti_ie": ["I", "E", "unknown", "missing"],
}


def ensure_dirs() -> None:
    """Create the runtime data directories if they do not exist."""
    for d in (IMAGES_DIR, FEATURES_DIR, SUBMISSIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)
