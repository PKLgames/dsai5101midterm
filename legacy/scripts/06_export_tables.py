#!/usr/bin/env python3
"""Export ready-to-use train/test tables: image features + observer profile + choice.

One row per annotator response (24,896 train / 6,224 test).  Everything is derived
from the feature cache and the course CSVs, so a single run reproduces the tables:

    tables/train_pairs.csv.gz   features + profile + observed choice
    tables/test_pairs.csv.gz    same columns, labels removed
    tables/image_features.csv.gz  per-photo rows (used for the A/B diffs above)

Columns
-------
id, a_cat, b_cat, a_image, b_image, comparison_type
    row identity and the two photos
gender, pet_history, pet_care, pet_contact, future_pet, mbti_ie
    the observer profile exactly as recorded (missing -> "missing")
demo_<col>_<level>
    the 26 one-hot columns the models actually consume
d_0000 ... d_0575
    embedding difference f(A) - f(B) (L2-normalised MobileNetV3-Small by default)
h_<name>
    the 20 low-level statistics, also differenced
choice, y (train only)
    "A"/"B" and 1/0 for "chose A"

Usage
    python scripts/06_export_tables.py
    python scripts/06_export_tables.py --tag resnet18_224 --out-dir tables/resnet18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from model import features, index, paths  # noqa: E402


def profile_block(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw profile columns + the same columns one-hot encoded (missing level included)."""
    cols = paths.DEMO_COLS
    raw = df[cols].astype("string").fillna("missing").reset_index(drop=True)
    onehot = {}
    for c, levels in paths.CATEGORICAL_LEVELS.items():
        if c not in cols:
            continue
        vals = raw[c].where(raw[c].isin(levels), "missing")
        for lv in levels:
            onehot[f"demo_{c}_{lv}"] = (vals == lv).astype(np.int8).to_numpy()
    return raw, pd.DataFrame(onehot)


def pair_features(df: pd.DataFrame, feats: features.Features) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A-minus-B difference of the deep block and of the hand block."""
    row = {p: i for i, p in enumerate(feats.paths)}
    missing = [p for p in set(df["a_image"]) | set(df["b_image"]) if p not in row]
    if missing:
        raise KeyError(f"{len(missing)} images are not in the feature cache, e.g. {missing[:3]}")
    ia = [row[p] for p in df["a_image"]]
    ib = [row[p] for p in df["b_image"]]
    diff = feats.deep[ia] - feats.deep[ib]
    hand_diff = feats.hand[ia] - feats.hand[ib]

    deep = pd.DataFrame(diff, columns=[f"d_{i:04d}" for i in range(diff.shape[1])])
    names = list(feats.meta.get("hand_features", []))
    hand = pd.DataFrame(hand_diff, columns=[f"h_{n}" for n in names])
    return deep, hand


def image_table(feats: features.Features) -> pd.DataFrame:
    idx = pd.read_csv(paths.IMAGE_INDEX_CSV, encoding="utf-8-sig")
    cat_of = dict(zip(idx["image"].astype(str), idx["cat"].astype(str)))
    table = pd.DataFrame({"image_path": feats.paths.astype(str)})
    table["cat"] = table["image_path"].map(cat_of)
    deep = pd.DataFrame(feats.deep, columns=[f"deep_{i:04d}" for i in range(feats.deep.shape[1])])
    hand = pd.DataFrame(feats.hand, columns=[f"hand_{n}" for n in feats.meta.get("hand_features", [])])
    return pd.concat([table, deep, hand], axis=1)


def build(df: pd.DataFrame, feats, split: str) -> pd.DataFrame:
    id_col = "train_row_id" if split == "train" else "test_row_id"
    base = df[["a_cat", "b_cat", "a_image", "b_image", "comparison_type"]].reset_index(drop=True)
    deep, hand = pair_features(df, feats)
    raw, onehot = profile_block(df)
    out = pd.concat(
        [pd.DataFrame({id_col: df[id_col].to_numpy()}), base, raw, onehot, deep, hand], axis=1
    )
    if split == "train":
        out["choice"] = df["choice"].to_numpy()
        out["y"] = features.label_a(df)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="mobilenet_v3_small_224")
    ap.add_argument("--out-dir", default=str(paths.ROOT / "tables"))
    ap.add_argument("--no-images", action="store_true", help="skip the per-photo table")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = features.load_features(args.tag)
    print(f"tag={feats.meta.get('tag')}  deep {feats.deep.shape[1]}-d  hand {feats.hand.shape[1]}-d")

    for split, loader in (("train", index.load_train), ("test", index.load_test)):
        df = loader()
        table = build(df, feats, split)
        dest = out_dir / f"{split}_pairs.csv.gz"
        table.to_csv(dest, index=False, float_format="%.6g", compression="gzip")
        print(f"{split:5s}: {table.shape[0]:,} rows x {table.shape[1]} columns -> {dest} "
              f"({dest.stat().st_size / 1e6:.1f} MB)")

    if not args.no_images:
        images = image_table(feats)
        dest = out_dir / "image_features.csv.gz"
        images.to_csv(dest, index=False, float_format="%.6g", compression="gzip")
        print(f"images: {images.shape[0]:,} rows x {images.shape[1]} columns -> {dest} "
              f"({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
