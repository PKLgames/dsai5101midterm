"""Read the CSV tables and build the image index / pair tables.

The dataset stores paths relative to the ``images/`` root, e.g.
``cat_000002/cat_000002_0001.jpg``.  Everything downstream works with those
relative keys, never with absolute paths, so caches stay portable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import paths


def read_csv(path: Path) -> pd.DataFrame:
    """Read one of the competition CSV files (BOM-prefixed UTF-8)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    for col in ("a_cat", "b_cat", "a_image", "b_image", "comparison_type"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    return df


def load_train() -> pd.DataFrame:
    return read_csv(paths.TRAIN_CSV)


def load_test() -> pd.DataFrame:
    return read_csv(paths.TEST_CSV)


def assign_split(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Per-image table: which split(s) an image appears in and how often."""
    tr_a = train[["a_image"]].rename(columns={"a_image": "image"})
    tr_b = train[["b_image"]].rename(columns={"b_image": "image"})
    te_a = test[["a_image"]].rename(columns={"a_image": "image"})
    te_b = test[["b_image"]].rename(columns={"b_image": "image"})

    tr_counts = pd.concat([tr_a, tr_b]).groupby("image").size().rename("n_train")
    te_counts = pd.concat([te_a, te_b]).groupby("image").size().rename("n_test")

    cats = pd.concat(
        [
            train[["a_image", "a_cat"]].rename(columns={"a_image": "image", "a_cat": "cat"}),
            train[["b_image", "b_cat"]].rename(columns={"b_image": "image", "b_cat": "cat"}),
            test[["a_image", "a_cat"]].rename(columns={"a_image": "image", "a_cat": "cat"}),
            test[["b_image", "b_cat"]].rename(columns={"b_image": "image", "b_cat": "cat"}),
        ]
    ).drop_duplicates("image")

    idx = (
        cats.set_index("image")
        .join(tr_counts, how="outer")
        .join(te_counts, how="outer")
        .fillna({"n_train": 0, "n_test": 0})
    )
    idx["n_train"] = idx["n_train"].astype(int)
    idx["n_test"] = idx["n_test"].astype(int)
    idx["in_train"] = idx["n_train"] > 0
    idx["in_test"] = idx["n_test"] > 0
    idx["split"] = "both"
    idx.loc[idx["in_train"] & ~idx["in_test"], "split"] = "train"
    idx.loc[~idx["in_train"] & idx["in_test"], "split"] = "test"
    return idx.reset_index().sort_values("image").reset_index(drop=True)


def build_image_index(save: bool = True) -> pd.DataFrame:
    train, test = load_train(), load_test()
    idx = assign_split(train, test)

    missing = [
        p for p in idx["image"] if not (paths.IMAGES_DIR / p).is_file()
    ]
    idx["file_exists"] = ~idx["image"].isin(missing)
    if save:
        paths.ensure_dirs()
        idx.to_csv(paths.IMAGE_INDEX_CSV, index=False)
    return idx


def pair_key(df: pd.DataFrame) -> pd.Series:
    """Unordered image-pair key, identical for (A,B) and (B,A)."""
    a = df["a_image"].to_numpy()
    b = df["b_image"].to_numpy()
    return pd.Series([f"{x}|{y}" if x <= y else f"{y}|{x}" for x, y in zip(a, b)])


if __name__ == "__main__":  # manual smoke test
    i = build_image_index()
    print(i.shape)
    print(i["split"].value_counts())
    print("file_exists:", i["file_exists"].sum(), "/", len(i))
