#!/usr/bin/env python3
"""Build artifacts/image_index.csv: one row per image with cat, split and hit counts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from model import index, paths  # noqa: E402


def main() -> None:
    idx = index.build_image_index(save=True)
    train, test = index.load_train(), index.load_test()

    tri = set(train["a_image"]) | set(train["b_image"])
    tei = set(test["a_image"]) | set(test["b_image"])
    trc = set(train["a_cat"]) | set(train["b_cat"])
    tec = set(test["a_cat"]) | set(test["b_cat"])
    pk = index.pair_key
    pair_overlap = len(set(pk(train)) & set(pk(test)))

    print(f"images          : {len(idx)}  (train {len(tri)}, test {len(tei)}, shared {len(tri & tei)})")
    print(f"cats            : {len(trc | tec)}  (train {len(trc)}, test {len(tec)}, test-only {len(tec - trc)})")
    print(f"test images never seen in train : {len(tei - tri)}")
    print(f"train/test image-pair overlap   : {pair_overlap} / {len(test)}")
    print(f"files present on disk           : {int(idx['file_exists'].sum())} / {len(idx)}")
    print(f"split counts                    : {idx['split'].value_counts().to_dict()}")
    print(f"\nwritten -> {paths.IMAGE_INDEX_CSV}")

    if not idx["file_exists"].all():
        missing = idx.loc[~idx["file_exists"], "image"].tolist()
        print(f"\n!! {len(missing)} images missing, e.g. {missing[:5]}")
        print("   run scripts/01_download_images.py")
        sys.exit(1)

    print("\nrows per cat (train):", end=" ")
    print(pd.Series(idx["cat"]).value_counts().describe()[["mean", "min", "max"]].round(1).to_dict())


if __name__ == "__main__":
    main()
