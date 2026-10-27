#!/usr/bin/env python3
"""Smallest possible "image -> features" demo, plus a cached-features example.

    python scripts/10_quickstart.py                      # 1 photo through the model
    python scripts/10_quickstart.py --image path/to.jpg  # your own photo
    python scripts/10_quickstart.py --batch 4            # embed several photos

What it shows
-------------
1. one photo -> 576-d deep embedding + 20 hand statistics (no cache involved);
2. the same numbers are what `data/features/<tag>.npz` already holds;
3. the cached features feed a pair row of 604 columns (embedding diff +
   26 demographic one-hot + 2 comparison types) exactly as the models use them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from catarena import features, index, paths, pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=None, help="photo to embed (default: a sample from data/images)")
    ap.add_argument("--batch", type=int, default=1, help="embed this many sample photos")
    ap.add_argument("--backbone", default=pipeline.DEFAULT_BACKBONE)
    ap.add_argument("--tag", default="mobilenet_v3_small_224", help="cache to compare against")
    args = ap.parse_args()

    if args.image:
        images = [Path(args.image)]
    else:
        images = sorted(paths.IMAGES_DIR.glob("cat_*/*.jpg"))[: max(1, args.batch)]
    print(f"embedding {len(images)} image(s) with {args.backbone}\n")

    rows = pipeline.images_to_features(images, backbone=args.backbone)
    for r in rows:
        print(f"{Path(r['path']).name}: deep {r['deep'].shape} (L2 norm {np.linalg.norm(r['deep']):.3f}), "
              f"hand {r['hand'].shape}")
    print("\nfirst hand features:",
          {n: round(float(v), 3) for n, v in zip(rows[0]["hand_names"][:5], rows[0]["hand"][:5])})

    # ---- same numbers come out of the cache --------------------------------- #
    try:
        feats = features.load_features(args.tag)
    except FileNotFoundError:
        print("\n(no feature cache yet - run scripts/03_extract_features.py to compare)")
        return
    cached = feats.lookup([str(p.relative_to(paths.IMAGES_DIR)) for p in images])[0]
    diff = np.abs(cached - np.stack([r["deep"] for r in rows])).max()
    print(f"\ncache check: max |cache - fresh| = {diff:.2e} (0 means the cache is current)")

    # ---- what a model row looks like --------------------------------------- #
    train = index.load_train().head(1)
    X, pa, pb = features.pair_matrix(train, feats, mode="diff", with_demo=True)
    print(f"one training row -> X.shape {X.shape} "
          f"(576 embedding diff + 26 demographics + 2 comparison types), label={int(features.label_a(train)[0])}")


if __name__ == "__main__":
    main()
