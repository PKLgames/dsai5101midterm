#!/usr/bin/env python3
"""Extract frozen-CNN + low-level features for every image and cache them.

Examples
--------
    # default: MobileNetV3-Small @224, 576-d, ~1-3 min on a 16-thread laptop CPU
    python scripts/03_extract_features.py

    # stronger / slower alternatives
    python scripts/03_extract_features.py --backbone resnet18
    python scripts/03_extract_features.py --backbone efficientnet_b0 --tta-flip

    # quick smoke test on 32 images
    python scripts/03_extract_features.py --limit 32 --workers 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catarena import backbones, extract  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="mobilenet_v3_small")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=None, help="torch intra-op threads (default min(8, cores))")
    ap.add_argument("--tta-flip", action="store_true", help="average original + horizontal flip (2x cost)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (smoke test)")
    ap.add_argument("--tag", default=None, help="cache name (default: <backbone>_<size>[_tta])")
    ap.add_argument("--write-csv", action="store_true",
                    help="also write <tag>.csv (per-photo table; scripts/06 exports the curated one)")
    ap.add_argument("--list-backbones", action="store_true")
    args = ap.parse_args()

    if args.list_backbones:
        print("available backbones:\n" + backbones.describe())
        return

    cfg = extract.ExtractConfig(
        backbone=args.backbone,
        size=args.size,
        batch_size=args.batch_size,
        workers=args.workers,
        threads=args.threads,
        tta_flip=args.tta_flip,
        limit=args.limit,
        tag=args.tag,
        write_csv=args.write_csv,
    )
    print(f"extracting with {cfg.backbone} @ {cfg.size}px -> tag {cfg.resolved_tag()}")
    extract.extract(cfg)


if __name__ == "__main__":
    main()
