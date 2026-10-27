#!/usr/bin/env python3
"""Download every image referenced by train.csv / test.csv from Hugging Face."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catarena import download  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=16, help="parallel HTTP connections")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (smoke test)")
    args = ap.parse_args()
    download.download_all(workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
