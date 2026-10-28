#!/usr/bin/env python3
"""Check that the embeddings carry real signal before fitting any preference model.

Writes artifacts/features/qa_summary_<tag>.json plus three PNG diagnostics:
    qa_similarity_<tag>.png   same-cat vs different-cat cosine similarity
    qa_pca_<tag>.png          2-D PCA, top cats highlighted
    qa_neighbors_<tag>.png    query image + its 5 nearest neighbours
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import features, qa  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None, help="feature cache (default: most recent)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if args.tag is None:
        print("available caches:", features.available_tags())
    stats = qa.run(tag=args.tag, make_plots=not args.no_plots)
    ok = stats["within_gt_between"] and stats["knn5_cat_accuracy"] > stats["knn5_chance_level"]
    print("\nQA verdict:", "PASS - features separate cats" if ok else "WARN - features look weak")


if __name__ == "__main__":
    main()
