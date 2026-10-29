"""Sanity checks that the frozen embeddings actually capture cat appearance.

Before any preference model is fitted we verify the features are informative:

1. cosine similarity of same-cat image pairs should exceed different-cat pairs;
2. a 5-NN cat-retrieval accuracy well above chance;
3. a PCA projection should show same-cat images nearby;
4. a nearest-neighbour image grid is a human-checkable version of (3).

If (1)-(2) hold, the pair model has a real signal to exploit; if they fail, the
problem is the extractor, not the classifier.
"""

from __future__ import annotations

import json
import textwrap

import numpy as np
import pandas as pd

from . import paths

CAT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _cat_labels(feats_paths: np.ndarray) -> np.ndarray:
    idx = pd.read_csv(paths.IMAGE_INDEX_CSV, encoding="utf-8-sig")
    m = dict(zip(idx["image"].astype(str), idx["cat"].astype(str)))
    return np.asarray([m.get(p, "unknown") for p in feats_paths])


def similarity_stats(feats):
    """Cosine similarity structure by cat membership (full 1558x1558 matrix)."""
    x = feats.deep
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    sim = x @ x.T
    cats = _cat_labels(feats.paths)
    n = len(cats)
    iu = np.triu_indices(n, k=1)
    s = sim[iu]
    same = (cats[:, None] == cats[None, :])[iu]

    within, between = s[same], s[~same]

    # 5-NN cat retrieval: share of neighbours that belong to the same cat
    np.fill_diagonal(sim, -np.inf)
    k = 5
    nn = np.argpartition(-sim, k, axis=1)[:, :k]
    knn_acc = float((cats[nn] == cats[:, None]).mean())
    chance = float(((np.bincount(pd.factorize(cats)[0]) / n) ** 2).sum())

    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(same.astype(np.int8), s))
    except Exception:  # noqa: BLE001
        auc = float("nan")

    return {
        "n_images": int(n),
        "n_cats": int(len(set(cats))),
        "n_pairs": int(len(s)),
        "within_cat": {
            "mean": float(within.mean()),
            "std": float(within.std()),
            "p10": float(np.percentile(within, 10)),
            "p90": float(np.percentile(within, 90)),
        },
        "between_cat": {
            "mean": float(between.mean()),
            "std": float(between.std()),
            "p10": float(np.percentile(between, 10)),
            "p90": float(np.percentile(between, 90)),
        },
        "gap_mean": float(within.mean() - between.mean()),
        "same_cat_auc": auc,
        "knn5_cat_accuracy": knn_acc,
        "knn5_chance_level": chance,
        "within_gt_between": bool(within.mean() > between.mean()),
        "hist": _histograms(within, between),
        "_sim": sim,
        "_cats": cats,
    }


def _histograms(within: np.ndarray, between: np.ndarray, bins: int = 60) -> dict:
    """Density histograms for the diagnostic plot (kept in the JSON too)."""
    lo, hi = -0.2, 1.0
    w, e = np.histogram(within, bins=bins, range=(lo, hi), density=True)
    b, _ = np.histogram(between, bins=bins, range=(lo, hi), density=True)
    return {
        "bin_edges": [round(float(x), 4) for x in e],
        "same_cat_density": [round(float(x), 5) for x in w],
        "different_cat_density": [round(float(x), 5) for x in b],
    }


def _plot_hist(stats: dict, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.asarray(stats["hist"]["bin_edges"])
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.figure(figsize=(7.5, 4.2))
    plt.plot(centers, stats["hist"]["different_cat_density"], color="#d62728", lw=2,
             label=f"different cats (mean={stats['between_cat']['mean']:.3f})")
    plt.fill_between(centers, stats["hist"]["different_cat_density"], color="#d62728", alpha=0.15)
    plt.plot(centers, stats["hist"]["same_cat_density"], color="#2ca02c", lw=2,
             label=f"same cat (mean={stats['within_cat']['mean']:.3f})")
    plt.fill_between(centers, stats["hist"]["same_cat_density"], color="#2ca02c", alpha=0.2)
    plt.xlabel("cosine similarity of frozen ImageNet embeddings")
    plt.ylabel("density")
    plt.title(
        f"Same-cat vs different-cat similarity | gap {stats['gap_mean']:+.3f} | "
        f"AUC {stats['same_cat_auc']:.3f} | 5-NN cat acc {stats['knn5_cat_accuracy']:.1%}"
    )
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def _plot_pca(feats, stats, out, n_show: int = 10):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    x = feats.deep
    xy = PCA(n_components=2, random_state=0).fit_transform(x)
    cats = stats["_cats"]
    counts = pd.Series(cats).value_counts()
    top = list(counts.index[:n_show])

    plt.figure(figsize=(7.5, 6))
    rest = ~np.isin(cats, top)
    plt.scatter(xy[rest, 0], xy[rest, 1], s=6, c="#cccccc", alpha=0.5, label="other cats")
    for i, cat in enumerate(top):
        m = cats == cat
        plt.scatter(xy[m, 0], xy[m, 1], s=22, c=CAT_COLORS[i % len(CAT_COLORS)],
                    alpha=0.9, label=cat, edgecolors="white", linewidths=0.3)
    plt.title("PCA of frozen MobileNetV3-Small embeddings (top cats highlighted)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(fontsize=7, markerscale=1.2, loc="best", ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def _plot_neighbors(feats, stats, out, n_queries: int = 5, k: int = 5, seed: int = 0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rng = np.random.default_rng(seed)
    ok_idx = np.flatnonzero(feats.ok)
    sim = stats["_sim"].copy()
    np.fill_diagonal(sim, -np.inf)

    queries = rng.choice(ok_idx, size=min(n_queries, len(ok_idx)), replace=False)
    fig, axes = plt.subplots(len(queries), k + 1, figsize=(2.2 * (k + 1), 2.4 * len(queries)))
    axes = np.atleast_2d(axes)
    cats = stats["_cats"]

    for r, q in enumerate(queries):
        order = np.argsort(-sim[q])[:k]
        for c, j in enumerate([q] + list(order)):
            ax = axes[r, c]
            try:
                im = Image.open(paths.IMAGES_DIR / feats.paths[j])
                im.thumbnail((160, 160))
                ax.imshow(im)
            except Exception:  # noqa: BLE001
                ax.text(0.5, 0.5, "n/a", ha="center")
            tag = "query" if c == 0 else f"nn{c}"
            val = 1.0 if c == 0 else sim[q, j]
            ax.set_title(f"{tag}: {cats[j]}\nsim={val:.3f}", fontsize=7)
            ax.axis("off")
    fig.suptitle("Nearest neighbours in embedding space (query + 5 most similar)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def run(tag: str | None = None, out_dir=None, make_plots: bool = True) -> dict:
    from . import features as featmod

    out_dir = out_dir or paths.FEATURES_DIR
    feats = featmod.load_features(tag, kind="deep")
    tag = feats.meta.get("tag", tag)

    print(f"QA on tag={tag}  n={feats.n}  dim={feats.deep.shape[1]}  ok={feats.ok.sum()}")
    stats = similarity_stats(feats)

    sim = stats.pop("_sim")
    cats = stats.pop("_cats")
    stats["tag"] = tag
    stats["hand_features_checked"] = [
        {"name": n, "mean": float(feats.hand[feats.ok, i].mean()),
         "std": float(feats.hand[feats.ok, i].std()),
         "nan": int(np.isnan(feats.hand[:, i]).sum())}
        for i, n in enumerate(feats.meta.get("hand_features", []))
    ]
    stats["deep_nan"] = int(np.isnan(feats.deep).sum())
    stats["deep_norm_mean"] = float(np.linalg.norm(feats.deep[feats.ok], axis=1).mean())

    if make_plots:
        _plot_hist(stats, out_dir / f"qa_similarity_{tag}.png")
        _plot_pca(feats, {"_cats": cats}, out_dir / f"qa_pca_{tag}.png")
        _plot_neighbors(feats, {"_sim": sim, "_cats": cats}, out_dir / f"qa_neighbors_{tag}.png")

    (out_dir / f"qa_summary_{tag}.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(textwrap.dedent(f"""
    similarity  within-cat  {stats['within_cat']['mean']:+.3f}
                between-cat {stats['between_cat']['mean']:+.3f}
                gap         {stats['gap_mean']:+.3f}   same-cat AUC {stats['same_cat_auc']:.3f}
    5-NN cat retrieval {stats['knn5_cat_accuracy']:.2%} (chance {stats['knn5_chance_level']:.2%})
    deep NaN count {stats['deep_nan']}, mean L2 norm {stats['deep_norm_mean']:.3f}
    """).strip())
    print(f"summary -> {out_dir / f'qa_summary_{tag}.json'}")
    return stats
