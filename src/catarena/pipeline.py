"""One-call image -> feature vector, mirroring what ``scripts/03`` bakes into the cache.

The batch pipeline (`catarena.extract`) is the fast path for the whole dataset; this
module exists for the small cases: embedding a handful of new photos, sanity-checking
that a checkpoint gives the same numbers as the cache, or using the features from a
notebook without touching the cache format.

Example
-------
>>> from catarena import pipeline
>>> out = pipeline.image_to_features("data/images/cat_000002/cat_000002_0001.jpg")
>>> out["deep"].shape, out["hand"].shape
((576,), (20,))
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import backbones, handcrafted, preprocess

DEFAULT_BACKBONE = "mobilenet_v3_small"
DEFAULT_SIZE = 224


def image_to_features(
    path: str | Path,
    backbone: str = DEFAULT_BACKBONE,
    size: int = DEFAULT_SIZE,
    tta_flip: bool = False,
):
    """Embed one image.  Returns ``{"deep", "hand", "hand_names", ...}``.

    ``deep`` is the L2-normalised pooled embedding (576-d for the default
    backbone); ``hand`` is the 20-dim interpretable statistics vector.  This is
    exactly the pair of blocks stored in ``data/features/<tag>.npz``.
    """
    return images_to_features([path], backbone=backbone, size=size, tta_flip=tta_flip)[0]


def images_to_features(
    paths,
    backbone: str = DEFAULT_BACKBONE,
    size: int = DEFAULT_SIZE,
    tta_flip: bool = False,
    batch_size: int = 16,
):
    """Embed a list of images, loading the backbone once."""
    paths = [Path(p) for p in paths]
    model, _dim = backbones.build_backbone(backbone)
    transform = preprocess.build_transform(size)
    torch.set_grad_enabled(False)

    out = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        tensors, hands = [], []
        for p in chunk:
            if not p.is_file():
                raise FileNotFoundError(f"{p} does not exist")
            img = preprocess.load_image(p)
            hands.append(handcrafted.compute(img, os.path.getsize(p)))
            tensors.append(transform(img))
        x = torch.stack(tensors)
        with torch.inference_mode():
            feat = model(x)
            if tta_flip:
                feat = 0.5 * (feat + model(torch.flip(x, dims=[3])))
            feat = F.normalize(feat, p=2, dim=1)
        feat = feat.numpy()
        for i, p in enumerate(chunk):
            out.append(
                {
                    "path": str(p),
                    "deep": feat[i].astype(np.float32),
                    "hand": hands[i].astype(np.float32),
                    "hand_names": list(handcrafted.HAND_NAMES),
                    "backbone": backbone,
                    "size": size,
                }
            )
    return out


def image_to_table(paths, backbone: str = DEFAULT_BACKBONE, size: int = DEFAULT_SIZE):
    """Convenience: list of images -> DataFrame with ``deep_*`` and ``hand_*`` columns."""
    import pandas as pd

    rows = images_to_features(paths, backbone=backbone, size=size)
    frame = pd.DataFrame(
        {"image_path": [r["path"] for r in rows]},
    )
    deep = np.stack([r["deep"] for r in rows])
    hand = np.stack([r["hand"] for r in rows])
    deep_cols = [f"deep_{i:04d}" for i in range(deep.shape[1])]
    hand_cols = [f"hand_{n}" for n in rows[0]["hand_names"]]
    return pd.concat(
        [frame, pd.DataFrame(deep, columns=deep_cols), pd.DataFrame(hand, columns=hand_cols)],
        axis=1,
    )
