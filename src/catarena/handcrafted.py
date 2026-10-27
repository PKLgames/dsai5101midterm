"""Cheap, interpretable low-level image statistics.

These are deliberately not learned: they are the "simple features" a human can
read off a slide (brightness, sharpness, colourfulness, ...).  They cost almost
nothing next to the CNN forward pass and give the downstream preference model
interpretable dimensions to attribute to.

Everything is computed on a 256 px thumbnail of the *original* image (not the
224 centre crop) so file-level properties such as aspect ratio survive.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

HAND_NAMES: list[str] = [
    "brightness",
    "contrast",
    "saturation_mean",
    "saturation_std",
    "colorfulness",
    "sharpness_laplacian_var",
    "entropy_bits",
    "edge_density",
    "warm_pixel_ratio",
    "hue_mean",
    "hue_concentration",
    "mean_r",
    "mean_g",
    "mean_b",
    "aspect_ratio",
    "log_pixels",
    "overexposed_frac",
    "underexposed_frac",
    "is_effectively_gray",
    "log_file_kb",
]

_THUMB = 256


def _laplacian_var(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian (a standard blur/focus proxy)."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    c = gray[1:-1, 1:-1]
    lap = (
        4.0 * c
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    return float(lap.var())


def _edge_density(gray: np.ndarray, threshold: float = 0.12) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    gx = gray[1:-1, 2:] - gray[1:-1, :-2]
    gy = gray[2:, 1:-1] - gray[:-2, 1:-1]
    mag = np.sqrt(gx * gx + gy * gy)
    return float((mag > threshold).mean())


def _entropy(gray: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(gray, bins=bins, range=(0.0, 1.0))
    p = hist.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) colourfulness metric."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(
        math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean())
    )


def _hue_stats(hsv: np.ndarray, sat: np.ndarray) -> tuple[float, float]:
    """Weighted circular mean of hue and its resultant length in [0, 1]."""
    theta = hsv[..., 0].astype(np.float64) * (2.0 * math.pi / 255.0)
    w = sat.astype(np.float64)
    wsum = w.sum()
    if wsum <= 1e-9:
        return 0.0, 0.0
    cos_m = float((w * np.cos(theta)).sum() / wsum)
    sin_m = float((w * np.sin(theta)).sum() / wsum)
    angle = math.atan2(sin_m, cos_m) % (2.0 * math.pi)
    return angle / (2.0 * math.pi), float(math.hypot(cos_m, sin_m))


def compute(img: Image.Image, n_bytes: int = 0) -> np.ndarray:
    """Return the 20-dim handcrafted vector for one PIL RGB image."""
    width, height = img.size
    thumb = img.copy()
    thumb.thumbnail((_THUMB, _THUMB), Image.BICUBIC)

    rgb = np.asarray(thumb, dtype=np.float32) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    hsv = np.asarray(thumb.convert("HSV"), dtype=np.uint8)

    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0
    hue_mean, hue_conc = _hue_stats(hsv, sat)

    channel_spread = float(
        np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean()
    )

    feats = [
        float(gray.mean()),
        float(gray.std()),
        float(sat.mean()),
        float(sat.std()),
        _colorfulness(rgb),
        _laplacian_var(gray),
        _entropy(gray),
        _edge_density(gray),
        float((rgb[..., 0] - rgb[..., 2] > 0.04).mean()),
        hue_mean,
        hue_conc,
        float(rgb[..., 0].mean()),
        float(rgb[..., 1].mean()),
        float(rgb[..., 2].mean()),
        width / max(height, 1),
        math.log(max(width * height, 1)),
        float((val > 0.98).mean()),
        float((val < 0.02).mean()),
        1.0 if channel_spread < 0.01 else 0.0,
        math.log1p(max(n_bytes, 1) / 1024.0),
    ]
    out = np.asarray(feats, dtype=np.float32)
    assert out.shape == (len(HAND_NAMES),), out.shape
    return out
