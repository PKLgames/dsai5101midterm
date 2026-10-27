"""Frozen ImageNet backbones used as fixed feature extractors.

Why frozen: the training signal is pairwise preference (24,896 rows over 1,296
images) and the machine is CPU-only with 7 GB RAM.  Fine-tuning a backbone
would take hours and overfit 1.3k images, while a frozen generic embedding
already separates cats/appearance well (see ``scripts/04_feature_qa.py``).

Every factory returns ``(model, dim)`` where ``model`` maps a normalised
``(B, 3, H, W)`` batch to an L2-normalisable ``(B, dim)`` pooled embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

#: name -> (constructor, weights enum, feature dim, description)
BACKBONES: dict[str, tuple] = {
    "mobilenet_v3_small": (
        models.mobilenet_v3_small,
        models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        576,
        "MobileNetV3-Small, 2.5M params, ~0.06 GFLOPs @224 - default (fastest on CPU)",
    ),
    "resnet18": (
        models.resnet18,
        models.ResNet18_Weights.IMAGENET1K_V1,
        512,
        "ResNet18, 11.7M params, ~1.8 GFLOPs @224 - stronger, ~3x slower",
    ),
    "efficientnet_b0": (
        models.efficientnet_b0,
        models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        1280,
        "EfficientNet-B0, 5.3M params, ~0.4 GFLOPs @224 - widest embedding",
    ),
}


class _Pooled(nn.Module):
    """Run ``trunk`` then global-pool + flatten."""

    def __init__(self, trunk: nn.Module, pool: nn.Module | None = None) -> None:
        super().__init__()
        self.trunk = trunk
        self.pool = pool if pool is not None else nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trunk(x)
        if x.ndim == 4:
            x = self.pool(x)
        return torch.flatten(x, 1)


def build_backbone(name: str = "mobilenet_v3_small", pretrained: bool = True):
    """Return ``(eval model, feature dim)``.  Raises ``KeyError`` for unknown names."""
    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    ctor, weights_enum, dim, _ = BACKBONES[name]
    model = ctor(weights=weights_enum if pretrained else None)

    if name == "resnet18":
        trunk = nn.Sequential(*list(model.children())[:-1])  # drop fc -> (B,512,1,1)
        pooled = _Pooled(trunk, nn.Identity())
    else:  # mobilenet_v3_*, efficientnet_*
        pooled = _Pooled(model.features, model.avgpool)

    for p in pooled.parameters():
        p.requires_grad_(False)
    pooled.eval()
    return pooled, dim


def weights_tag(name: str) -> str:
    return BACKBONES[name][1].url.rsplit("/", 1)[-1]


def describe() -> str:
    return "\n".join(f"  {k:<22} {v[2]:>5}-d  {v[3]}" for k, v in BACKBONES.items())
