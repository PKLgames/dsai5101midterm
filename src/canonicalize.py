"""Pair coordinates and orientation helpers shared by later stages."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence


def canonical_pair(a_image: str, b_image: str) -> tuple[str, str, int]:
    if not a_image or not b_image or a_image == b_image:
        raise ValueError("A canonical pair requires two distinct image paths.")
    if a_image < b_image:
        return a_image, b_image, 1
    return b_image, a_image, -1


def pair_id(image_i: str, image_j: str) -> str:
    if not image_i or image_i >= image_j:
        raise ValueError("Pair endpoints must be distinct and in canonical order.")
    return json.dumps([image_i, image_j], ensure_ascii=False, separators=(",", ":"))


def canonical_outcome(choice: str, orientation: int) -> int:
    if choice not in {"A", "B"} or orientation not in {-1, 1}:
        raise ValueError("Choice must be A/B and orientation must be +1/-1.")
    y = int(choice == "A")
    return y if orientation == 1 else 1 - y


def displayed_logit(score_a: float, score_b: float, side_bias: float) -> float:
    return score_a - score_b + side_bias


def canonical_logit(score_i: float, score_j: float, orientation: int, side_bias: float) -> float:
    if orientation not in {-1, 1}:
        raise ValueError("Orientation must be +1 or -1.")
    return score_i - score_j + orientation * side_bias


def feature_difference(
    a_image: str, b_image: str, features: Mapping[str, Sequence[float]]
) -> list[float]:
    a, b = features[a_image], features[b_image]
    if len(a) != len(b):
        raise ValueError("Feature dimensions differ.")
    return [x - y for x, y in zip(a, b)]
