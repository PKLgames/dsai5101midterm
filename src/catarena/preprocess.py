"""Deterministic, inference-only preprocessing.

No random augmentation is applied: the feature cache must be bit-reproducible
for a given torch/torchvision version, otherwise the reported results in the
report could not be regenerated.
"""

from __future__ import annotations

from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Shorter side is resized to ``resize`` then centre-cropped to ``size``.
#: 224 is the ImageNet training resolution of every backbone we use; going
#: larger costs ~2x CPU time for a negligible accuracy gain on this task.
def build_transform(size: int = 224, resize: int | None = None) -> transforms.Compose:
    resize = resize or int(round(size * 232 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_image(path) -> Image.Image:
    """Open an image, fix EXIF orientation, and return an RGB copy."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")
