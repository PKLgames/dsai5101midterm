"""Batch feature extraction on CPU with an on-disk cache.

Pipeline (one image, one pass):
    JPEG -> exif fix -> RGB -> 232 resize -> 224 centre crop -> ImageNet norm
         -> frozen backbone -> 576/512/1280-d pooled embedding -> L2 norm
    and in parallel: 256 px thumbnail -> 20 interpretable low-level statistics

The result is a single ``.npz`` keyed by the CSV image path, so the downstream
preference model never touches pixels again (fast iteration, reproducible).
"""

from __future__ import annotations

import json
import os
import platform
import resource
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import backbones, handcrafted, paths, preprocess

#: Bump when the preprocessing changes so stale caches are not silently reused.
FEATURE_SPEC_VERSION = 1


@dataclass
class ExtractConfig:
    backbone: str = "mobilenet_v3_small"
    size: int = 224
    resize: int | None = None
    batch_size: int = 32
    workers: int = 4
    tta_flip: bool = False
    threads: int | None = None
    limit: int | None = None
    tag: str | None = None
    write_csv: bool = False

    def resolved_tag(self) -> str:
        tag = self.tag or f"{self.backbone}_{self.size}"
        return tag + ("_tta" if self.tta_flip else "")


class _ImageDataset(Dataset):
    """Returns (tensor, handcrafted, ok). Never raises on a bad image."""

    def __init__(self, items: list[tuple[str, str]], transform, size: int) -> None:
        self.items = items
        self.transform = transform
        self.size = size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        rel, abspath = self.items[i]
        try:
            img = preprocess.load_image(abspath)
            hand = handcrafted.compute(img, os.path.getsize(abspath))
            x = self.transform(img)
            return x, torch.from_numpy(hand), True
        except Exception:  # noqa: BLE001 - corrupt/truncated image
            return (
                torch.zeros(3, self.size, self.size),
                torch.zeros(len(handcrafted.HAND_NAMES)),
                False,
            )


def _worker_init(_worker_id: int) -> None:
    # Image decoding is single-threaded here; torch threads are for the main
    # process only, otherwise 4 workers x 8 threads oversubscribe 16 cores.
    torch.set_num_threads(1)


def _workers_available() -> bool:
    """True when this process may create the semaphores DataLoader workers need."""
    try:
        import multiprocessing as mp

        q = mp.get_context("fork").Queue(1)
        q.close()
        return True
    except (PermissionError, OSError, ImportError):
        return False


def _items(limit: int | None) -> list[tuple[str, str]]:
    idx = pd.read_csv(paths.IMAGE_INDEX_CSV, encoding="utf-8-sig")
    rel = sorted(idx["image"].astype(str))
    if limit:
        rel = rel[:limit]
    items = []
    for r in rel:
        p = paths.IMAGES_DIR / r
        if not p.is_file():
            raise FileNotFoundError(
                f"missing image {p} - run scripts/01_download_images.py first"
            )
        items.append((r, str(p)))
    return items


def extract(cfg: ExtractConfig | None = None, progress: bool = True) -> dict:
    """Run extraction and write ``artifacts/features/<tag>.npz`` + ``.meta.json``."""
    cfg = cfg or ExtractConfig()
    paths.ensure_dirs()
    tag = cfg.resolved_tag()
    threads = cfg.threads or min(8, os.cpu_count() or 1)
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    items = _items(cfg.limit)
    n = len(items)
    model, dim = backbones.build_backbone(cfg.backbone)
    transform = preprocess.build_transform(cfg.size, cfg.resize)

    deep = np.zeros((n, dim), dtype=np.float32)
    hand = np.zeros((n, len(handcrafted.HAND_NAMES)), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)

    dataset = _ImageDataset(items, transform, cfg.size)

    def make_loader(workers: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=False,
            worker_init_fn=_worker_init if workers else None,
            drop_last=False,
        )

    workers_requested = cfg.workers
    if workers_requested and not _workers_available():
        # Some sandboxes deny the POSIX semaphores worker processes need;
        # decoding in-process still gives ~40 img/s here, which is plenty.
        print(
            f"  ! cannot spawn {workers_requested} dataloader workers in this "
            "environment; falling back to num_workers=0",
            flush=True,
        )
        cfg.workers = 0

    loader = make_loader(cfg.workers)
    batches = 0
    t0 = time.time()
    first_batch_done = None
    with torch.inference_mode():
        for x, h, good in loader:
            f = model(x)
            if cfg.tta_flip:
                f = 0.5 * (f + model(torch.flip(x, dims=[3])))
            f = F.normalize(f, p=2, dim=1)
            s = batches * cfg.batch_size
            e = s + x.shape[0]
            deep[s:e] = f.numpy()
            hand[s:e] = h.numpy()
            ok[s:e] = good.numpy()
            batches += 1
            if first_batch_done is None:
                first_batch_done = time.time()
            if progress and (batches % 10 == 0 or e == n):
                el = time.time() - t0
                print(
                    f"  {e:5d}/{n}  {el:6.1f}s  {e / el:6.1f} img/s",
                    flush=True,
                )

    elapsed = time.time() - t0
    # exclude one-off model load / first batch warm-up from the steady-state rate
    steady = elapsed - (first_batch_done - t0) if first_batch_done else elapsed
    steady_rate = (n - cfg.batch_size) / steady if steady > 0 and n > cfg.batch_size else n / max(elapsed, 1e-9)
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    np.savez_compressed(
        paths.FEATURES_DIR / f"{tag}.npz",
        paths=np.asarray([r for r, _ in items]),
        deep=deep,
        hand=hand,
        ok=ok,
        hand_names=np.asarray(handcrafted.HAND_NAMES),
    )

    meta = {
        "tag": tag,
        "spec_version": FEATURE_SPEC_VERSION,
        "backbone": cfg.backbone,
        "backbone_weights": backbones.weights_tag(cfg.backbone),
        "backbone_description": backbones.BACKBONES[cfg.backbone][3],
        "feature_dim_deep": int(dim),
        "feature_dim_hand": len(handcrafted.HAND_NAMES),
        "hand_features": list(handcrafted.HAND_NAMES),
        "input_size": cfg.size,
        "resize": cfg.resize or int(round(cfg.size * 232 / 224)),
        "normalize_mean": list(preprocess.IMAGENET_MEAN),
        "normalize_std": list(preprocess.IMAGENET_STD),
        "l2_normalized": True,
        "tta_flip": cfg.tta_flip,
        "n_images": int(n),
        "n_ok": int(ok.sum()),
        "n_failed": int((~ok).sum()),
        "batch_size": cfg.batch_size,
        "num_workers": cfg.workers,
        "num_workers_requested": workers_requested,
        "torch_threads": threads,
        "device": "cpu",
        "dtype": "float32",
        "elapsed_sec": round(elapsed, 2),
        "img_per_sec": round(n / max(elapsed, 1e-9), 2),
        "steady_img_per_sec": round(steady_rate, 2),
        "peak_rss_mb": round(rss_mb, 1),
        "cpu_model": _cpu_model(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": _torchvision_version(),
        "seed": 0,
        "created_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (paths.FEATURES_DIR / f"{tag}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if cfg.write_csv and cfg.limit is None:
        _write_csv(tag, items, deep, hand)
    _print_summary(meta, tag)
    return meta


def _write_csv(tag: str, items, deep: np.ndarray, hand: np.ndarray) -> None:
    df = pd.DataFrame({"image_path": [r for r, _ in items]})
    df = pd.concat(
        [
            df,
            pd.DataFrame(deep, columns=[f"deep_{i}" for i in range(deep.shape[1])]),
            pd.DataFrame(hand, columns=[f"hand_{c}" for c in handcrafted.HAND_NAMES]),
        ],
        axis=1,
    )
    df.to_csv(paths.FEATURES_DIR / f"{tag}.csv", index=False, float_format="%.6g")


def _print_summary(meta: dict, tag: str) -> None:
    print(f"\n== {tag} ==")
    print(f"  images      : {meta['n_images']} (ok {meta['n_ok']}, failed {meta['n_failed']})")
    print(f"  embedding   : deep {meta['feature_dim_deep']}-d + hand {meta['feature_dim_hand']}-d")
    print(f"  device      : cpu, {meta['torch_threads']} threads, batch {meta['batch_size']}")
    print(f"  wall time   : {meta['elapsed_sec']}s  ({meta['img_per_sec']} img/s)")
    print(f"  steady rate : {meta['steady_img_per_sec']} img/s")
    print(f"  peak RSS    : {meta['peak_rss_mb']} MB")
    print(f"  written     : {paths.FEATURES_DIR / (tag + '.npz')}")


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _torchvision_version() -> str:
    try:
        import torchvision

        return torchvision.__version__
    except Exception:  # noqa: BLE001
        return "unknown"
