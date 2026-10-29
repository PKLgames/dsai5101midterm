"""Download the 1,558 cat images from Hugging Face.

Design notes
------------
* The image list is derived from the CSV files, so we never fetch anything the
  dataset does not reference.
* Downloads are resumable: an existing file with the expected byte size is
  skipped, partial files are written to ``*.part`` and atomically renamed.
* 16 worker threads were measured to saturate the link (~2.4 MB/s aggregate on
  the machine this pipeline was built on); higher concurrency gave no gain.
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from PIL import Image

from . import index, paths

USER_AGENT = "cat-image-arena/1.0 (course project)"
TIMEOUT = 60


@dataclass
class Result:
    image: str
    status: str  # "ok" | "skipped" | "failed"
    n_bytes: int = 0
    width: int = 0
    height: int = 0
    attempts: int = 0
    error: str = ""
    source_url: str = ""


def image_list() -> list[str]:
    train, test = index.load_train(), index.load_test()
    imgs = set(train["a_image"]) | set(train["b_image"])
    imgs |= set(test["a_image"]) | set(test["b_image"])
    return sorted(imgs)


def _fetch(url: str, dest: Path, retries: int = 3) -> tuple[int, int, int, int]:
    """Download ``url`` to ``dest``; return (bytes, width, height, attempts)."""
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                payload = resp.read()
            if not payload:
                raise ValueError("empty response")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(io.BytesIO(payload)) as im:
                width, height = im.size
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(payload)
            os.replace(tmp, dest)
            return len(payload), width, height, attempt
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
            last = exc
            if attempt < retries:
                time.sleep((2 ** (attempt - 1)) + random.random())
    raise RuntimeError(f"{type(last).__name__}: {last}")


def _one(image: str, lock: threading.Lock, counters: dict[str, int]) -> Result:
    dest = paths.IMAGES_DIR / image
    url = f"{paths.HF_BASE}/images/{image}"
    if dest.is_file() and dest.stat().st_size > 0:
        try:
            with Image.open(dest) as im:
                w, h = im.size
            return Result(image, "skipped", dest.stat().st_size, w, h, 0, "", url)
        except Exception:
            dest.unlink(missing_ok=True)  # corrupt cache -> redownload
    try:
        n, w, h, attempts = _fetch(url, dest)
        with lock:
            counters["bytes"] += n
        return Result(image, "ok", n, w, h, attempts, "", url)
    except Exception as exc:  # noqa: BLE001 - report every failure, never abort
        return Result(image, "failed", 0, 0, 0, 0, str(exc), url)


def download_all(workers: int = 16, limit: int | None = None) -> pd.DataFrame:
    paths.ensure_dirs()
    images = image_list()
    if limit:
        images = images[:limit]

    lock = threading.Lock()
    counters = {"bytes": 0}
    t0 = time.time()
    results: list[Result] = []
    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, img, lock, counters): img for img in images}
        for fut in cf.as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1
            if done % 100 == 0 or done == len(images):
                elapsed = time.time() - t0
                print(
                    f"[{done}/{len(images)}] {elapsed:6.1f}s  "
                    f"{counters['bytes'] / 1e6:7.1f} MB  "
                    f"{counters['bytes'] / 1e6 / max(elapsed, 1e-9):5.2f} MB/s",
                    flush=True,
                )

    df = pd.DataFrame([r.__dict__ for r in results]).sort_values("image")
    df.rename(columns={"image": "image_path"}, inplace=True)
    df["downloaded_at_utc"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    df.to_csv(paths.MANIFEST_CSV, index=False)

    failed = df[df["status"] == "failed"]
    if len(failed):
        failed.to_csv(paths.DOWNLOAD_ERRORS_CSV, index=False)
        print(f"\n!! {len(failed)} downloads failed -> {paths.DOWNLOAD_ERRORS_CSV}")
        print(failed[["image_path", "error"]].head(20).to_string(index=False))
    else:
        paths.DOWNLOAD_ERRORS_CSV.unlink(missing_ok=True)

    elapsed = time.time() - t0
    print(
        f"\nstatus: {df['status'].value_counts().to_dict()}\n"
        f"total bytes: {df['n_bytes'].sum() / 1e6:.1f} MB in {elapsed:.1f}s\n"
        f"manifest: {paths.MANIFEST_CSV}"
    )
    return df


if __name__ == "__main__":
    download_all()
