"""Bounded, pixel-derived recovery for small, blurred Data Matrix symbols.

Fit the *observed* finder border before asking ZXing to decode a sampled grid.
No finder bits or data bits are repaired here; only the decoder's ECC may repair
errors. This module deliberately has no access to filenames or reference data.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import time

import cv2
import numpy as np
from PIL import Image
import zxingcpp


@dataclass(frozen=True)
class GridRecovery:
    text: str
    variant: str
    rendered: np.ndarray
    attempts: int
    raw_bytes: bytes
    finder_score: float


@lru_cache(maxsize=16)
def _geometry(width: int, height: int, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coarse module-center coordinates from relative image margins."""
    margins = np.arange(0.06, 0.281, 0.02, dtype=np.float32)
    boxes = np.asarray(list(product(margins, repeat=4)), dtype=np.float32)
    left, top, right, bottom = boxes.T
    region_width = width * (1 - left - right)
    region_height = height * (1 - top - bottom)
    keep = (region_width / region_height > 0.65) & (region_width / region_height < 1.4)
    params = np.column_stack((left * width, top * height, region_width / size, region_height / size))[keep]
    params[:, :2] += params[:, 2:] / 2
    indices = np.asarray([(r, c) for r in range(size) for c in range(size)
                          if r in (0, size - 1) or c in (0, size - 1)], dtype=np.int32)
    finder = np.zeros((size, size), dtype=bool)
    finder[:, 0] = True
    finder[-1, :] = True
    finder[0, :] = np.arange(size) % 2 == 0
    finder[:, -1] = np.arange(size) % 2 == 1
    patterns = np.asarray([np.rot90(finder, k)[indices[:, 0], indices[:, 1]] for k in range(4)])
    return params, indices, patterns


def _rank_geometry(source: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = source.shape
    params, indices, patterns = _geometry(width, height, size)
    xs = params[:, 0, None] + params[:, 2, None] * indices[:, 1]
    ys = params[:, 1, None] + params[:, 3, None] * indices[:, 0]
    samples = cv2.remap(source, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR)
    # Rank both polarities and all four orientations by finder evidence only.
    best = np.zeros(len(params), dtype=np.float32)
    deviation = float(source.std())
    for threshold in (0.0, deviation * 0.15, deviation * 0.30):
        matches = ((samples > threshold)[:, None, :] == patterns[None, :, :]).mean(axis=2)
        score = np.maximum(matches, 1 - matches).max(axis=1)
        best = np.maximum(best, score)
    candidates = np.flatnonzero(best >= 0.88)
    weights = patterns.astype(np.float32)
    weights -= weights.mean(axis=1, keepdims=True)
    separation = np.abs(samples @ weights.T).max(axis=1) / max(deviation, 0.001)
    ranking = best + separation * 0.0001
    candidates = candidates[np.argsort(ranking[candidates], kind="stable")[::-1]][:80]
    return params[candidates], indices, patterns


def _sources(rgb: np.ndarray):
    mean = rgb.mean(axis=2)
    candidates = [("mean", mean), ("yellow", (rgb[:, :, 0] + rgb[:, :, 1]) / 2 - rgb[:, :, 2]),
                  ("green-red", rgb[:, :, 1] - rgb[:, :, 0]), ("blue", rgb[:, :, 2])]
    for name, source in candidates:
        if float(source.std()) < 0.5:
            continue
        # The background varies more slowly than the module spacing.
        local = source - cv2.GaussianBlur(source, (0, 0), max(2.0, min(source.shape) / 11))
        if float(local.std()) >= 0.5:
            yield f"{name}-local", local.astype(np.float32)


def _render(matrix: np.ndarray) -> np.ndarray:
    pixels = np.pad(255 - matrix.astype(np.uint8) * 255, 3, constant_values=255)
    return cv2.resize(pixels, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)


def recover_grid(
    image: Image.Image,
    max_seconds: float = 3.0,
    grid_sizes: tuple[int, ...] = (14, 16, 18, 20),
) -> GridRecovery | None:
    """Recover an ECC-valid symbol with an observed finder agreement >= 90%.

    The bounded search targets axis-aligned, approximately square symbols with
    some image margin. It supplements ordinary detection, and returns ``None``
    when evidence or the time budget is insufficient.
    """
    started = time.monotonic()
    deadline = started + max(0.0, max_seconds)
    if max_seconds <= 0:
        return None
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    if min(rgb.shape[:2]) < 28 or float(rgb.std()) < 0.5:
        return None
    if max(rgb.shape[:2]) > 320:
        scale = 320 / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sources = list(_sources(rgb))
    attempts = 0
    seen: set[bytes] = set()
    valid_sizes = tuple(size for size in grid_sizes if size in (10, 12, 14, 16, 18, 20, 22, 24, 26))
    for size in valid_sizes:
        for source_name, source in sources:
            if time.monotonic() >= deadline:
                return None
            candidates, indices, patterns = _rank_geometry(source, size)
            # Allow another channel / size to run when one source is unhelpful.
            pass_deadline = min(deadline, time.monotonic() + max(0.12, max_seconds / 5))
            axis = np.arange(size, dtype=np.float32)
            offsets = [(0.0, 0.0, 0.0, 0.0)]
            offsets.extend(offset for offset in product((0.0, -0.5, 0.5), repeat=4) if any(offset))
            for parameter in candidates:
                for offset in offsets:
                    if time.monotonic() >= pass_deadline:
                        break
                    dx, dy, dp, dq = offset
                    x0, y0, px, py = parameter + np.array((dx, dy, dp * 0.16, dq * 0.16), dtype=np.float32)
                    xs, ys = np.meshgrid(x0 + axis * px, y0 + axis * py)
                    sampled = cv2.remap(source, xs, ys, cv2.INTER_LINEAR)
                    thresholds = np.percentile(sampled, (45, 40, 50, 55, 60))
                    for percentile, threshold in zip((45, 40, 50, 55, 60), thresholds):
                        base = sampled > threshold
                        border = base[indices[:, 0], indices[:, 1]]
                        matches = (patterns == border).mean(axis=1)
                        for inverted, scores in ((False, matches), (True, 1 - matches)):
                            rotation = int(np.argmax(scores))
                            finder_score = float(scores[rotation])
                            if finder_score < 0.90:
                                continue
                            matrix = np.rot90(~base if inverted else base, -rotation)
                            fingerprint = bytes((size,)) + np.packbits(matrix).tobytes()
                            if fingerprint in seen:
                                continue
                            seen.add(fingerprint)
                            rendered = _render(matrix)
                            attempts += 1
                            result = zxingcpp.read_barcode(
                                rendered, formats=zxingcpp.BarcodeFormat.DataMatrix,
                                is_pure=True, binarizer=zxingcpp.Binarizer.BoolCast,
                                try_rotate=True, try_invert=False, try_downscale=False,
                            )
                            if result is None or not result.valid or result.error or not result.bytes:
                                continue
                            variant = (
                                f"center-grid/{source_name}/size-{size}/"
                                f"origin-{x0:.2f}-{y0:.2f}/pitch-{px:.3f}-{py:.3f}/"
                                f"p{percentile}/invert-{inverted}/rotation-{rotation}/finder-{finder_score:.3f}"
                            )
                            return GridRecovery(result.text, variant, rendered, attempts, bytes(result.bytes), finder_score)
                if time.monotonic() >= pass_deadline:
                    break
    return None
