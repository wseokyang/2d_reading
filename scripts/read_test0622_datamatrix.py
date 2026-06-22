from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
import zxingcpp


@dataclass(frozen=True)
class DataMatrixHit:
    text: str
    engine: str
    variant: str
    attempts: int
    elapsed_seconds: float
    rendered: np.ndarray | None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read the low-contrast test0622-style Data Matrix by sampling a likely "
            "14x14 module grid, re-rendering it cleanly, then ECC-decoding it."
        )
    )
    parser.add_argument("image", nargs="?", default="test0622.png")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--save-rendered", help="Optional path for the reconstructed clean symbol.")
    args = parser.parse_args()

    path = Path(args.image)
    image = Image.open(path).convert("RGB")
    hit = read_low_contrast_datamatrix(image, max_seconds=args.seconds)

    if not hit:
        print(f"FAIL\t{path}\t{args.seconds:.1f}s")
        return 2

    print(f"OK\t{path}\t{hit.text}")
    print(f"engine\t{hit.engine}")
    print(f"variant\t{hit.variant}")
    print(f"attempts\t{hit.attempts}")
    print(f"elapsed_seconds\t{hit.elapsed_seconds:.3f}")

    if args.save_rendered and hit.rendered is not None:
        Image.fromarray(hit.rendered, mode="L").save(args.save_rendered)
        print(f"rendered\t{args.save_rendered}")

    return 0


def read_low_contrast_datamatrix(image: Image.Image, max_seconds: float = 12.0) -> DataMatrixHit | None:
    started = time.monotonic()
    rgb = np.array(image.convert("RGB"))

    direct = decode_with_zxing(rgb, "original-rgb", attempts=1, started=started)
    if direct:
        return direct

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    attempts = 1

    for source_name, source in gray_sources(gray):
        height, width = source.shape[:2]
        for box in grid_ratio_boxes(width, height):
            if time.monotonic() - started > max_seconds:
                return None

            left, top, right, bottom = box
            roi = source[top : height - bottom, left : width - right]
            if min(roi.shape[:2]) < 42:
                continue

            sampled = grid_cell_means(roi, size=14)
            for threshold_name in ("otsu", "mean", "p42", "p45", "p50", "p55", "p56", "p58", "p59", "p62"):
                threshold = grid_threshold_value(sampled, threshold_name)
                for invert in (False, True):
                    matrix = sampled < threshold if not invert else sampled >= threshold
                    for rotation in range(4):
                        attempts += 1
                        oriented = np.rot90(matrix, rotation)
                        finder_score = grid_finder_score(oriented)
                        if finder_score < 0.89:
                            continue

                        for corrected in (False, True):
                            clean = force_grid_finder(oriented) if corrected else oriented
                            rendered = render_grid_matrix(clean)
                            variant = (
                                f"{source_name}/box-{left}-{top}-{right}-{bottom}/"
                                f"{threshold_name}/invert-{invert}/rot-{rotation}/"
                                f"finder-{finder_score:.3f}/corrected-{corrected}"
                            )
                            decoded = decode_rendered_grid(rendered)
                            if decoded:
                                return DataMatrixHit(
                                    text=decoded[0],
                                    engine=decoded[1],
                                    variant=variant,
                                    attempts=attempts,
                                    elapsed_seconds=time.monotonic() - started,
                                    rendered=rendered,
                                )

    return None


def gray_sources(gray: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
    yield "gray", gray
    yield "gray-normalized", normalize(gray)
    yield "gray-clahe", cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
    yield "gray-median3", cv2.medianBlur(gray, 3)


def grid_ratio_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    min_side = min(width, height)
    unit = max(1, round(min_side * 0.01))
    spread = max(5, round(min_side * 0.025))
    offsets = sorted({-spread, -2 * unit, -unit, 0, unit, 2 * unit, spread})
    fractions = (0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28, 0.32, 0.35)
    priority_patterns = (
        (0, 0, 0, 0),
        (0, 1, 0, -1),
        (-1, 1, 1, -1),
        (-2, 2, 2, -2),
        (-3, 2, 3, -2),
        (-3, 3, 3, -3),
        (-4, 4, 4, -4),
        (2, 2, 2, 0),
        (3, 3, 3, 1),
        (-2, 0, 2, 0),
        (0, 2, 0, -2),
    )

    result: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()

    def add_box(box: tuple[int, int, int, int]) -> None:
        left, top, right, bottom = box
        if width - left - right < 30 or height - top - bottom < 30:
            return
        if box in seen:
            return
        seen.add(box)
        result.append(box)

    for fraction in fractions:
        x_margin = round(width * fraction)
        y_margin = round(height * fraction)

        for left_pattern, top_pattern, right_pattern, bottom_pattern in priority_patterns:
            add_box(
                (
                    max(0, x_margin + left_pattern * unit),
                    max(0, y_margin + top_pattern * unit),
                    max(0, x_margin + right_pattern * unit),
                    max(0, y_margin + bottom_pattern * unit),
                )
            )

        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        for left_offset in offsets:
            for top_offset in offsets:
                for right_offset in offsets:
                    for bottom_offset in offsets:
                        left = max(0, x_margin + left_offset)
                        top = max(0, y_margin + top_offset)
                        right = max(0, x_margin + right_offset)
                        bottom = max(0, y_margin + bottom_offset)
                        roi_width = width - left - right
                        roi_height = height - top - bottom
                        if roi_width < 30 or roi_height < 30:
                            continue
                        shape_cost = abs((roi_width / max(1, roi_height)) - 1.0)
                        balance_cost = (abs(left - right) + abs(top - bottom)) * 0.01
                        candidates.append((shape_cost + balance_cost, (left, top, right, bottom)))

        added = 0
        for _, box in sorted(candidates):
            before = len(result)
            add_box(box)
            if len(result) == before:
                continue
            added += 1
            if added >= 35:
                break

    return result


def grid_cell_means(roi: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(roi, (size * 9, size * 9), interpolation=cv2.INTER_CUBIC)
    return resized.reshape(size, 9, size, 9).mean(axis=(1, 3))


def grid_threshold_value(sampled: np.ndarray, name: str) -> float:
    if name == "otsu":
        normalized = normalize(sampled)
        threshold, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        span = float(sampled.max() - sampled.min())
        return float(threshold) / 255.0 * span + float(sampled.min())
    if name == "mean":
        return float(sampled.mean())
    return float(np.percentile(sampled, int(name[1:])))


def grid_finder_score(matrix: np.ndarray) -> float:
    size = matrix.shape[0]
    top = np.array([index % 2 == 0 for index in range(size)], dtype=bool)
    right = np.array([index % 2 == 1 for index in range(size)], dtype=bool)
    right[-1] = True
    checks = [
        matrix[:, 0] == True,
        matrix[-1, :] == True,
        matrix[0, :] == top,
        matrix[:, -1] == right,
    ]
    return float(np.concatenate(checks).mean())


def force_grid_finder(matrix: np.ndarray) -> np.ndarray:
    result = matrix.copy()
    size = result.shape[0]
    result[:, 0] = True
    result[-1, :] = True
    result[0, :] = np.array([index % 2 == 0 for index in range(size)], dtype=bool)
    right = np.array([index % 2 == 1 for index in range(size)], dtype=bool)
    right[-1] = True
    result[:, -1] = right
    result[-1, -1] = True
    return result


def render_grid_matrix(matrix: np.ndarray, module_size: int = 18, quiet_modules: int = 4) -> np.ndarray:
    size = matrix.shape[0]
    rendered_size = (size + quiet_modules * 2) * module_size
    rendered = np.full((rendered_size, rendered_size), 255, dtype="uint8")
    for y in range(size):
        for x in range(size):
            if matrix[y, x]:
                y0 = (y + quiet_modules) * module_size
                x0 = (x + quiet_modules) * module_size
                rendered[y0 : y0 + module_size, x0 : x0 + module_size] = 0
    return rendered


def decode_rendered_grid(arr: np.ndarray) -> tuple[str, str] | None:
    for binarizer in (
        zxingcpp.Binarizer.BoolCast,
        zxingcpp.Binarizer.FixedThreshold,
        zxingcpp.Binarizer.GlobalHistogram,
        zxingcpp.Binarizer.LocalAverage,
    ):
        for pure in (True, False):
            try:
                decoded = zxingcpp.read_barcodes(
                    arr,
                    formats=zxingcpp.BarcodeFormat.DataMatrix,
                    try_rotate=True,
                    try_downscale=False,
                    try_invert=True,
                    binarizer=binarizer,
                    is_pure=pure,
                )
            except Exception:
                decoded = []
            for item in decoded:
                text = getattr(item, "text", "")
                if is_plausible_text(text) and not getattr(item, "error", None):
                    return text, f"zxing-cpp/{binarizer.name}/grid/pure={pure}"
    return None


def decode_with_zxing(arr: np.ndarray, variant: str, attempts: int, started: float) -> DataMatrixHit | None:
    for binarizer in (
        zxingcpp.Binarizer.LocalAverage,
        zxingcpp.Binarizer.GlobalHistogram,
        zxingcpp.Binarizer.FixedThreshold,
    ):
        try:
            decoded = zxingcpp.read_barcodes(
                arr,
                formats=zxingcpp.BarcodeFormat.DataMatrix,
                try_rotate=True,
                try_downscale=True,
                try_invert=True,
                binarizer=binarizer,
                is_pure=False,
            )
        except Exception:
            decoded = []
        for item in decoded:
            text = getattr(item, "text", "")
            if is_plausible_text(text) and not getattr(item, "error", None):
                return DataMatrixHit(
                    text=text,
                    engine=f"zxing-cpp/{binarizer.name}",
                    variant=variant,
                    attempts=attempts,
                    elapsed_seconds=time.monotonic() - started,
                    rendered=None,
                )
    return None


def normalize(arr: np.ndarray) -> np.ndarray:
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")


def is_plausible_text(text: str) -> bool:
    return bool(text and len(text.strip()) >= 6 and any(ch.isalnum() for ch in text))


if __name__ == "__main__":
    raise SystemExit(main())
