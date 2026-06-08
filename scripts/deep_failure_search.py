from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.filters import threshold_niblack, threshold_sauvola
import zxingcpp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from pylibdmtx.pylibdmtx import decode as dmtx_decode
    from pylibdmtx.wrapper import DmtxSymbolSize
except Exception:  # pragma: no cover
    dmtx_decode = None
    DmtxSymbolSize = None


DEFAULT_FILES = [
    "{0783A71B-ACC9-4179-A3AF-25FC9042B4FC}.png",
    "{B3CB698A-59D5-41A0-8DC4-01E686BF9AB9}.png",
    "{E7CA298A-DC03-45A1-BABC-BEF2C7D89DFB}.png",
]


@dataclass(frozen=True)
class Candidate:
    name: str
    arr: np.ndarray


@dataclass(frozen=True)
class Hit:
    text: str
    engine: str
    variant: str
    attempts: int
    elapsed: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Exhaustive local preprocessing search for the remaining hard cases.")
    parser.add_argument("--cases", default="test_cases")
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    parser.add_argument("--seconds", type=float, default=420.0)
    parser.add_argument("--output", default="reports/deep_failure_search.csv")
    parser.add_argument("--json", default="reports/deep_failure_search.json")
    parser.add_argument("--save-candidates", action="store_true")
    args = parser.parse_args()

    case_dir = ROOT / args.cases
    rows = []
    for file_name in args.files:
        path = case_dir / file_name
        hit = search_one(path, args.seconds, args.save_candidates)
        rows.append(
            {
                "file": file_name,
                "ok": hit is not None,
                "decoded_text": hit.text if hit else "",
                "engine": hit.engine if hit else "",
                "variant": hit.variant if hit else "",
                "attempts": hit.attempts if hit else "",
                "elapsed_seconds": round(hit.elapsed, 3) if hit else args.seconds,
            }
        )
        if hit:
            print(f"OK\t{file_name}\t{hit.text}\t{hit.engine}\t{hit.variant}\t{hit.elapsed:.2f}s", flush=True)
        else:
            print(f"FAIL\t{file_name}\t{args.seconds:.0f}s", flush=True)

    output = ROOT / args.output
    json_output = ROOT / args.json
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {json_output.relative_to(ROOT)}")
    return 0 if all(row["ok"] for row in rows) else 2


def search_one(path: Path, seconds: float, save_candidates: bool) -> Hit | None:
    started = time.monotonic()
    attempts = 0
    zxing_java_every = 160
    pyzxing_dir = ROOT / "reports" / "deep_candidates" / path.stem.strip("{}")
    if save_candidates:
        pyzxing_dir.mkdir(parents=True, exist_ok=True)

    for candidate in candidates(path):
        if time.monotonic() - started > seconds:
            break
        if too_large(candidate.arr):
            continue
        attempts += 1

        hit = try_zxing_cpp(candidate, attempts, started)
        if hit:
            maybe_save(candidate, pyzxing_dir, save_candidates, "hit")
            return hit

        hit = try_dmtx(candidate, attempts, started)
        if hit:
            maybe_save(candidate, pyzxing_dir, save_candidates, "hit")
            return hit

        if attempts % zxing_java_every == 0:
            hit = try_java_zxing(candidate, attempts, started)
            if hit:
                maybe_save(candidate, pyzxing_dir, save_candidates, "hit")
                return hit

        if save_candidates and attempts <= 80:
            maybe_save(candidate, pyzxing_dir, True, f"{attempts:04d}")

    return None


def candidates(path: Path) -> Iterable[Candidate]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]

    base_arrays: list[tuple[str, np.ndarray]] = [
        ("rgb", rgb),
        ("gray", gray),
        ("red", r),
        ("green", g),
        ("blue", b),
        ("value", hsv[:, :, 2]),
        ("saturation", hsv[:, :, 1]),
        ("lab-l", lab[:, :, 0]),
        ("lab-a", lab[:, :, 1]),
        ("lab-b", lab[:, :, 2]),
        ("cr", ycrcb[:, :, 1]),
        ("cb", ycrcb[:, :, 2]),
        ("green-red", normalize(g.astype("int16") - r.astype("int16"))),
        ("blue-red", normalize(b.astype("int16") - r.astype("int16"))),
        ("maxgb-red", normalize(np.maximum(g, b).astype("int16") - r.astype("int16"))),
        ("red-maxgb", normalize(r.astype("int16") - np.maximum(g, b).astype("int16"))),
        ("green-blue", normalize(g.astype("int16") - b.astype("int16"))),
    ]

    for base_name, base in base_arrays:
        for proc_name, proc in process_variants(base_name, base):
            for geom_name, geom in geometric_variants(proc_name, proc):
                yield Candidate(geom_name, geom)

        for warped_name, warped in contour_warps(base_name, base):
            for proc_name, proc in process_variants(warped_name, warped):
                for geom_name, geom in geometric_variants(proc_name, proc, compact=True):
                    yield Candidate(geom_name, geom)


def process_variants(name: str, arr: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
    yield name, arr
    if arr.ndim == 3:
        return

    arr = ensure_u8(arr)
    clahe4 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(arr)
    clahe8 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(arr)
    yield f"{name}-eq", cv2.equalizeHist(arr)
    yield f"{name}-clahe4", clahe4
    yield f"{name}-clahe8", clahe8
    yield f"{name}-med3", cv2.medianBlur(arr, 3)
    yield f"{name}-med5", cv2.medianBlur(arr, 5)
    yield f"{name}-bilateral", cv2.bilateralFilter(arr, 5, 35, 35)
    yield f"{name}-nlm", cv2.fastNlMeansDenoising(arr, None, 14, 7, 21)
    yield f"{name}-sharp", cv2.addWeighted(arr, 2.4, cv2.GaussianBlur(arr, (0, 0), 1.2), -1.4, 0)

    for sigma in (3, 6, 10, 16):
        bg = cv2.GaussianBlur(arr, (0, 0), sigma)
        yield f"{name}-bgsub-g{sigma}", normalize(arr.astype("int16") - bg.astype("int16"))
        yield f"{name}-bgdiv-g{sigma}", normalize((arr.astype("float32") / (bg.astype("float32") + 1.0)) * 128.0)

    for gamma in (0.35, 0.5, 0.7, 1.35, 1.8, 2.4):
        yield f"{name}-gamma{gamma}", gamma_correct(arr, gamma)

    for src_name, src in [
        (name, arr),
        (f"{name}-clahe4", clahe4),
        (f"{name}-clahe8", clahe8),
        (f"{name}-med3", cv2.medianBlur(arr, 3)),
    ]:
        yield from threshold_variants(src_name, src)


def threshold_variants(name: str, src: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
    _, otsu = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield f"{name}-otsu", otsu
    yield f"{name}-otsu-inv", 255 - otsu

    for block in (15, 21, 31, 41, 61):
        if min(src.shape[:2]) <= block:
            continue
        for constant in (-11, -7, -3, 0, 3, 7, 11):
            adaptive = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, constant)
            yield f"{name}-ad{block}c{constant}", adaptive
            yield f"{name}-ad{block}c{constant}-inv", 255 - adaptive

    for window in (15, 25, 35, 51):
        if min(src.shape[:2]) <= window:
            continue
        try:
            sauvola = (src > threshold_sauvola(src, window_size=window, k=0.2)).astype("uint8") * 255
            yield f"{name}-sauvola{window}", sauvola
            yield f"{name}-sauvola{window}-inv", 255 - sauvola
        except Exception:
            pass
        try:
            niblack = (src > threshold_niblack(src, window_size=window, k=0.2)).astype("uint8") * 255
            yield f"{name}-niblack{window}", niblack
            yield f"{name}-niblack{window}-inv", 255 - niblack
        except Exception:
            pass

    for kernel_size in (2, 3):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        yield f"{name}-otsu-open{kernel_size}", opened
        yield f"{name}-otsu-close{kernel_size}", closed
        yield f"{name}-otsu-open{kernel_size}-inv", 255 - opened
        yield f"{name}-otsu-close{kernel_size}-inv", 255 - closed


def geometric_variants(name: str, arr: np.ndarray, compact: bool = False) -> Iterable[tuple[str, np.ndarray]]:
    scales = (1, 2, 3, 4, 6, 8) if not compact else (1, 2, 4)
    angles = (0, -1, 1, -3, 3, -5, 5, -7, 7, -9, 9, -12, 12, -15, 15) if not compact else (0, -5, 5, -9, 9)
    pad_values = (255, 0, 128, None)

    for scale in scales:
        scaled = resize_arr(arr, scale)
        for pad_value in pad_values:
            if pad_value is None:
                padded = scaled
                padded_name = f"{name}/x{scale}"
            else:
                padded = pad_arr(scaled, pad_value, 0.12)
                padded_name = f"{name}/x{scale}/pad{pad_value}"
            yield padded_name, padded
            for angle in angles:
                if angle == 0:
                    continue
                yield f"{padded_name}/rot{angle}", rotate_arr(padded, angle, pad_value if pad_value is not None else 255)


def contour_warps(name: str, arr: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = ensure_u8(arr)
    sources = [gray, cv2.equalizeHist(gray), cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(gray)]
    for source_index, src in enumerate(sources):
        for invert in (False, True):
            work = 255 - src if invert else src
            _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
            for contour_index, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < 0.08 * gray.shape[0] * gray.shape[1]:
                    continue
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if min(width, height) < 25:
                    continue
                ratio = max(width, height) / max(1.0, min(width, height))
                if ratio > 2.0:
                    continue
                box = cv2.boxPoints(rect).astype("float32")
                size = int(max(width, height))
                dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")
                ordered = order_points(box)
                matrix = cv2.getPerspectiveTransform(ordered, dst)
                warped = cv2.warpPerspective(gray, matrix, (size, size), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                yield f"{name}-warp{sourcen(source_index)}-{int(invert)}-{contour_index}", warped


def try_zxing_cpp(candidate: Candidate, attempts: int, started: float) -> Hit | None:
    arr = ensure_u8(candidate.arr)
    for binarizer in (zxingcpp.Binarizer.LocalAverage, zxingcpp.Binarizer.GlobalHistogram, zxingcpp.Binarizer.FixedThreshold, zxingcpp.Binarizer.BoolCast):
        for pure in (False, True):
            for downscale in (False, True):
                try:
                    decoded = zxingcpp.read_barcodes(
                        arr,
                        formats=zxingcpp.BarcodeFormat.DataMatrix,
                        try_rotate=True,
                        try_downscale=downscale,
                        try_invert=True,
                        binarizer=binarizer,
                        is_pure=pure,
                    )
                except Exception:
                    decoded = []
                for item in decoded:
                    text = getattr(item, "text", "")
                    if not getattr(item, "error", None) and plausible(text):
                        return Hit(text, f"zxing-cpp/{binarizer.name}/pure={pure}/down={downscale}", candidate.name, attempts, time.monotonic() - started)
    return None


def try_dmtx(candidate: Candidate, attempts: int, started: float) -> Hit | None:
    if dmtx_decode is None:
        return None
    image = arr_to_pil(ensure_u8(candidate.arr))
    for params in dmtx_params():
        try:
            decoded = dmtx_decode(image, timeout=85, max_count=1, **params)
        except Exception:
            decoded = []
        for item in decoded:
            text = item.data.decode("utf-8", errors="replace")
            if plausible(text):
                return Hit(text, f"pylibdmtx/{format_params(params)}", candidate.name, attempts, time.monotonic() - started)
    return None


def try_java_zxing(candidate: Candidate, attempts: int, started: float) -> Hit | None:
    java = shutil.which("java")
    if java is None:
        return None
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "candidate.png"
        arr_to_pil(ensure_u8(candidate.arr)).save(path)
        try:
            from pyzxing import BarCodeReader

            decoded = BarCodeReader().decode(str(path))
        except Exception:
            return None
    for item in decoded or []:
        parsed = item.get("parsed") if isinstance(item, dict) else None
        if isinstance(parsed, bytes):
            text = parsed.decode("utf-8", errors="replace")
        else:
            text = str(parsed or "")
        if plausible(text):
            return Hit(text, "java-zxing", candidate.name, attempts, time.monotonic() - started)
    return None


def dmtx_params() -> list[dict[str, object]]:
    square = DmtxSymbolSize.DmtxSymbolSquareAuto if DmtxSymbolSize is not None else None
    params: list[dict[str, object]] = []
    for shrink in (1, 2, 3, 4, 5, 6, 8, 10, 12):
        params.append({"shrink": shrink})
        if square is not None:
            params.append({"shrink": shrink, "shape": square})
    if square is not None:
        params.append({"shape": square})
        for threshold in (8, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 220):
            params.append({"threshold": threshold, "shape": square})
        for deviation in (5, 10, 20, 40, 60, 80, 100):
            params.append({"deviation": deviation, "shape": square})
        for edge in (5, 10, 20, 30, 50):
            params.append({"min_edge": edge, "shape": square})
    params.append({})
    return [param for param in params if not any(value is None for value in param.values())]


def format_params(params: dict[str, object]) -> str:
    if not params:
        return "default"
    parts = []
    for key, value in params.items():
        if key == "shape":
            value = getattr(value, "name", value)
        parts.append(f"{key}={value}")
    return ",".join(parts)


def plausible(text: str) -> bool:
    if not text or len(text.strip()) < 6:
        return False
    if not any(ch.isalnum() for ch in text):
        return False
    controls = sum(ord(ch) < 32 and ch not in "\r\n\t" for ch in text)
    return controls <= max(1, len(text) // 5)


def too_large(arr: np.ndarray) -> bool:
    height, width = arr.shape[:2]
    return height * width > 2_800_000


def ensure_u8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    return normalize(arr)


def normalize(arr: np.ndarray) -> np.ndarray:
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")


def gamma_correct(arr: np.ndarray, gamma: float) -> np.ndarray:
    table = (np.power(np.arange(256) / 255.0, gamma) * 255).astype("uint8")
    return cv2.LUT(arr, table)


def resize_arr(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return arr
    height, width = arr.shape[:2]
    unique_count = len(np.unique(arr)) if arr.ndim == 2 else 999
    interpolation = cv2.INTER_NEAREST if unique_count <= 8 else cv2.INTER_CUBIC
    return cv2.resize(arr, (width * scale, height * scale), interpolation=interpolation)


def pad_arr(arr: np.ndarray, value: int, fraction: float) -> np.ndarray:
    height, width = arr.shape[:2]
    pad = max(8, int(max(height, width) * fraction))
    border = (value, value, value) if arr.ndim == 3 else value
    return cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=border)


def rotate_arr(arr: np.ndarray, degrees: float, value: int) -> np.ndarray:
    height, width = arr.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2 - width / 2
    matrix[1, 2] += new_height / 2 - height / 2
    border = (value, value, value) if arr.ndim == 3 else value
    return cv2.warpAffine(arr, matrix, (new_width, new_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def arr_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")


def order_points(points: np.ndarray) -> np.ndarray:
    result = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)
    result[0] = points[np.argmin(sums)]
    result[2] = points[np.argmax(sums)]
    result[1] = points[np.argmin(diffs)]
    result[3] = points[np.argmax(diffs)]
    return result


def sourcen(index: int) -> str:
    return ["raw", "eq", "clahe"][index] if index < 3 else str(index)


def maybe_save(candidate: Candidate, directory: Path, enabled: bool, prefix: str) -> None:
    if not enabled:
        return
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in candidate.name)[:180]
    arr_to_pil(ensure_u8(candidate.arr)).save(directory / f"{prefix}_{safe_name}.png")


if __name__ == "__main__":
    raise SystemExit(main())
