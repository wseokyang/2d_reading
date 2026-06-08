from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
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


FAILURES = [
    "{0783A71B-ACC9-4179-A3AF-25FC9042B4FC}.png",
    "{B3CB698A-59D5-41A0-8DC4-01E686BF9AB9}.png",
    "{E7CA298A-DC03-45A1-BABC-BEF2C7D89DFB}.png",
]


@dataclass(frozen=True)
class Candidate:
    name: str
    arr: np.ndarray


@dataclass(frozen=True)
class Result:
    ok: bool
    text: str
    engine: str
    variant: str
    attempts: int
    elapsed_seconds: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Targeted search for the remaining hard Data Matrix crops.")
    parser.add_argument("--cases", default="test_cases")
    parser.add_argument("--files", nargs="*", default=FAILURES)
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--output", default="reports/targeted_failure_search.csv")
    parser.add_argument("--json", default="reports/targeted_failure_search.json")
    args = parser.parse_args()

    rows = []
    for file_name in args.files:
        print(f"START\t{file_name}", flush=True)
        result = search_file(ROOT / args.cases / file_name, args.seconds)
        rows.append(
            {
                "file": file_name,
                "ok": result.ok,
                "decoded_text": result.text,
                "engine": result.engine,
                "variant": result.variant,
                "attempts": result.attempts,
                "elapsed_seconds": round(result.elapsed_seconds, 3),
            }
        )
        status = "OK" if result.ok else "FAIL"
        print(f"{status}\t{file_name}\t{result.text or '-'}\t{result.engine}\t{result.variant}\t{result.elapsed_seconds:.2f}s", flush=True)

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


def search_file(path: Path, seconds: float) -> Result:
    started = time.monotonic()
    zxing_attempts = 0
    candidate_cache: list[Candidate] = []

    # Stage 1: zxing-cpp is fast enough to run broadly.
    for candidate in iter_candidates(path, broad=True):
        if time.monotonic() - started > seconds * 0.45:
            break
        if too_large(candidate.arr):
            continue
        candidate_cache.append(candidate)
        zxing_attempts += 1
        hit = try_zxing(candidate)
        if hit:
            return Result(True, hit[0], hit[1], candidate.name, zxing_attempts, time.monotonic() - started)
        if zxing_attempts % 1000 == 0:
            print(f"  zxing {path.name}: {zxing_attempts} candidates", flush=True)

    # Stage 2: libdmtx is slower; use higher-value candidates first.
    dmtx_attempts = 0
    ordered = sorted(candidate_cache, key=candidate_priority)
    for candidate in ordered:
        if time.monotonic() - started > seconds:
            break
        if too_large(candidate.arr):
            continue
        for params in dmtx_params(candidate.name):
            if time.monotonic() - started > seconds:
                break
            dmtx_attempts += 1
            hit = try_dmtx(candidate, params)
            if hit:
                attempts = zxing_attempts + dmtx_attempts
                return Result(True, hit, f"pylibdmtx/{format_params(params)}", candidate.name, attempts, time.monotonic() - started)
        if dmtx_attempts % 300 == 0 and dmtx_attempts:
            print(f"  dmtx {path.name}: {dmtx_attempts} attempts", flush=True)

    return Result(False, "", "", "", zxing_attempts + dmtx_attempts, time.monotonic() - started)


def iter_candidates(path: Path, broad: bool) -> Iterable[Candidate]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]

    channels = [
        ("rgb", rgb),
        ("gray", gray),
        ("green", g),
        ("blue", b),
        ("red", r),
        ("value", hsv[:, :, 2]),
        ("sat", hsv[:, :, 1]),
        ("lab-l", lab[:, :, 0]),
        ("lab-a", lab[:, :, 1]),
        ("cr", ycc[:, :, 1]),
        ("cb", ycc[:, :, 2]),
        ("g-r", normalize(g.astype("int16") - r.astype("int16"))),
        ("b-r", normalize(b.astype("int16") - r.astype("int16"))),
        ("maxgb-r", normalize(np.maximum(g, b).astype("int16") - r.astype("int16"))),
        ("r-maxgb", normalize(r.astype("int16") - np.maximum(g, b).astype("int16"))),
    ]
    if not broad:
        channels = channels[:6]

    for name, arr in channels:
        for proc_name, proc in preprocess(name, arr):
            for geom_name, geom in geometry(proc_name, proc):
                yield Candidate(geom_name, geom)


def preprocess(name: str, arr: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
    yield name, arr
    if arr.ndim == 3:
        return

    arr = normalize(arr)
    clahe4 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(arr)
    clahe8 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(arr)
    variants = [
        ("eq", cv2.equalizeHist(arr)),
        ("clahe4", clahe4),
        ("clahe8", clahe8),
        ("med3", cv2.medianBlur(arr, 3)),
        ("med5", cv2.medianBlur(arr, 5)),
        ("sharp", cv2.addWeighted(arr, 2.6, cv2.GaussianBlur(arr, (0, 0), 1.2), -1.6, 0)),
        ("nlm", cv2.fastNlMeansDenoising(arr, None, 18, 7, 21)),
    ]
    for sigma in (4, 8, 14):
        bg = cv2.GaussianBlur(arr, (0, 0), sigma)
        variants.append((f"bgsub{sigma}", normalize(arr.astype("int16") - bg.astype("int16"))))
        variants.append((f"bgdiv{sigma}", normalize((arr.astype("float32") / (bg.astype("float32") + 1.0)) * 128.0)))
    for label, variant in variants:
        yield f"{name}-{label}", variant

    threshold_sources = [(name, arr), (f"{name}-clahe4", clahe4), (f"{name}-clahe8", clahe8)]
    for src_name, src in threshold_sources:
        _, otsu = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield f"{src_name}-otsu", otsu
        yield f"{src_name}-otsu-inv", 255 - otsu
        for block in (15, 21, 31, 41, 61):
            if min(src.shape[:2]) <= block:
                continue
            for constant in (-9, -5, 0, 5, 9):
                adaptive = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, constant)
                yield f"{src_name}-ad{block}c{constant}", adaptive
                yield f"{src_name}-ad{block}c{constant}-inv", 255 - adaptive
        for window in (15, 25, 35, 51):
            if min(src.shape[:2]) <= window:
                continue
            for label, func in (("sauvola", threshold_sauvola), ("niblack", threshold_niblack)):
                try:
                    binary = (src > func(src, window_size=window)).astype("uint8") * 255
                except Exception:
                    continue
                yield f"{src_name}-{label}{window}", binary
                yield f"{src_name}-{label}{window}-inv", 255 - binary


def geometry(name: str, arr: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
    for scale in (1, 2, 3, 4, 6, 8, 10):
        scaled = resize(arr, scale)
        for pad_value in (255, 0, 128, None):
            if pad_value is None:
                padded = scaled
                pad_name = f"{name}/x{scale}"
            else:
                padded = pad(scaled, pad_value)
                pad_name = f"{name}/x{scale}/pad{pad_value}"
            yield pad_name, padded
            for angle in (-15, -12, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 12, 15):
                yield f"{pad_name}/rot{angle}", rotate(padded, angle, pad_value if pad_value is not None else 255)


def try_zxing(candidate: Candidate) -> tuple[str, str] | None:
    arr = as_u8(candidate.arr)
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
                        return text, f"zxing-cpp/{binarizer.name}/pure={pure}/down={downscale}"
    return None


def try_dmtx(candidate: Candidate, params: dict[str, object]) -> str | None:
    if dmtx_decode is None:
        return None
    image = to_pil(as_u8(candidate.arr))
    try:
        decoded = dmtx_decode(image, timeout=120, max_count=1, **params)
    except Exception:
        return None
    for item in decoded:
        text = item.data.decode("utf-8", errors="replace")
        if plausible(text):
            return text
    return None


def dmtx_params(candidate_name: str) -> list[dict[str, object]]:
    square = DmtxSymbolSize.DmtxSymbolSquareAuto if DmtxSymbolSize is not None else None
    base = []
    for shrink in (1, 2, 3, 4, 5, 6, 8, 10, 12):
        base.append({"shrink": shrink})
        if square is not None:
            base.append({"shrink": shrink, "shape": square})
    if square is not None:
        base.append({"shape": square})
        if "otsu" in candidate_name or "ad" in candidate_name or "sauvola" in candidate_name or "niblack" in candidate_name:
            for threshold in (8, 16, 24, 32, 48, 64, 96, 128, 160, 192):
                base.append({"threshold": threshold, "shape": square})
        for deviation in (5, 10, 20, 40, 60, 80):
            base.append({"deviation": deviation, "shape": square})
    base.append({})
    return [param for param in base if not any(value is None for value in param.values())]


def candidate_priority(candidate: Candidate) -> tuple[int, int]:
    name = candidate.name
    score = 0
    if "pad255" in name or "pad0" in name:
        score -= 20
    if "/x2" in name or "/x3" in name or "/x4" in name:
        score -= 15
    if "clahe" in name or "eq" in name or "sharp" in name:
        score -= 10
    if "otsu" in name or "ad" in name or "sauvola" in name or "niblack" in name:
        score += 5
    if "rot" in name:
        score += 2
    height, width = candidate.arr.shape[:2]
    return score, height * width


def plausible(text: str) -> bool:
    if not text or len(text.strip()) < 6:
        return False
    if not any(ch.isalnum() for ch in text):
        return False
    controls = sum(ord(ch) < 32 and ch not in "\r\n\t" for ch in text)
    return controls <= max(1, len(text) // 5)


def too_large(arr: np.ndarray) -> bool:
    height, width = arr.shape[:2]
    return height * width > 3_200_000


def normalize(arr: np.ndarray) -> np.ndarray:
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")


def as_u8(arr: np.ndarray) -> np.ndarray:
    return arr if arr.dtype == np.uint8 else normalize(arr)


def resize(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return arr
    height, width = arr.shape[:2]
    unique = len(np.unique(arr)) if arr.ndim == 2 else 999
    interp = cv2.INTER_NEAREST if unique <= 8 else cv2.INTER_CUBIC
    return cv2.resize(arr, (width * scale, height * scale), interpolation=interp)


def pad(arr: np.ndarray, value: int) -> np.ndarray:
    height, width = arr.shape[:2]
    amount = max(8, int(max(height, width) * 0.18))
    border = (value, value, value) if arr.ndim == 3 else value
    return cv2.copyMakeBorder(arr, amount, amount, amount, amount, cv2.BORDER_CONSTANT, value=border)


def rotate(arr: np.ndarray, degrees: float, value: int) -> np.ndarray:
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


def to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")


def format_params(params: dict[str, object]) -> str:
    if not params:
        return "default"
    parts = []
    for key, value in params.items():
        if key == "shape":
            value = getattr(value, "name", value)
        parts.append(f"{key}={value}")
    return ",".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
