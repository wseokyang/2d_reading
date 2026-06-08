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
from PIL import Image, ImageOps
import zxingcpp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from pylibdmtx.pylibdmtx import decode as dmtx_decode
    from pylibdmtx.wrapper import DmtxSymbolSize
except Exception:  # pragma: no cover - optional package
    dmtx_decode = None
    DmtxSymbolSize = None


@dataclass(frozen=True)
class Candidate:
    name: str
    image: Image.Image


@dataclass(frozen=True)
class EngineResult:
    ok: bool
    text: str
    engine: str
    variant: str
    attempts: int
    elapsed_seconds: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare zxing-cpp and libdmtx on the same preprocessing candidates.")
    parser.add_argument("--cases", default="test_cases2")
    parser.add_argument("--output", default="reports/test_cases2_engine_compare.csv")
    parser.add_argument("--json", default="reports/test_cases2_engine_compare.json")
    parser.add_argument("--per-engine-seconds", type=float, default=12.0)
    parser.add_argument("--candidate-limit", type=int, default=420)
    args = parser.parse_args()

    case_dir = ROOT / args.cases
    images = sorted(
        [path for path in case_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}],
        key=natural_key,
    )
    if not images:
        print(f"No images found in {case_dir}")
        return 1

    rows: list[dict[str, object]] = []
    zxing_ok = 0
    dmtx_ok = 0
    both_ok = 0

    for path in images:
        image = Image.open(path).convert("RGB")
        candidates = list(iter_candidates(image, limit=args.candidate_limit))

        zxing = scan_zxing(candidates, args.per_engine_seconds)
        dmtx = scan_dmtx(candidates, args.per_engine_seconds)
        opencv = scan_opencv(candidates[:80])

        zxing_ok += int(zxing.ok)
        dmtx_ok += int(dmtx.ok)
        both_ok += int(zxing.ok or dmtx.ok or opencv.ok)

        row = {
            "file": path.name,
            "any_ok": zxing.ok or dmtx.ok or opencv.ok,
            "zxing_ok": zxing.ok,
            "zxing_text": zxing.text,
            "zxing_variant": zxing.variant,
            "zxing_attempts": zxing.attempts,
            "zxing_seconds": round(zxing.elapsed_seconds, 3),
            "libdmtx_ok": dmtx.ok,
            "libdmtx_text": dmtx.text,
            "libdmtx_variant": dmtx.variant,
            "libdmtx_attempts": dmtx.attempts,
            "libdmtx_seconds": round(dmtx.elapsed_seconds, 3),
            "opencv_ok": opencv.ok,
            "opencv_text": opencv.text,
            "opencv_variant": opencv.variant,
        }
        rows.append(row)

        print(
            f"{path.name}\t"
            f"zxing={'OK ' + zxing.text if zxing.ok else 'FAIL'}\t"
            f"libdmtx={'OK ' + dmtx.text if dmtx.ok else 'FAIL'}\t"
            f"opencv={'OK ' + opencv.text if opencv.ok else 'FAIL'}",
            flush=True,
        )

    output = ROOT / args.output
    json_output = ROOT / args.json
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"SUMMARY\tzxing={zxing_ok}/{len(images)}\tlibdmtx={dmtx_ok}/{len(images)}\tany={both_ok}/{len(images)}")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {json_output.relative_to(ROOT)}")
    return 0 if both_ok == len(images) else 2


def natural_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (10**9, path.name)


def iter_candidates(image: Image.Image, limit: int) -> Iterable[Candidate]:
    rgb = image.convert("RGB")
    rgb_arr = np.array(rgb)
    gray = cv2.cvtColor(cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]

    priority = [
        ("priority-rgb", rgb),
        ("priority-rgb-x2", pil_resize(rgb, 2)),
        ("priority-rgb-pad-white", pil_pad(rgb, 0.18, "white")),
        ("priority-rgb-pad-black", pil_pad(rgb, 0.18, "black")),
        ("priority-gray", Image.fromarray(gray, mode="L")),
        ("priority-gray-autocontrast", ImageOps.autocontrast(Image.fromarray(gray, mode="L"))),
        ("priority-rgb-cv-rotate--9", arr_to_pil(rotate_arr(rgb_arr, -9, 255), "RGB")),
        ("priority-rgb-cv-x2-rotate-3", arr_to_pil(rotate_arr(resize_arr(rgb_arr, 2), 3, 255), "RGB")),
    ]

    bases: list[tuple[str, Image.Image]] = [
        ("rgb", rgb),
        ("gray", Image.fromarray(gray, mode="L")),
        ("gray-autocontrast", ImageOps.autocontrast(Image.fromarray(gray, mode="L"))),
        ("green", Image.fromarray(g, mode="L")),
        ("blue", Image.fromarray(b, mode="L")),
        ("green-red", Image.fromarray(normalize(g.astype("int16") - r.astype("int16")), mode="L")),
        ("blue-red", Image.fromarray(normalize(b.astype("int16") - r.astype("int16")), mode="L")),
        ("lab-a", Image.fromarray(lab[:, :, 1], mode="L")),
        ("cr", Image.fromarray(ycrcb[:, :, 1], mode="L")),
        ("cb", Image.fromarray(ycrcb[:, :, 2], mode="L")),
        ("hsv-saturation", Image.fromarray(hsv[:, :, 1], mode="L")),
    ]

    yielded = 0
    for name, priority_image in priority:
        yield Candidate(name, priority_image)
        yielded += 1
        if yielded >= limit:
            return

    for base_name, base_image in bases:
        for variant_name, variant in channel_variants(base_name, base_image):
            for candidate in geometric_variants(variant_name, variant):
                yield candidate
                yielded += 1
                if yielded >= limit:
                    return


def channel_variants(name: str, image: Image.Image) -> Iterable[tuple[str, Image.Image]]:
    yield name, image
    if image.mode != "L":
        return

    arr = np.array(image)
    yield f"{name}-equalized", Image.fromarray(cv2.equalizeHist(arr), mode="L")
    yield f"{name}-clahe", Image.fromarray(cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(arr), mode="L")
    yield f"{name}-median3", Image.fromarray(cv2.medianBlur(arr, 3), mode="L")
    yield f"{name}-sharp", Image.fromarray(cv2.addWeighted(arr, 2.0, cv2.GaussianBlur(arr, (0, 0), 1.0), -1.0, 0), mode="L")

    for source_name, source in (
        (name, arr),
        (f"{name}-clahe", cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(arr)),
    ):
        _, otsu = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield f"{source_name}-otsu", Image.fromarray(otsu, mode="L")
        yield f"{source_name}-otsu-inverted", Image.fromarray(255 - otsu, mode="L")
        if min(source.shape[:2]) > 25:
            for constant in (-5, 0, 5):
                adaptive = cv2.adaptiveThreshold(
                    source,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    25,
                    constant,
                )
                yield f"{source_name}-adaptive25-{constant}", Image.fromarray(adaptive, mode="L")
                yield f"{source_name}-adaptive25-{constant}-inverted", Image.fromarray(255 - adaptive, mode="L")


def geometric_variants(name: str, image: Image.Image) -> Iterable[Candidate]:
    arr = np.array(image)
    for scale in (1, 2, 3):
        scaled = resize_arr(arr, scale)
        for pad_value in (255, 0, 128, None):
            if pad_value is None:
                padded = scaled
                padded_name = f"{name}/x{scale}"
            else:
                padded = pad_arr(scaled, pad_value)
                padded_name = f"{name}/x{scale}/pad-{pad_value}"
            for angle in (0, -9, 9, -5, 5, 3, -3):
                if angle == 0:
                    yield Candidate(padded_name, arr_to_pil(padded, image.mode))
                else:
                    rotated = rotate_arr(padded, angle, pad_value if pad_value is not None else 255)
                    yield Candidate(f"{padded_name}/rotate-{angle}", arr_to_pil(rotated, image.mode))


def scan_zxing(candidates: list[Candidate], max_seconds: float) -> EngineResult:
    started = time.monotonic()
    attempts = 0
    for candidate in candidates:
        if time.monotonic() - started > max_seconds:
            break
        arr = np.array(candidate.image)
        for binarizer in (zxingcpp.Binarizer.LocalAverage, zxingcpp.Binarizer.GlobalHistogram, zxingcpp.Binarizer.FixedThreshold):
            attempts += 1
            try:
                decoded = zxingcpp.read_barcodes(
                    arr,
                    formats=zxingcpp.BarcodeFormat.DataMatrix,
                    try_rotate=True,
                    try_downscale=True,
                    try_invert=True,
                    binarizer=binarizer,
                )
            except Exception:
                decoded = []
            for item in decoded:
                text = item.text
                if not getattr(item, "error", None) and plausible(text):
                    return EngineResult(True, text, f"zxing-cpp/{binarizer.name}", candidate.name, attempts, time.monotonic() - started)
    return EngineResult(False, "", "zxing-cpp", "", attempts, time.monotonic() - started)


def scan_dmtx(candidates: list[Candidate], max_seconds: float) -> EngineResult:
    if dmtx_decode is None:
        return EngineResult(False, "", "pylibdmtx-unavailable", "", 0, 0.0)

    started = time.monotonic()
    attempts = 0
    for candidate in candidates:
        if time.monotonic() - started > max_seconds:
            break
        if not candidate.name.startswith("priority-"):
            continue
        for params in dmtx_priority_params(candidate.name):
            if time.monotonic() - started > max_seconds:
                break
            attempts += 1
            try:
                decoded = dmtx_decode(candidate.image, timeout=450, max_count=1, **params)
            except Exception:
                decoded = []
            for item in decoded:
                text = item.data.decode("utf-8", errors="replace")
                if plausible(text):
                    return EngineResult(True, text, f"pylibdmtx/{format_params(params)}", candidate.name, attempts, time.monotonic() - started)

    for candidate in candidates:
        if time.monotonic() - started > max_seconds:
            break
        for params in dmtx_params():
            if time.monotonic() - started > max_seconds:
                break
            attempts += 1
            try:
                decoded = dmtx_decode(candidate.image, timeout=320, max_count=1, **params)
            except Exception:
                decoded = []
            for item in decoded:
                text = item.data.decode("utf-8", errors="replace")
                if plausible(text):
                    return EngineResult(True, text, f"pylibdmtx/{format_params(params)}", candidate.name, attempts, time.monotonic() - started)
    return EngineResult(False, "", "pylibdmtx", "", attempts, time.monotonic() - started)


def scan_opencv(candidates: list[Candidate]) -> EngineResult:
    if not hasattr(cv2, "barcode"):
        return EngineResult(False, "", "opencv-unavailable", "", 0, 0.0)
    started = time.monotonic()
    attempts = 0
    detector = cv2.barcode.BarcodeDetector()
    for candidate in candidates:
        attempts += 1
        arr = np.array(candidate.image.convert("RGB"))
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        try:
            ok, decoded_info, decoded_type, _ = detector.detectAndDecodeWithType(bgr)
        except Exception:
            continue
        if not ok:
            continue
        for text in decoded_info:
            if plausible(text):
                return EngineResult(True, text, "opencv-barcode", candidate.name, attempts, time.monotonic() - started)
    return EngineResult(False, "", "opencv-barcode", "", attempts, time.monotonic() - started)


def dmtx_params() -> list[dict[str, object]]:
    square = DmtxSymbolSize.DmtxSymbolSquareAuto if DmtxSymbolSize is not None else None
    params: list[dict[str, object]] = [{}]
    for shrink in (1, 2, 3, 4, 5, 6, 7, 8):
        params.append({"shrink": shrink})
        if square is not None:
            params.append({"shrink": shrink, "shape": square})
    if square is not None:
        for threshold in (16, 32, 48, 64, 96, 128, 160, 192):
            params.append({"threshold": threshold, "shape": square})
    return params


def dmtx_priority_params(candidate_name: str) -> list[dict[str, object]]:
    square = DmtxSymbolSize.DmtxSymbolSquareAuto if DmtxSymbolSize is not None else None
    common = clean_params(
        [
            {},
            {"shrink": 2},
            {"shrink": 3},
            {"shrink": 4},
            {"shrink": 5},
            {"shrink": 2, "shape": square},
            {"shrink": 3, "shape": square},
            {"shrink": 4, "shape": square},
            {"shrink": 5, "shape": square},
        ]
    )
    shrink3 = clean_params([{"shrink": 3}, {"shrink": 3, "shape": square}])
    shrink4 = clean_params([{"shrink": 4}, {"shrink": 4, "shape": square}])
    shrink5 = clean_params([{"shrink": 5}, {"shrink": 5, "shape": square}])

    if "gray-autocontrast" in candidate_name:
        return shrink3
    if candidate_name.endswith("gray"):
        return shrink4
    if "cv-rotate" in candidate_name or "x2" in candidate_name:
        return shrink5
    if "pad-white" in candidate_name or "pad-black" in candidate_name:
        return common
    return common


def clean_params(params: list[dict[str, object]]) -> list[dict[str, object]]:
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


def normalize(arr: np.ndarray) -> np.ndarray:
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")


def resize_arr(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return arr
    height, width = arr.shape[:2]
    unique_count = len(np.unique(arr)) if arr.ndim == 2 else 999
    interpolation = cv2.INTER_NEAREST if unique_count <= 8 else cv2.INTER_CUBIC
    return cv2.resize(arr, (width * scale, height * scale), interpolation=interpolation)


def pad_arr(arr: np.ndarray, value: int) -> np.ndarray:
    height, width = arr.shape[:2]
    pad = max(8, int(max(height, width) * 0.22))
    return cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=value)


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


def arr_to_pil(arr: np.ndarray, mode: str) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")


def pil_resize(image: Image.Image, scale: int) -> Image.Image:
    width, height = image.size
    return image.resize((width * scale, height * scale), Image.Resampling.BICUBIC)


def pil_pad(image: Image.Image, fraction: float, color: str) -> Image.Image:
    width, height = image.size
    pad = max(8, int(max(width, height) * fraction))
    fill = (255, 255, 255) if color == "white" else (0, 0, 0)
    return ImageOps.expand(image, border=pad, fill=fill)


def plausible(text: str) -> bool:
    if not text:
        return False
    if len(text.strip()) < 6:
        return False
    if not any(ch.isalnum() for ch in text):
        return False
    control_count = sum(ord(ch) < 32 and ch not in "\r\n\t" for ch in text)
    return control_count <= max(1, len(text) // 5)


if __name__ == "__main__":
    raise SystemExit(main())
