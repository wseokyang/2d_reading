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
    "{E7CA298A-DC03-45A1-BABC-BEF2C7D89DFB}.png",
]


@dataclass(frozen=True)
class Candidate:
    method: str
    name: str
    arr: np.ndarray


@dataclass(frozen=True)
class Hit:
    text: str
    engine: str
    method: str
    candidate: str
    attempts: int
    elapsed_seconds: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce named preprocessing categories against hard Data Matrix crops.")
    parser.add_argument("--cases", default="test_cases")
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--output", default="reports/named_preprocessing_probe.csv")
    parser.add_argument("--json", default="reports/named_preprocessing_probe.json")
    parser.add_argument("--save-hit", action="store_true")
    args = parser.parse_args()

    rows = []
    for file_name in args.files:
        path = ROOT / args.cases / file_name
        print(f"START\t{file_name}", flush=True)
        hit = search_one(path, args.seconds, args.save_hit)
        rows.append(
            {
                "file": file_name,
                "ok": hit is not None,
                "decoded_text": hit.text if hit else "",
                "engine": hit.engine if hit else "",
                "method": hit.method if hit else "",
                "candidate": hit.candidate if hit else "",
                "attempts": hit.attempts if hit else "",
                "elapsed_seconds": round(hit.elapsed_seconds, 3) if hit else args.seconds,
            }
        )
        if hit:
            print(
                f"OK\t{file_name}\t{hit.text}\t{hit.engine}\t{hit.method}\t{hit.candidate}\t{hit.elapsed_seconds:.2f}s",
                flush=True,
            )
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


def search_one(path: Path, seconds: float, save_hit: bool) -> Hit | None:
    started = time.monotonic()
    attempts = 0
    candidates = list(named_candidates(path))

    # zxing-cpp is fast, so scan all named preprocessing variants first.
    for candidate in candidates:
        if time.monotonic() - started > seconds:
            break
        for decode_candidate in decode_geometry(candidate):
            if too_large(decode_candidate.arr):
                continue
            attempts += 1
            hit = try_zxing(decode_candidate, attempts, started)
            if hit:
                maybe_save(path, decode_candidate, save_hit)
                return hit
        if attempts % 500 == 0 and attempts:
            print(f"  zxing {path.name}: {attempts} attempts", flush=True)

    # libdmtx is slower; prioritize exactly the listed method families and their padded variants.
    for candidate in candidates:
        if time.monotonic() - started > seconds:
            break
        for decode_candidate in decode_geometry(candidate, compact=True):
            if time.monotonic() - started > seconds or too_large(decode_candidate.arr):
                continue
            attempts += 1
            hit = try_dmtx(decode_candidate, attempts, started)
            if hit:
                maybe_save(path, decode_candidate, save_hit)
                return hit
        if attempts % 250 == 0 and attempts:
            print(f"  dmtx {path.name}: {attempts} attempts", flush=True)

    return None


def named_candidates(path: Path) -> Iterable[Candidate]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    yield Candidate("original", "original", rgb)
    yield Candidate("grayscale", "grayscale", gray)
    yield Candidate("histogram_equalization", "histogram_equalization", cv2.equalizeHist(gray))
    yield Candidate("clahe", "clahe_clip2_tile8", clahe(gray, 2.0, 8))
    yield Candidate("gaussian_blur_3x3", "gaussian_blur_3x3", cv2.GaussianBlur(gray, (3, 3), 0))
    yield Candidate("gaussian_blur_5x5", "gaussian_blur_5x5", cv2.GaussianBlur(gray, (5, 5), 0))
    yield Candidate("median_blur_3x3", "median_blur_3x3", cv2.medianBlur(gray, 3))
    yield Candidate("median_blur_5x5", "median_blur_5x5", cv2.medianBlur(gray, 5))
    yield Candidate("bilateral_filter", "bilateral_filter_d5", cv2.bilateralFilter(gray, 5, 45, 45))
    yield Candidate("nlm_denoising", "nlm_denoising_h14", cv2.fastNlMeansDenoising(gray, None, 14, 7, 21))

    for gamma in (0.35, 0.5, 0.7, 1.4, 1.8, 2.4):
        gamma_img = gamma_correction(gray, gamma)
        yield Candidate("gamma_correction", f"gamma_correction_{gamma}", gamma_img)
        yield from threshold_family("gamma_then_threshold", f"gamma_then_threshold_{gamma}", gamma_img)

    yield from threshold_family("threshold_otsu", "threshold_otsu", gray, otsu_only=True)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield Candidate("threshold_otsu_inv", "threshold_otsu_inv", 255 - otsu)
    yield from adaptive_family("adaptive_gaussian", gray, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, inverted=False)
    yield from adaptive_family("adaptive_gaussian_inv", gray, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, inverted=True)
    yield from adaptive_family("adaptive_mean", gray, cv2.ADAPTIVE_THRESH_MEAN_C, inverted=False)
    yield from adaptive_family("adaptive_mean_inv", gray, cv2.ADAPTIVE_THRESH_MEAN_C, inverted=True)

    for kernel in ((3, 3), (5, 5)):
        blurred = cv2.GaussianBlur(gray, kernel, 0)
        yield from threshold_family("blur_then_threshold", f"blur_then_threshold_{kernel[0]}x{kernel[1]}", blurred)

    for clip, tile in ((2.0, 8), (3.0, 8), (4.0, 4), (5.0, 4)):
        cl = clahe(gray, clip, tile)
        yield Candidate("clahe", f"clahe_clip{clip}_tile{tile}", cl)
        yield from threshold_family("clahe_then_threshold", f"clahe_then_threshold_clip{clip}_tile{tile}", cl)

    denoised = cv2.fastNlMeansDenoising(gray, None, 18, 7, 21)
    yield Candidate("nlm_denoising", "nlm_denoising_h18", denoised)
    yield from threshold_family("denoise_then_threshold", "denoise_then_threshold_h18", denoised)
    denoise_clahe = clahe(denoised, 4.0, 4)
    yield Candidate("denoise_then_clahe_then_threshold", "denoise_then_clahe", denoise_clahe)
    yield from threshold_family("denoise_then_clahe_then_threshold", "denoise_then_clahe_then_threshold", denoise_clahe)

    for angle in (90, 180, 270, 45):
        rotated = rotate_bound(gray, angle, 255)
        yield Candidate(f"rotate_{angle}", f"rotate_{angle}", rotated)
        yield from threshold_family("rotate_then_threshold", f"rotate_then_threshold_{angle}", rotated)
        for scale in (1.5, 2.0, 3.0):
            scaled = resize_float(rotated, scale, binary=False)
            yield from threshold_family(
                "rotate_then_scale_then_threshold",
                f"rotate_then_scale_then_threshold_{angle}_{scale}x",
                scaled,
            )

    for scale in (0.5, 0.75, 1.5, 2.0, 3.0):
        scaled = resize_float(gray, scale, binary=False)
        scale_label = str(scale).replace(".", "_")
        yield Candidate(f"scale_{scale_label}x", f"scale_{scale}x", scaled)
        yield from threshold_family("scale_then_threshold", f"scale_then_threshold_{scale}x", scaled)
        cl = clahe(scaled, 3.0, 8)
        yield Candidate("scale_then_clahe", f"scale_then_clahe_{scale}x", cl)
        yield from threshold_family("scale_then_clahe_then_threshold", f"scale_then_clahe_then_threshold_{scale}x", cl)


def threshold_family(method: str, prefix: str, src: np.ndarray, otsu_only: bool = False) -> Iterable[Candidate]:
    _, otsu = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield Candidate(method, f"{prefix}_otsu", otsu)
    if otsu_only:
        return
    yield Candidate(method, f"{prefix}_otsu_inv", 255 - otsu)
    for threshold in (64, 80, 96, 112, 128, 144, 160, 176):
        _, binary = cv2.threshold(src, threshold, 255, cv2.THRESH_BINARY)
        yield Candidate(method, f"{prefix}_fixed{threshold}", binary)
        yield Candidate(method, f"{prefix}_fixed{threshold}_inv", 255 - binary)


def adaptive_family(method: str, src: np.ndarray, adaptive_method: int, inverted: bool) -> Iterable[Candidate]:
    for block_size in (11, 15, 21, 31, 41, 61):
        if min(src.shape[:2]) <= block_size:
            continue
        for constant in (-13, -9, -5, -1, 0, 3, 7, 11, 15):
            binary = cv2.adaptiveThreshold(src, 255, adaptive_method, cv2.THRESH_BINARY, block_size, constant)
            if inverted:
                binary = 255 - binary
            yield Candidate(method, f"{method}_block{block_size}_c{constant}", binary)


def decode_geometry(candidate: Candidate, compact: bool = False) -> Iterable[Candidate]:
    pad_fracs = (0.0, 0.12, 0.24) if not compact else (0.12, 0.24)
    pad_values = (255, 0, 128)
    scales = (1.0, 1.5, 2.0, 3.0) if not compact else (1.0, 2.0)

    for scale in scales:
        scaled = resize_float(candidate.arr, scale, binary=is_binary(candidate.arr))
        for pad_frac in pad_fracs:
            if pad_frac == 0:
                yield Candidate(candidate.method, f"{candidate.name}/decode_scale{scale}", scaled)
                continue
            for pad_value in pad_values:
                padded = pad(scaled, pad_frac, pad_value)
                yield Candidate(candidate.method, f"{candidate.name}/decode_scale{scale}/pad{pad_value}_{pad_frac}", padded)


def try_zxing(candidate: Candidate, attempts: int, started: float) -> Hit | None:
    arr = ensure_u8(candidate.arr)
    for binarizer in (
        zxingcpp.Binarizer.LocalAverage,
        zxingcpp.Binarizer.GlobalHistogram,
        zxingcpp.Binarizer.FixedThreshold,
        zxingcpp.Binarizer.BoolCast,
    ):
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
                    if plausible(text) and not getattr(item, "error", None):
                        return Hit(
                            text,
                            f"zxing-cpp/{binarizer.name}/pure={pure}/downscale={downscale}",
                            candidate.method,
                            candidate.name,
                            attempts,
                            time.monotonic() - started,
                        )
    return None


def try_dmtx(candidate: Candidate, attempts: int, started: float) -> Hit | None:
    if dmtx_decode is None:
        return None
    image = arr_to_pil(ensure_u8(candidate.arr))
    square = DmtxSymbolSize.DmtxSymbolSquareAuto if DmtxSymbolSize is not None else None
    params: list[dict[str, object]] = [{}]
    for shrink in (1, 2, 3, 4, 5, 6, 8, 10):
        params.append({"shrink": shrink})
        if square is not None:
            params.append({"shrink": shrink, "shape": square})
    if square is not None:
        params.append({"shape": square})
        for threshold in (16, 32, 48, 64, 96, 128, 160, 192, 220):
            params.append({"threshold": threshold, "shape": square})

    for param in params:
        if any(value is None for value in param.values()):
            continue
        try:
            decoded = dmtx_decode(image, timeout=80, max_count=1, **param)
        except Exception:
            decoded = []
        for item in decoded:
            text = item.data.decode("utf-8", errors="replace")
            if plausible(text):
                return Hit(
                    text,
                    f"pylibdmtx/{format_params(param)}",
                    candidate.method,
                    candidate.name,
                    attempts,
                    time.monotonic() - started,
                )
    return None


def clahe(arr: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)).apply(ensure_u8(arr))


def gamma_correction(arr: np.ndarray, gamma: float) -> np.ndarray:
    table = (np.power(np.arange(256) / 255.0, gamma) * 255).astype("uint8")
    return cv2.LUT(ensure_u8(arr), table)


def resize_float(arr: np.ndarray, scale: float, binary: bool) -> np.ndarray:
    if scale == 1.0:
        return arr
    height, width = arr.shape[:2]
    interpolation = cv2.INTER_NEAREST if binary else cv2.INTER_CUBIC
    return cv2.resize(arr, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=interpolation)


def rotate_bound(arr: np.ndarray, angle: float, value: int) -> np.ndarray:
    height, width = arr.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2 - width / 2
    matrix[1, 2] += new_height / 2 - height / 2
    border = (value, value, value) if arr.ndim == 3 else value
    return cv2.warpAffine(
        arr,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_NEAREST if is_binary(arr) else cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def pad(arr: np.ndarray, fraction: float, value: int) -> np.ndarray:
    height, width = arr.shape[:2]
    pad_size = max(8, int(max(height, width) * fraction))
    border = (value, value, value) if arr.ndim == 3 else value
    return cv2.copyMakeBorder(arr, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=border)


def ensure_u8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")


def is_binary(arr: np.ndarray) -> bool:
    if arr.ndim != 2:
        return False
    return len(np.unique(arr)) <= 4


def too_large(arr: np.ndarray) -> bool:
    height, width = arr.shape[:2]
    return height * width > 2_800_000


def arr_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")


def plausible(text: str) -> bool:
    if not text or len(text.strip()) < 6:
        return False
    if not any(ch.isalnum() for ch in text):
        return False
    controls = sum(ord(ch) < 32 and ch not in "\r\n\t" for ch in text)
    return controls <= max(1, len(text) // 5)


def format_params(params: dict[str, object]) -> str:
    if not params:
        return "default"
    parts = []
    for key, value in params.items():
        if key == "shape":
            value = getattr(value, "name", value)
        parts.append(f"{key}={value}")
    return ",".join(parts)


def maybe_save(path: Path, candidate: Candidate, enabled: bool) -> None:
    if not enabled:
        return
    out_dir = ROOT / "reports" / "named_preprocessing_hits"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_file = path.stem.strip("{}")[:8]
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in candidate.name)[:180]
    arr_to_pil(ensure_u8(candidate.arr)).save(out_dir / f"{safe_file}_{safe_name}.png")


if __name__ == "__main__":
    raise SystemExit(main())
