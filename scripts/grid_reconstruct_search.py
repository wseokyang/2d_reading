from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
import zxingcpp

try:
    from pylibdmtx.pylibdmtx import decode as dmtx_decode
except Exception:  # pragma: no cover
    dmtx_decode = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = [
    "{0783A71B-ACC9-4179-A3AF-25FC9042B4FC}.png",
    "{E7CA298A-DC03-45A1-BABC-BEF2C7D89DFB}.png",
]


@dataclass(frozen=True)
class Candidate:
    text: str
    engine: str
    source: str
    transform: str
    size: int
    score: float
    attempts: int
    elapsed_seconds: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover hard Data Matrix crops by sampling and re-rendering module grids.")
    parser.add_argument("--cases", default="test_cases")
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--output", default="reports/grid_reconstruct_search.csv")
    parser.add_argument("--json", default="reports/grid_reconstruct_search.json")
    parser.add_argument("--save-hit", action="store_true")
    parser.add_argument("--use-dmtx", action="store_true", help="Also try libdmtx on re-rendered grids. Slower.")
    args = parser.parse_args()

    rows = []
    for file_name in args.files:
        path = ROOT / args.cases / file_name
        print(f"START\t{file_name}", flush=True)
        hit = search_one(path, args.seconds, args.save_hit, args.use_dmtx)
        rows.append(
            {
                "file": file_name,
                "ok": hit is not None,
                "decoded_text": hit.text if hit else "",
                "engine": hit.engine if hit else "",
                "source": hit.source if hit else "",
                "transform": hit.transform if hit else "",
                "size": hit.size if hit else "",
                "finder_score": round(hit.score, 4) if hit else "",
                "attempts": hit.attempts if hit else "",
                "elapsed_seconds": round(hit.elapsed_seconds, 3) if hit else args.seconds,
            }
        )
        if hit:
            print(
                f"OK\t{file_name}\t{hit.text}\t{hit.engine}\t{hit.source}\t{hit.transform}\t"
                f"n={hit.size}\tscore={hit.score:.3f}\t{hit.elapsed_seconds:.2f}s",
                flush=True,
            )
        else:
            print(f"FAIL\t{file_name}\t{args.seconds:.0f}s", flush=True)

    out = ROOT / args.output
    out_json = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)}")
    print(f"Wrote {out_json.relative_to(ROOT)}")
    return 0 if all(row["ok"] for row in rows) else 2


def search_one(path: Path, seconds: float, save_hit: bool, use_dmtx: bool) -> Candidate | None:
    started = time.monotonic()
    attempts = 0
    best_score = 0.0

    for source_name, source in source_images(path):
        for processed_name, processed in preprocess(source_name, source):
            for angle in (0, -1, 1, -2, 2, -3, 3, -5, 5, -7, 7, -9, 9, -12, 12):
                if time.monotonic() - started > seconds:
                    return None
                rotated = rotate_bound(processed, angle, 255) if angle else processed
                for size in (10, 12, 14, 16, 18, 20, 22):
                    for transform, matrix, score in sampled_matrices(rotated, size):
                        attempts += 1
                        best_score = max(best_score, score)
                        if attempts % 20000 == 0:
                            print(
                                f"  grid {path.name}: {attempts} attempts, best finder score {best_score:.3f}",
                                flush=True,
                            )
                        if score < 0.76:
                            continue
                        for corrected in (False, True):
                            clean = force_finder(matrix) if corrected else matrix
                            rendered = render_matrix(clean, module_size=14, quiet_modules=4)
                            decoded = try_decode(rendered, use_dmtx)
                            if decoded:
                                elapsed = time.monotonic() - started
                                full_transform = f"{processed_name}/angle{angle}/{transform}/finder_corrected={corrected}"
                                hit = Candidate(decoded[0], decoded[1], source_name, full_transform, size, score, attempts, elapsed)
                                maybe_save(path, rendered, save_hit, hit)
                                return hit
                        if time.monotonic() - started > seconds:
                            return None
    return None


def source_images(path: Path) -> Iterable[tuple[str, np.ndarray]]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    b, g, r = [bgr[:, :, index].astype("float32") for index in range(3)]
    total = r + g + b + 1.0

    yield "gray", gray
    yield "red", normalize(r)
    yield "green", normalize(g)
    yield "blue", normalize(b)
    yield "sat", hsv[:, :, 1]
    yield "lab-a", lab[:, :, 1]
    yield "lab-b", lab[:, :, 2]
    yield "cr", ycc[:, :, 1]
    yield "cb", ycc[:, :, 2]
    yield "green-red", normalize(g - r)
    yield "blue-red", normalize(b - r)
    yield "maxgb-red", normalize(np.maximum(g, b) - r)
    yield "red-maxgb", normalize(r - np.maximum(g, b))
    yield "green-over-red", normalize(g / (r + 1.0))
    yield "blue-over-red", normalize(b / (r + 1.0))
    yield "gb-over-red", normalize((g + b) / (r + 1.0))
    yield "red-over-gb", normalize(r / (g + b + 1.0))
    yield "green-share", normalize(g / total)
    yield "blue-share", normalize(b / total)

    pixels = rgb.reshape(-1, 3).astype("float32")
    pixels -= pixels.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(pixels, full_matrices=False)
    for index, vector in enumerate(vt):
        yield f"pca{index}", normalize((pixels @ vector).reshape(rgb.shape[:2]))


def preprocess(name: str, arr: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
    arr = ensure_u8(arr)
    yield name, arr
    yield f"{name}-eq", cv2.equalizeHist(arr)
    yield f"{name}-clahe4", cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(arr)
    yield f"{name}-clahe8", cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(arr)
    yield f"{name}-med3", cv2.medianBlur(arr, 3)
    yield f"{name}-med5", cv2.medianBlur(arr, 5)
    yield f"{name}-bilateral", cv2.bilateralFilter(arr, 5, 40, 40)
    yield f"{name}-sharp", cv2.addWeighted(arr, 2.7, cv2.GaussianBlur(arr, (0, 0), 1.2), -1.7, 0)
    for gamma in (0.45, 0.65, 1.6, 2.2):
        table = (np.power(np.arange(256) / 255.0, gamma) * 255).astype("uint8")
        yield f"{name}-gamma{gamma}", cv2.LUT(arr, table)


def sampled_matrices(arr: np.ndarray, size: int) -> Iterable[tuple[str, np.ndarray, float]]:
    height, width = arr.shape[:2]
    x_margins = margin_values(width)
    y_margins = margin_values(height)
    thresholds = ("otsu", "mean", "p42", "p46", "p50", "p54", "p58")

    for left in x_margins:
        for right in x_margins:
            if width - left - right < size * 3:
                continue
            for top in y_margins:
                for bottom in y_margins:
                    if height - top - bottom < size * 3:
                        continue
                    roi = arr[top : height - bottom, left : width - right]
                    sampled = cell_means(roi, size)
                    for threshold_name in thresholds:
                        threshold = threshold_value(sampled, threshold_name)
                        for invert in (False, True):
                            matrix = sampled < threshold if not invert else sampled >= threshold
                            matrix = matrix.astype(bool)
                            for rotation in range(4):
                                oriented = np.rot90(matrix, rotation)
                                score = finder_score(oriented)
                                yield (
                                    f"box{left},{top},{right},{bottom}/thr={threshold_name}/invert={invert}/rot{rotation}",
                                    oriented,
                                    score,
                                )


def margin_values(length: int) -> tuple[int, ...]:
    values = {0, 2, 4, 6, 8, 10, 12, 15, 18, 22}
    values.update({int(length * frac) for frac in (0.03, 0.06, 0.09, 0.12, 0.16)})
    return tuple(sorted(value for value in values if value < length // 3))


def cell_means(roi: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(roi, (size * 9, size * 9), interpolation=cv2.INTER_CUBIC)
    cells = resized.reshape(size, 9, size, 9).mean(axis=(1, 3))
    return cells


def threshold_value(sampled: np.ndarray, name: str) -> float:
    if name == "otsu":
        src = normalize(sampled)
        threshold, _ = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float(threshold) / 255.0 * (sampled.max() - sampled.min()) + sampled.min()
    if name == "mean":
        return float(sampled.mean())
    percentile = int(name[1:])
    return float(np.percentile(sampled, percentile))


def finder_score(matrix: np.ndarray) -> float:
    size = matrix.shape[0]
    expected_top = np.array([index % 2 == 0 for index in range(size)], dtype=bool)
    expected_right = np.array([index % 2 == 1 for index in range(size)], dtype=bool)
    expected_right[-1] = True
    checks = [
        matrix[:, 0] == True,
        matrix[-1, :] == True,
        matrix[0, :] == expected_top,
        matrix[:, -1] == expected_right,
    ]
    return float(np.concatenate(checks).mean())


def force_finder(matrix: np.ndarray) -> np.ndarray:
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


def render_matrix(matrix: np.ndarray, module_size: int, quiet_modules: int) -> np.ndarray:
    size = matrix.shape[0]
    image_size = (size + quiet_modules * 2) * module_size
    rendered = np.full((image_size, image_size), 255, dtype="uint8")
    for y in range(size):
        for x in range(size):
            if matrix[y, x]:
                y0 = (y + quiet_modules) * module_size
                x0 = (x + quiet_modules) * module_size
                rendered[y0 : y0 + module_size, x0 : x0 + module_size] = 0
    return rendered


def try_decode(arr: np.ndarray, use_dmtx: bool) -> tuple[str, str] | None:
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
                if plausible(text) and not getattr(item, "error", None):
                    return text, f"zxing-cpp/{binarizer.name}/pure={pure}"

    if use_dmtx and dmtx_decode is not None:
        image = Image.fromarray(arr, mode="L")
        for timeout in (20, 60):
            try:
                decoded = dmtx_decode(image, timeout=timeout, max_count=1)
            except Exception:
                decoded = []
            for item in decoded:
                text = item.data.decode("utf-8", errors="replace")
                if plausible(text):
                    return text, "pylibdmtx/re-rendered"
    return None


def rotate_bound(arr: np.ndarray, angle: float, value: int) -> np.ndarray:
    height, width = arr.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2 - width / 2
    matrix[1, 2] += new_height / 2 - height / 2
    return cv2.warpAffine(arr, matrix, (new_width, new_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=value)


def normalize(arr: np.ndarray) -> np.ndarray:
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")


def ensure_u8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    return normalize(arr)


def plausible(text: str) -> bool:
    if not text or len(text.strip()) < 6:
        return False
    if not any(ch.isalnum() for ch in text):
        return False
    controls = sum(ord(ch) < 32 and ch not in "\r\n\t" for ch in text)
    return controls <= max(1, len(text) // 5)


def maybe_save(path: Path, arr: np.ndarray, enabled: bool, hit: Candidate) -> None:
    if not enabled:
        return
    out_dir = ROOT / "reports" / "grid_reconstruct_hits"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_file = path.stem.strip("{}")[:8]
    safe_transform = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in hit.transform)[:160]
    Image.fromarray(arr, mode="L").save(out_dir / f"{safe_file}_{hit.text}_{safe_transform}.png")


if __name__ == "__main__":
    raise SystemExit(main())
