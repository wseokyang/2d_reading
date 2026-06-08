from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pylibdmtx.pylibdmtx import decode as dmtx_decode

from reader import ReaderOptions, expand_candidate, image_to_png, iter_variants, read_datamatrix, read_image_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description="Explore extra preprocessing candidates on failed cases.")
    parser.add_argument("--cases", default="test_cases")
    parser.add_argument("--previous", default="reports/benchmark_cases.csv")
    parser.add_argument("--output", default="reports/preprocess_search.csv")
    parser.add_argument("--per-image-seconds", type=float, default=18.0)
    parser.add_argument("--variant-limit", type=int, default=80)
    args = parser.parse_args()

    case_dir = ROOT / args.cases
    files = failed_files(ROOT / args.previous)
    if not files:
        files = [path.name for path in sorted(case_dir.glob("*")) if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for file_name in files:
        path = case_dir / file_name
        started = time.monotonic()
        result = search_file(path, args.per_image_seconds, args.variant_limit)
        rows.append(result)
        status = "OK" if result["ok"] else "FAIL"
        print(
            f"{status}\t{file_name}\t{result['decoded_text'] or '-'}\t{result['engine']}\t{result['variant']}\t{result['elapsed_seconds']}",
            flush=True,
        )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(row["ok"] for row in rows)
    print(f"SUMMARY\t{ok_count}/{len(rows)} newly decoded")
    print(f"Wrote {output_path.relative_to(ROOT)}")
    return 0 if ok_count == len(rows) else 2


def failed_files(previous: Path) -> list[str]:
    if not previous.exists():
        return []
    with previous.open(newline="", encoding="utf-8") as handle:
        return [row["file"] for row in csv.DictReader(handle) if row.get("ok") == "False"]


def search_file(path: Path, per_image_seconds: float, variant_limit: int) -> dict[str, object]:
    started = time.monotonic()
    image = read_image_bytes(path.read_bytes())

    manual_options = [
        ReaderOptions(mode="aggressive", max_seconds=min(6.0, per_image_seconds), max_attempts=1200),
        ReaderOptions(mode="aggressive", brightness=35, contrast=1.8, saturation=0.5, sharpness=2.2, max_seconds=4, max_attempts=900),
        ReaderOptions(mode="aggressive", brightness=-20, contrast=2.4, saturation=0.2, sharpness=2.8, max_seconds=4, max_attempts=900),
        ReaderOptions(mode="aggressive", brightness=45, contrast=2.0, saturation=1.6, sharpness=2.0, max_seconds=4, max_attempts=900),
    ]
    for options in manual_options:
        if time.monotonic() - started > per_image_seconds:
            break
        direct = read_datamatrix(image, options)
        if direct.ok:
            return row(path, direct.hits[0].text, direct.hits[0].engine, direct.best_variant, 0, started, True)

    attempts = 0
    for variant_index, (variant_name, arr) in enumerate(iter_variants(image, "aggressive")):
        if variant_index >= variant_limit or time.monotonic() - started > per_image_seconds:
            break
        for expanded_name, candidate in expand_candidate(variant_name, arr, "balanced"):
            if time.monotonic() - started > per_image_seconds:
                break
            h, w = candidate.shape[:2]
            if h * w > 900_000:
                continue
            for params in dmtx_param_sets():
                attempts += 1
                text = decode_dmtx(candidate, params)
                if text:
                    return row(path, text, f"pylibdmtx/{params}", expanded_name, attempts, started, True)

    return row(path, "", "", "", attempts, started, False)


def dmtx_param_sets() -> list[dict[str, int]]:
    params: list[dict[str, int]] = [{}]
    for threshold in (8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192):
        params.append({"threshold": threshold})
    for deviation in (10, 20, 30, 40, 60, 80):
        params.append({"deviation": deviation})
    for shrink in (2, 3):
        params.append({"shrink": shrink})
    for min_edge in (5, 10, 20):
        params.append({"min_edge": min_edge})
    return params


def decode_dmtx(candidate, params: dict[str, int]) -> str:
    if candidate.ndim == 2:
        image = Image.fromarray(candidate, mode="L")
    else:
        image = Image.fromarray(candidate)

    try:
        decoded = dmtx_decode(image, timeout=220, max_count=1, **params)
    except Exception:
        return ""
    if not decoded:
        return ""
    return decoded[0].data.decode("utf-8", errors="replace")


def row(path: Path, text: str, engine: str, variant: str, attempts: int, started: float, ok: bool) -> dict[str, object]:
    return {
        "file": path.name,
        "ok": ok,
        "decoded_text": text,
        "engine": engine,
        "variant": variant,
        "attempts": attempts,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


if __name__ == "__main__":
    raise SystemExit(main())
