from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader import ReaderOptions, read_datamatrix, read_image_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Data Matrix decoding on local image cases.")
    parser.add_argument("--cases", default="test_cases", help="Directory containing image files.")
    parser.add_argument("--mode", choices=["fast", "balanced", "aggressive"], default="aggressive")
    parser.add_argument("--csv", default="reports/benchmark_cases.csv")
    parser.add_argument("--json", default="reports/benchmark_cases.json")
    parser.add_argument("--max-seconds", type=float, default=12.0)
    args = parser.parse_args()

    case_dir = ROOT / args.cases
    images = sorted(
        [p for p in case_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}],
        key=natural_key,
    )
    if not images:
        print(f"No images found in {case_dir}")
        return 1

    options = ReaderOptions(mode=args.mode, max_seconds=args.max_seconds)
    rows: list[dict[str, object]] = []
    ok_count = 0

    for path in images:
        image = read_image_bytes(path.read_bytes())
        result = read_datamatrix(image, options)
        ok_count += int(result.ok)
        texts = [hit.text for hit in result.hits]
        engines = [hit.engine for hit in result.hits]
        rows.append(
            {
                "file": path.name,
                "ok": result.ok,
                "decoded_text": " | ".join(texts),
                "engine": " | ".join(engines),
                "variant": result.best_variant,
                "attempts": result.attempts,
                "elapsed_seconds": round(result.elapsed_seconds, 3),
            }
        )
        status = "OK" if result.ok else "FAIL"
        text = " | ".join(texts) if texts else "-"
        print(f"{status}\t{path.name}\t{text}\t{result.best_variant}\t{result.elapsed_seconds:.2f}s", flush=True)

    print(f"SUMMARY\t{ok_count}/{len(images)} decoded", flush=True)

    csv_path = ROOT / args.csv
    json_path = ROOT / args.json
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path.relative_to(ROOT)}", flush=True)
    print(f"Wrote {json_path.relative_to(ROOT)}", flush=True)
    return 0 if ok_count == len(images) else 2


def natural_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (10**9, path.name)


if __name__ == "__main__":
    raise SystemExit(main())
