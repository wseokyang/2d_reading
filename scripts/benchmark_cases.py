from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_manifest(path: Path | None) -> dict[str, dict[str, object]]:
    """Read regression references only in the benchmark parent, never the reader."""
    if path is None:
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1 or not isinstance(document.get("cases"), dict):
        raise ValueError(f"Invalid manifest schema: {path}")
    cases = document["cases"]
    for name, case in cases.items():
        if not isinstance(case, dict):
            raise ValueError(f"Invalid manifest case: {name}")
        expected = case.get("expected_text")
        if expected is not None and (not isinstance(expected, str) or not expected):
            raise ValueError(f"Expected payload must be a nonempty string or null: {name}")
    return cases


def compare_payloads(ok: bool, texts: list[str], expected: str | None) -> tuple[bool | None, str]:
    if expected is None:
        return None, "unverified" if ok else "unread"
    if not ok:
        return False, "unread"
    # A second, different payload also fails an exact single-symbol reference.
    verified = bool(texts) and set(texts) == {expected}
    return verified, "match" if verified else "mismatch"


def run_case(path: Path, args: argparse.Namespace) -> dict[str, object]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--_worker", str(path),
        "--mode", args.mode, "--max-seconds", str(args.max_seconds),
        "--max-attempts", str(args.max_attempts),
    ]
    if args.no_libdmtx:
        command.append("--no-libdmtx")
    if args.glm:
        command.extend(["--glm", "--glm-timeout-seconds", str(args.glm_timeout_seconds)])
    timeout = args.case_timeout or (args.max_seconds + (args.glm_timeout_seconds if args.glm else 0) + 10.0)
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
        if completed.returncode:
            raise RuntimeError(f"Worker exited with status {completed.returncode}: {completed.stderr.strip()[-1000:]}")
        result = json.loads(completed.stdout)
        if not isinstance(result, dict) or not isinstance(result.get("decoded_texts"), list):
            raise ValueError("Worker returned invalid result JSON")
    except subprocess.TimeoutExpired:
        result = {"ok": False, "decoded_texts": [], "error": f"Hard per-case timeout after {timeout:g}s"}
    except (OSError, ValueError, RuntimeError) as exc:
        result = {"ok": False, "decoded_texts": [], "error": str(exc)}
    result["wall_seconds"] = round(time.monotonic() - started, 3)
    return result


def worker(args: argparse.Namespace) -> int:
    # Import decoder dependencies only in the isolated worker. A decoder crash or
    # native-library timeout cannot prevent the other cases being evaluated.
    sys.path.insert(0, str(ROOT))
    from reader import ReaderOptions, read_datamatrix, read_image_bytes

    options = dict(mode=args.mode, max_seconds=args.max_seconds, max_attempts=args.max_attempts,
                   use_libdmtx=not args.no_libdmtx)
    if args.glm:
        options.update(use_glm=True, glm_timeout_seconds=args.glm_timeout_seconds)
    result = read_datamatrix(read_image_bytes(Path(args._worker).read_bytes()), ReaderOptions(**options))
    print(json.dumps({
        "ok": result.ok,
        "decoded_texts": [hit.text for hit in result.hits],
        "engine": " | ".join(hit.engine for hit in result.hits),
        "variant": result.best_variant,
        "attempts": result.attempts,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "error": result.error,
        "assistance": getattr(result, "assistance", None),
    }, ensure_ascii=False))
    return 0


def write_reports(rows: list[dict[str, object]], csv_path: Path, json_path: Path) -> None:
    # Checkpoint after every completed case so an interrupted run retains evidence.
    for path in (csv_path, json_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(json_path)
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    temporary.replace(csv_path)


def natural_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (10**9, path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode each case in an isolated process and compare exact reference strings.")
    parser.add_argument("--cases", default="test_cases", help="Directory containing image files.")
    parser.add_argument("--manifest", help="Reference JSON; defaults to manifest.json in the cases directory when present.")
    parser.add_argument("--only", action="append", default=[], metavar="FILENAME", help="Run only this filename; repeat for more cases.")
    parser.add_argument("--mode", choices=["fast", "balanced", "aggressive"], default="aggressive")
    parser.add_argument("--csv", default="reports/benchmark_cases.csv")
    parser.add_argument("--json", default="reports/benchmark_cases.json")
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=650)
    parser.add_argument("--case-timeout", type=float, help="Hard subprocess timeout; default is local + GLM budgets + 10s.")
    parser.add_argument("--no-libdmtx", action="store_true")
    parser.add_argument("--glm", action="store_true", help="Enable API assistance using GLM_* environment settings.")
    parser.add_argument("--glm-timeout-seconds", type=float, help="Override GLM_TIMEOUT_SECONDS (default 20).")
    parser.add_argument("--allow-unverified", action="store_true", help="Allow decoded cases without reference labels to pass. Mismatches still fail.")
    parser.add_argument("--_worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.glm and not args._worker:
        from dotenv import load_dotenv

        # Match the app: explicit shell settings take precedence, and children
        # inherit the resolved settings without putting secrets on the CLI.
        load_dotenv(ROOT / ".env", override=False)
    if args.glm_timeout_seconds is None:
        try:
            args.glm_timeout_seconds = float(os.getenv("GLM_TIMEOUT_SECONDS", "20"))
        except ValueError:
            parser.error("GLM_TIMEOUT_SECONDS must be a number.")
    limits = (args.max_seconds, args.glm_timeout_seconds, args.case_timeout or 1.0)
    if any(not math.isfinite(value) or value <= 0 for value in limits) or args.max_attempts <= 0:
        parser.error("Time and attempt limits must be finite positive numbers.")
    if args._worker:
        return worker(args)

    case_dir = ROOT / args.cases
    if not case_dir.is_dir():
        parser.error(f"Cases directory does not exist: {case_dir}")
    manifest_path = ROOT / args.manifest if args.manifest else case_dir / "manifest.json"
    if not args.manifest and not manifest_path.exists():
        manifest_path = None
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    images = sorted((path for path in case_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)
    if args.only:
        missing = set(args.only) - {path.name for path in images}
        if missing:
            parser.error(f"Requested images not found: {', '.join(sorted(missing))}")
        images = [path for path in images if path.name in args.only]
    if not images:
        parser.error(f"No images found in {case_dir}")
    csv_path, json_path = ROOT / args.csv, ROOT / args.json
    rows: list[dict[str, object]] = []
    for path in images:
        result = run_case(path, args)
        reference = manifest.get(path.name, {})
        expected = reference.get("expected_text")
        texts = result["decoded_texts"]
        verified, status = compare_payloads(bool(result["ok"]), texts, expected)
        row = {
            "file": path.name,
            "ok": bool(result["ok"]),
            "verified": verified,
            "verification_status": status,
            "expected_text": expected,
            "expected_provenance": reference.get("provenance", "unknown"),
            "decoded_text": " | ".join(texts),
            "decoded_texts": texts,
            "engine": result.get("engine", ""),
            "variant": result.get("variant", ""),
            "attempts": result.get("attempts", 0),
            "elapsed_seconds": result.get("elapsed_seconds", result["wall_seconds"]),
            "wall_seconds": result["wall_seconds"],
            "error": result.get("error"),
            "assistance": result.get("assistance"),
        }
        rows.append(row)
        write_reports(rows, csv_path, json_path)
        print(f"{status.upper()}\t{path.name}\t{row['decoded_text'] or '-'}\t{row['variant']}\t{row['elapsed_seconds']:.2f}s", flush=True)
        if row["error"]:
            print(f"  ERROR: {row['error']}", flush=True)

    decoded = sum(row["ok"] for row in rows)
    verified = sum(row["verified"] is True for row in rows)
    known = sum(row["expected_text"] is not None for row in rows)
    print(f"SUMMARY\t{decoded}/{len(rows)} decoded; {verified}/{known} exact reference matches; {len(rows) - known} unknown references", flush=True)
    print("Reference matches are regression checks, not independent ground-truth certification.", flush=True)
    print(f"Wrote {csv_path}\nWrote {json_path}", flush=True)
    passed = all(row["ok"] and (row["verified"] is True or (args.allow_unverified and row["verified"] is None)) for row in rows)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
