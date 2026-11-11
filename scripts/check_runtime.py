from __future__ import annotations

import argparse
import ctypes.util
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from PIL import Image
try:
    from pylibdmtx.pylibdmtx import decode, encode
except ImportError:
    decode = encode = None

ROOT = Path(__file__).resolve().parents[1]

PACKAGES = (
    "numpy",
    "opencv-python-headless",
    "pandas",
    "pillow",
    "pylibdmtx",
    "pyzxing",
    "scikit-image",
    "scipy",
    "streamlit",
    "zxing-cpp",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Data Matrix reader runtime reproducibility.")
    parser.add_argument("--benchmark", action="store_true", help="Also run the local test_cases benchmark.")
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--expected-successes", type=int, default=26,
                        help="Minimum number of exact reference matches in test_cases.")
    args = parser.parse_args()

    info = runtime_info()
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)

    if not info["libdmtx_found"]:
        print("\nERROR: native libdmtx was not found. pylibdmtx will not be reproducible.", file=sys.stderr)
        return 2

    if not dmtx_self_test():
        print("\nERROR: pylibdmtx self-test failed.", file=sys.stderr)
        return 2

    if args.benchmark:
        return run_benchmark(args.max_seconds, args.expected_successes)

    return 0


def runtime_info() -> dict[str, object]:
    libdmtx_path = ctypes.util.find_library("dmtx")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "libdmtx_find_library": libdmtx_path,
        "libdmtx_found": bool(libdmtx_path),
        "java": command_version(["java", "-version"]),
        "packages": {name: package_version(name) for name in PACKAGES},
    }


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def command_version(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return "not installed"
    output = (completed.stdout + completed.stderr).strip().splitlines()
    return output[0] if output else "unknown"


def dmtx_self_test() -> bool:
    if decode is None or encode is None:
        return False
    payload = b"REPRODUCIBILITY-CHECK-001"
    encoded = encode(payload)
    image = Image.frombytes("RGB", (encoded.width, encoded.height), encoded.pixels)
    decoded = decode(image, timeout=500, max_count=1)
    texts = [item.data for item in decoded]
    print("pylibdmtx_self_test:", texts, flush=True)
    return payload in texts


def run_benchmark(max_seconds: float, expected_successes: int) -> int:
    json_path = ROOT / "reports" / "runtime_check_test_cases.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_cases.py"),
        "--cases",
        "test_cases",
        "--mode",
        "aggressive",
        "--max-seconds",
        str(max_seconds),
        "--csv",
        "reports/runtime_check_test_cases.csv",
        "--json",
        str(json_path.relative_to(ROOT)),
    ]
    # Do not accept a previous run's report if argument validation or startup fails.
    json_path.unlink(missing_ok=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode not in (0, 2) or not json_path.exists():
        print("runtime_benchmark_summary: benchmark did not produce a valid report", flush=True)
        return 2
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    decoded = sum(1 for row in rows if row.get("ok"))
    verified = sum(1 for row in rows if row.get("verified") is True)
    total = len(rows)
    print(f"runtime_benchmark_summary: {decoded}/{total} decoded; {verified}/{total} exact reference matches; expected >= {expected_successes}", flush=True)
    return 0 if verified >= expected_successes else 2


if __name__ == "__main__":
    raise SystemExit(main())
