"""Reproducible simulated API protocol test with real local barcode decoding.

This is NOT a GLM inference benchmark. A localhost server returns one identical,
manually specified preprocessing plan for every request. It never accesses the
test manifest, filenames, encoded values, or decoder results.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
import threading
import time

from PIL import Image

from benchmark_cases import IMAGE_SUFFIXES, ROOT, compare_payloads, load_manifest, run_case, write_reports


RECIPES = [
    {"bbox": [0, 0, 1, 1], "channel": channel, "blur_sigma": sigma,
     "threshold": threshold, "invert": False, "grid_size": None}
    for channel, sigma, threshold in (
        ("blue", 3, None), ("yellow_blue", 3, None), ("gray", 3, None), ("green", 2, 110)
    )
]
SIMULATED_MODEL = "simulated-preprocessing-plan-not-real-glm"


class RecipeHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        self.server.api_calls += 1
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/v1/chat/completions"
            assert body["model"] == SIMULATED_MODEL
            assert not self.headers.get("Authorization")
            assert "reasoning_effort" not in body
            image_block = body["messages"][1]["content"][1]
            assert image_block["type"] == "image_url"
            url = image_block["image_url"]["url"]
            assert url.startswith("data:image/png;base64,")
            with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1], validate=True))) as image:
                image.load()
                assert image.mode == "RGB" and max(image.size) <= 1024
            status = 200
            response = {"model": SIMULATED_MODEL, "choices": [{
                "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps({"hints": RECIPES})}
            }]}
        except (AssertionError, KeyError, IndexError, TypeError, ValueError, OSError):
            self.server.invalid_requests += 1
            status, response = 400, {"error": "Invalid simulated protocol request"}
        raw = json.dumps(response).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@contextmanager
def simulated_api():
    server = ThreadingHTTPServer(("127.0.0.1", 0), RecipeHandler)
    server.daemon_threads = True
    server.api_calls = 0
    server.invalid_requests = 0
    settings = {
        "GLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
        "GLM_MODEL": SIMULATED_MODEL,
        "GLM_API_KEY": "", "ZAI_API_KEY": "", "GLM_HEADERS_JSON": "{}",
        "GLM_REASONING_EFFORT": "", "GLM_TIMEOUT_SECONDS": "2",
    }
    original = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="test_cases")
    parser.add_argument("--mode", choices=["fast", "balanced", "aggressive"], default="aggressive")
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--max-attempts", type=int, default=650)
    parser.add_argument("--no-libdmtx", action="store_true")
    parser.add_argument("--report-prefix", default="reports/glm_simulated_flow")
    args = parser.parse_args()
    if args.max_seconds <= 0 or args.max_attempts <= 0:
        parser.error("Time and attempt budgets must be positive")
    args.glm, args.glm_timeout_seconds, args.case_timeout = True, 2.0, None
    case_dir = ROOT / args.cases
    manifest = load_manifest(case_dir / "manifest.json")
    images = sorted(path for path in case_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        parser.error("No case images found")
    csv_path = (ROOT / args.report_prefix).with_suffix(".csv")
    json_path = (ROOT / args.report_prefix).with_suffix(".json")
    rows = []
    started = time.monotonic()
    print("SIMULATION ONLY: shared manual recipes over localhost HTTP; real local ECC decoding.", flush=True)
    with simulated_api() as server:
        for image in images:
            before = server.api_calls
            result = run_case(image, args)
            expected = manifest.get(image.name, {}).get("expected_text")
            verified, verification = compare_payloads(result["ok"], result["decoded_texts"], expected)
            assistance = result.get("assistance") or {}
            calls = server.api_calls - before
            path = "direct" if assistance.get("status") == "not-needed" else (
                "simulated-hint" if assistance.get("used") else "local-fallback"
            )
            row = {
                "evaluation": "simulated_api_protocol_not_real_glm_inference",
                "file": image.name, "ok": result["ok"], "verified": verified,
                "verification_status": verification, "expected_text": expected,
                "decoded_texts": result["decoded_texts"], "path": path,
                "api_calls": calls, **{key: value for key, value in result.items() if key not in {"ok", "decoded_texts"}},
            }
            rows.append(row)
            # Reuse the benchmark's atomic checkpoint writer; the final JSON
            # below adds explicit simulation metadata and aggregate counts.
            write_reports(rows, csv_path, json_path)
            print(f"{len(rows):02d}/{len(images)} {verification}: {image.name} ({path})", flush=True)
        summary = {
            "total_cases": len(rows), "decoded": sum(row["ok"] for row in rows),
            "exact_matches": sum(row["verified"] is True for row in rows),
            "api_calls": server.api_calls, "invalid_api_requests": server.invalid_requests,
            "direct_skips": sum(row["path"] == "direct" for row in rows),
            "hint_used": sum(row["path"] == "simulated-hint" for row in rows),
            "local_fallback": sum(row["path"] == "local-fallback" for row in rows),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    document = {
        "evaluation": "simulated_api_protocol_not_real_glm_inference",
        "real_glm_inference": False,
        "description": "One shared manual preprocessing plan for every image; no filename or payload lookup in the API server or reader.",
        "recipes": RECIPES, "summary": summary, "cases": rows,
    }
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(json_path)
    print(json.dumps(summary), flush=True)
    return 0 if summary["exact_matches"] == len(rows) and not summary["invalid_api_requests"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
