"""HTTP protocol + real local decoding tests, not a GLM inference benchmark.

The local server returns the same generic preprocessing recipes for every image.
It does not contain payloads, inspect filenames, or pretend to run a vision model.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import zxingcpp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader import ReaderOptions, read_datamatrix, read_image_bytes


GENERIC_RECIPES = {
    "hints": [
        {
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "channel": channel,
            "blur_sigma": 3,
            "threshold": None,
            "invert": False,
            "grid_size": None,
        }
        for channel in ("blue", "yellow_blue")
    ]
}


class ProtocolServer(ThreadingHTTPServer):
    daemon_threads = True
    requests: list[dict]


class RecipeHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress even the stdlib access log: request metadata is asserted below.
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append({
            "path": self.path,
            "headers": dict(self.headers.items()),
            "body": json.loads(raw),
        })
        if self.server.behavior == "timeout":
            self.server.release_response.wait(1)
        if self.server.behavior == "unauthorized":
            status, data = 401, b'{"error":"private-response-must-not-leak"}'
        elif self.server.behavior == "malformed":
            status, data = 200, b'{"choices": [null]}'
        elif self.server.behavior == "invalid-json":
            status, data = 200, b'not valid JSON'
        elif self.server.behavior == "guess":
            status, data = 200, json.dumps({
                "choices": [{"message": {"content": '{"text":"9999999999999999"}'}}],
            }).encode()
        else:
            status, data = 200, json.dumps({
                "model": "internal/glm-5.3-flash",
                "choices": [{
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(self.server.recipes)},
                }],
            }).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            # A timeout test deliberately disconnects before receiving a reply.
            pass


@contextmanager
def recipe_server(behavior="recipes", recipes=None):
    server = ProtocolServer(("127.0.0.1", 0), RecipeHandler)
    server.requests = []
    server.behavior = behavior
    server.recipes = GENERIC_RECIPES if recipes is None else recipes
    server.release_response = threading.Event()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.release_response.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def server_env(server):
    # clear=True in callers prevents the developer's actual credentials from
    # being consulted, and demonstrates an internal endpoint requiring no key.
    return {
        "GLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
        "GLM_MODEL": "internal/glm-5.3-flash",
    }


class GLMReaderProtocolTests(unittest.TestCase):
    def read_fixture(self, name, *, timeout=1):
        return read_datamatrix(
            read_image_bytes((ROOT / "test_cases" / name).read_bytes()),
            ReaderOptions(mode="balanced", max_seconds=8, use_libdmtx=False,
                          use_glm=True, glm_timeout_seconds=timeout),
        )

    def test_both_user_images_decode_from_generic_http_recipes_and_real_zxing(self):
        fixtures = (
            ("user_yellow_mark_blue_metal.png", "0540234986250871"),
            ("user_yellow_mark_dark_metal.png", "0523413046220941"),
        )
        with recipe_server() as server, patch.dict("os.environ", server_env(server), clear=True):
            for name, payload in fixtures:
                with self.subTest(image=name):
                    result = self.read_fixture(name)
                    self.assertTrue(result.ok, result)
                    self.assertEqual({payload}, {hit.text for hit in result.hits})
                    self.assertEqual("ok", result.assistance["status"])
                    self.assertTrue(result.assistance["used"])
                    self.assertTrue(result.best_variant.startswith("glm/hint-"))
                    self.assertTrue(all(hit.engine.startswith("zxing-cpp") for hit in result.hits))
            self.assertEqual(2, len(server.requests))
            for request, (name, _) in zip(server.requests, fixtures):
                self.assertEqual("/v1/chat/completions", request["path"])
                self.assertNotIn("Authorization", request["headers"])
                self.assertEqual("internal/glm-5.3-flash", request["body"]["model"])
                self.assertNotIn("reasoning_effort", request["body"])
                content = request["body"]["messages"][1]["content"]
                self.assertIn("normalized", content[0]["text"])
                data_url = content[1]["image_url"]["url"]
                self.assertTrue(data_url.startswith("data:image/png;base64,"))
                uploaded = Image.open(BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
                with Image.open(ROOT / "test_cases" / name) as original:
                    np.testing.assert_array_equal(np.asarray(original.convert("RGB")), np.asarray(uploaded))

    def test_api_failures_preserve_real_local_fallback_and_do_not_expose_secrets(self):
        for behavior in ("unauthorized", "malformed", "invalid-json", "timeout", "guess"):
            with self.subTest(response=behavior), recipe_server(behavior) as server:
                env = {**server_env(server), "GLM_HEADERS_JSON": '{"X-API-Key":"test-secret-header"}'}
                stdout, stderr = StringIO(), StringIO()
                with patch.dict("os.environ", env, clear=True), redirect_stdout(stdout), redirect_stderr(stderr):
                    started = time.monotonic()
                    result = self.read_fixture("user_yellow_mark_dark_metal.png", timeout=0.05)
                    elapsed = time.monotonic() - started
                self.assertTrue(result.ok, result)
                self.assertEqual({"0523413046220941"}, {hit.text for hit in result.hits})
                self.assertEqual("error", result.assistance["status"])
                self.assertFalse(result.assistance["used"])
                self.assertFalse(result.best_variant.startswith("glm/"))
                self.assertEqual(1, len(server.requests))
                self.assertLess(elapsed, 2, "HTTP failure should leave the fast local fallback available")
                visible = stdout.getvalue() + stderr.getvalue() + repr(result)
                self.assertNotIn("test-secret-header", visible)
                self.assertNotIn("private-response-must-not-leak", visible)
                self.assertNotIn("9999999999999999", visible)

    def test_already_readable_generated_barcode_skips_api(self):
        payload = "0035792468001122"
        barcode = zxingcpp.create_barcode(payload, zxingcpp.BarcodeFormat.DataMatrix)
        image = Image.fromarray(np.asarray(zxingcpp.write_barcode_to_image(barcode, scale=8)))
        with recipe_server() as server, patch.dict("os.environ", server_env(server), clear=True):
            result = read_datamatrix(image, ReaderOptions(use_glm=True))
            self.assertTrue(result.ok)
            self.assertEqual({payload}, {hit.text for hit in result.hits})
            self.assertEqual({"status": "not-needed", "used": False}, result.assistance)
            self.assertEqual([], server.requests)

    def test_environment_timeout_is_used_without_reader_override(self):
        with recipe_server("timeout") as server:
            env = {**server_env(server), "GLM_TIMEOUT_SECONDS": "0.05"}
            with patch.dict("os.environ", env, clear=True):
                started = time.monotonic()
                result = self.read_fixture("user_yellow_mark_dark_metal.png", timeout=None)
                elapsed = time.monotonic() - started
            self.assertTrue(result.ok)
            self.assertEqual({"0523413046220941"}, {hit.text for hit in result.hits})
            self.assertEqual("error", result.assistance["status"])
            self.assertIn("timed out", result.assistance["error"])
            self.assertFalse(result.assistance["used"])
            self.assertLess(elapsed, 0.75)
            self.assertEqual(1, len(server.requests))

    def test_tight_grid_bbox_is_padded_and_real_grid_recovery_validates_payload(self):
        payload = "0011223344556677"
        barcode = zxingcpp.create_barcode(payload, zxingcpp.BarcodeFormat.DataMatrix)
        symbol = np.asarray(zxingcpp.write_barcode_to_image(barcode, scale=5, add_quiet_zones=False))
        margin = 20
        pixels = np.pad(symbol, margin, constant_values=255)
        height, width = pixels.shape
        recipes = {"hints": [{
            "bbox": [margin / width, margin / height, (width - margin) / width, (height - margin) / height],
            "channel": "gray", "blur_sigma": 0, "threshold": None,
            "invert": False, "grid_size": symbol.shape[0] // 5,
        }]}
        with recipe_server(recipes=recipes) as server, patch.dict("os.environ", server_env(server), clear=True):
            # Only bypass ordinary detection to force the grid-hint branch;
            # recover_grid and its underlying ZXing ECC validation stay real.
            with patch("reader.decode_with_zxing", return_value=[]):
                result = read_datamatrix(Image.fromarray(pixels), ReaderOptions(use_glm=True, max_seconds=8))
            self.assertTrue(result.ok, result)
            self.assertEqual({payload}, {hit.text for hit in result.hits})
            self.assertEqual("ok", result.assistance["status"])
            self.assertTrue(result.assistance["used"])
            self.assertIn("center-grid", result.best_variant)
            self.assertTrue(all(hit.engine == "zxing-cpp/center-grid" for hit in result.hits))
            self.assertEqual(1, len(server.requests))


if __name__ == "__main__":
    unittest.main()
