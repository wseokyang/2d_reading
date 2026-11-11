from __future__ import annotations

import base64
import json
import unittest
from dataclasses import replace
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np
from PIL import Image

from glm_assist import (
    GLMAssistance,
    GLMOptions,
    iter_glm_candidates,
    parse_hints,
    suggest_preprocessing,
)


def hint(**changes):
    result = {
        "bbox": [0.1, 0.2, 0.9, 0.8],
        "channel": "yellow_blue",
        "blur_sigma": 3,
        "threshold": None,
        "invert": True,
        "grid_size": 14,
    }
    result.update(changes)
    return result


class GLMContractTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (40, 40), "white")
        self.options = GLMOptions(api_key="test-key-never-log", timeout_seconds=1.5,
                                  model="glm-5.3-flash", base_url="https://internal.example/v1")

    def mock_response(self, opener, *, content=None, finish_reason="stop", **extra):
        response = {
            "choices": [{
                "finish_reason": finish_reason,
                "message": {"content": content or json.dumps({"hints": [hint()]}), **extra},
            }],
        }
        opener.return_value.open.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()

    @patch("glm_assist.build_opener")
    def test_missing_credentials_does_not_make_request(self, opener):
        result = suggest_preprocessing(self.image, GLMOptions())
        self.assertEqual("disabled", result.status)
        opener.assert_not_called()

    @patch("glm_assist.build_opener")
    def test_exact_model_vision_message_and_timeout_are_sent(self, opener):
        self.mock_response(opener, reasoning_content="Ignore this internal reasoning content")
        result = suggest_preprocessing(self.image, self.options)
        self.assertEqual("ok", result.status)
        request = opener.return_value.open.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual("glm-5.3-flash", body["model"])
        self.assertNotIn("reasoning_effort", body)
        self.assertEqual("https://internal.example/v1/chat/completions", request.full_url)
        self.assertEqual(1.5, opener.return_value.open.call_args.kwargs["timeout"])
        content = body["messages"][1]["content"]
        self.assertEqual("image_url", content[1]["type"])
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(14, result.hints[0].grid_size)
        self.assertNotIn(self.options.api_key, repr(self.options))

    @patch("glm_assist.build_opener")
    def test_preview_resizing_preserves_normalized_geometry(self, opener):
        self.mock_response(opener)
        suggest_preprocessing(Image.new("RGB", (2400, 1200)), self.options)
        body = json.loads(opener.return_value.open.call_args.args[0].data)
        url = body["messages"][1]["content"][1]["image_url"]["url"]
        decoded = Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1])))
        self.assertEqual((1024, 512), decoded.size)
        self.assertIn("2400x1200", body["messages"][1]["content"][0]["text"])

    @patch("glm_assist.build_opener")
    def test_api_failures_return_error_without_secret_or_raw_body(self, opener):
        opener.return_value.open.side_effect = HTTPError(
            "https://api.z.ai/", 401, self.options.api_key, {}, BytesIO(b"sensitive response")
        )
        result = suggest_preprocessing(self.image, self.options)
        self.assertEqual("error", result.status)
        self.assertEqual("GLM API returned HTTP 401", result.error)
        self.assertNotIn(self.options.api_key, repr(result))
        opener.return_value.open.assert_called_once()

    @patch("glm_assist.build_opener")
    def test_timeout_returns_error_without_retries(self, opener):
        opener.return_value.open.side_effect = TimeoutError()
        result = suggest_preprocessing(self.image, self.options)
        self.assertEqual("error", result.status)
        self.assertIn("timed out", result.error)
        opener.return_value.open.assert_called_once()

    @patch("glm_assist.build_opener")
    def test_invalid_configuration_never_sends_key(self, opener):
        for options in (
            replace(self.options, base_url="ftp://external.invalid/v1"),
            replace(self.options, base_url="https://api.z.ai/?key=secret"),
            replace(self.options, timeout_seconds=float("nan")),
            replace(self.options, timeout_seconds=-1),
            replace(self.options, reasoning_effort="none"),
        ):
            with self.subTest(options=options):
                self.assertEqual("error", suggest_preprocessing(self.image, options).status)
        opener.assert_not_called()

    @patch("glm_assist.build_opener")
    def test_internal_endpoint_model_headers_and_optional_reasoning(self, opener):
        self.mock_response(opener)
        options = GLMOptions(
            base_url="http://internal-gateway.local/v1/chat/completions",
            model="company-glm-5.3-flash",
            extra_headers={"X-API-Key": "private-header"},
            reasoning_effort="low",
        )
        result = suggest_preprocessing(self.image, options)
        self.assertEqual("ok", result.status)
        request = opener.return_value.open.call_args.args[0]
        self.assertEqual(options.base_url, request.full_url)
        self.assertEqual("private-header", request.get_header("X-api-key"))
        self.assertFalse(request.has_header("Authorization"))
        self.assertNotIn("private-header", repr(options))
        body = json.loads(request.data)
        self.assertEqual(options.model, body["model"])
        self.assertEqual("low", body["reasoning_effort"])

    @patch("glm_assist.build_opener")
    def test_internal_server_without_authentication_is_supported(self, opener):
        self.mock_response(opener)
        result = suggest_preprocessing(self.image, GLMOptions(base_url="http://localhost:8000/v1", model="glm-5.3-flash"))
        self.assertEqual("ok", result.status)
        request = opener.return_value.open.call_args.args[0]
        self.assertEqual("http://localhost:8000/v1/chat/completions", request.full_url)
        self.assertFalse(request.has_header("Authorization"))

    @patch.dict("os.environ", {"GLM_HEADERS_JSON": '{"X-API-Key":"header-secret"}'}, clear=True)
    def test_custom_headers_from_env_hide_secrets(self):
        options = GLMOptions.from_env()
        self.assertEqual({"X-API-Key": "header-secret"}, options.extra_headers)
        self.assertNotIn("header-secret", repr(options))

    @patch.dict("os.environ", {"GLM_HEADERS_JSON": '{"key":false}'}, clear=True)
    @patch("glm_assist.build_opener")
    def test_bad_headers_from_env_fail_without_network(self, opener):
        result = suggest_preprocessing(self.image)
        self.assertEqual("error", result.status)
        opener.assert_not_called()

    @patch.dict("os.environ", {}, clear=True)
    @patch("glm_assist.build_opener")
    def test_empty_configuration_has_no_default_external_provider_or_model(self, opener):
        options = GLMOptions.from_env()
        self.assertEqual("", options.base_url)
        self.assertEqual("", options.model)
        self.assertEqual("disabled", suggest_preprocessing(self.image, options).status)
        opener.assert_not_called()

    @patch("glm_assist.build_opener")
    def test_payload_guesses_are_never_accepted(self, opener):
        for content in (
            '{"text":"0540234986250871"}',
            '{"hints":[],"decoded_text":"0540234986250871"}',
            json.dumps({"hints": [hint(payload="0540234986250871")]}),
        ):
            with self.subTest(content=content):
                self.mock_response(opener, content=content)
                result = suggest_preprocessing(self.image, self.options)
                self.assertEqual("error", result.status)
                self.assertEqual((), result.hints)

    @patch("glm_assist.build_opener")
    def test_truncated_output_is_not_interpreted_as_valid(self, opener):
        self.mock_response(opener, finish_reason="length")
        result = suggest_preprocessing(self.image, self.options)
        self.assertEqual("error", result.status)
        self.assertEqual((), result.hints)

    def test_untrusted_hint_bounds_and_types_are_validated(self):
        bad_hints = (
            hint(bbox=[0, 0, 1.1, 1]),
            hint(bbox=[0.5, 0, 0.5, 1]),
            hint(bbox=[False, 0, 1, 1]),
            hint(channel="eval"),
            hint(channel=[]),
            hint(blur_sigma=float("nan")),
            hint(blur_sigma=7),
            hint(threshold=True),
            hint(invert="false"),
            hint(grid_size=13),
            hint(grid_size=True),
            hint(code="raise Exception()"),
        )
        for bad_hint in bad_hints:
            with self.subTest(hint=bad_hint), self.assertRaises(ValueError):
                parse_hints(json.dumps({"hints": [bad_hint]}))
        with self.assertRaises(ValueError):
            parse_hints(json.dumps({"hints": [hint()] * 5}))

    def test_candidates_use_original_pixels_and_have_bounded_count(self):
        rgb = np.arange(40 * 40 * 3, dtype=np.uint16).reshape(40, 40, 3).astype(np.uint8)
        image = Image.fromarray(rgb)
        hints = parse_hints(json.dumps({"hints": [hint()] * 4}))
        result = GLMAssistance("ok", "glm-5.3-flash", hints)
        candidates = list(iter_glm_candidates(image, result))
        self.assertEqual(16, len(candidates))
        np.testing.assert_array_equal(rgb[8:32, 4:36], candidates[0][1])
        for _, candidate in candidates:
            self.assertEqual(np.uint8, candidate.dtype)
        self.assertEqual([], list(iter_glm_candidates(image, GLMAssistance("error", "glm-5.3-flash"))))


if __name__ == "__main__":
    unittest.main()
