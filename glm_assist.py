"""Optional GLM image analysis; decoded payloads always come from local decoders.

The API sees the image and suggests bounded preprocessing operations. Model text
is never treated as a barcode value and no model-generated code is executed.
"""

from __future__ import annotations

import base64
import json
import math
import os
import socket
import time
from dataclasses import dataclass, field
from http.client import HTTPException
from io import BytesIO
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import cv2
import numpy as np
from PIL import Image


DEFAULT_MODEL = ""
DEFAULT_BASE_URL = ""
MAX_HINTS = 4
MAX_RESPONSE_BYTES = 262_144
CHANNELS = {"gray", "red", "green", "blue", "yellow_blue", "saturation", "lab_a", "lab_b"}
SQUARE_GRID_SIZES = {10, 12, 14, 16, 18, 20, 22, 24, 26, 32, 36, 40, 44, 48, 52, 64, 72, 80, 88, 96, 104, 120, 132, 144}


@dataclass(frozen=True)
class GLMOptions:
    api_key: str = field(default="", repr=False)
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = 20.0
    reasoning_effort: str | None = None
    max_tokens: int = 4096
    extra_headers: dict[str, str] = field(default_factory=dict, repr=False)
    configuration_error: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> GLMOptions:
        """Read only the documented environment variables; never load key files."""
        error = None
        try:
            timeout = float(os.environ.get("GLM_TIMEOUT_SECONDS", "20"))
        except ValueError:
            timeout = 20.0
            error = "GLM_TIMEOUT_SECONDS must be a number"
        try:
            headers = json.loads(os.environ.get("GLM_HEADERS_JSON", "{}") or "{}")
            if not isinstance(headers, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in headers.items()
            ):
                raise ValueError("Invalid headers")
        except (TypeError, ValueError):
            headers = {}
            error = "GLM_HEADERS_JSON must be a JSON object with string header names and values"
        return cls(
            api_key=os.environ.get("GLM_API_KEY", "") or os.environ.get("ZAI_API_KEY", ""),
            model=os.environ.get("GLM_MODEL", DEFAULT_MODEL),
            base_url=os.environ.get("GLM_BASE_URL", DEFAULT_BASE_URL),
            timeout_seconds=timeout,
            reasoning_effort=os.environ.get("GLM_REASONING_EFFORT") or None,
            extra_headers=headers,
            configuration_error=error,
        )


@dataclass(frozen=True)
class GLMHint:
    # Normalized coordinates against the uploaded image: left, top, right, bottom.
    bbox: tuple[float, float, float, float]
    channel: str
    blur_sigma: float = 0.0
    threshold: int | None = None
    invert: bool = False
    grid_size: int | None = None


@dataclass(frozen=True)
class GLMAssistance:
    status: str
    model: str
    hints: tuple[GLMHint, ...] = ()
    elapsed_seconds: float = 0.0
    error: str | None = None


SYSTEM_PROMPT = """You assist a local Data Matrix ECC200 decoder by inspecting its image.
Return only a JSON object with a \"hints\" array containing at most four plausible
preprocessing recipes, ordered by usefulness. Do not guess, transcribe, reconstruct,
or return the encoded payload. Do not return code, shell commands, or grid bits.

Each hint must use exactly these fields:
{\"bbox\": [0.0, 0.0, 1.0, 1.0], \"channel\": \"blue\", \"blur_sigma\": 2.0,
 \"threshold\": null, \"invert\": false, \"grid_size\": null}

bbox is [left, top, right, bottom] in normalized 0..1 coordinates. It bounds the
entire visible symbol, including all finder/timing edges, in the supplied image.
Use the full image [0,0,1,1] if already tightly cropped or uncertain.
channel must be one of gray, red, green, blue, yellow_blue, saturation, lab_a,
lab_b. yellow_blue means (R+G)/2-B. Select channels that separate the marking from
the surface. blur_sigma is 0..6 pixels at the image resolution stated by the user.
threshold is null for automatic thresholding, otherwise an integer 0..255 after
normalization. invert is a boolean to make the modules dark on a light background.
grid_size is null if uncertain, otherwise the number of modules per edge for a
square Data Matrix (including borders). For rectangular symbols leave it null.
Do not invent visual evidence. Return {\"hints\": []} if no useful recipe is visible.
"""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward image data or a bearer key to a redirected endpoint.
        return None


def _endpoint(base_url: str) -> str:
    parts = urlsplit(base_url)
    if parts.scheme not in {"https", "http"} or not parts.hostname:
        raise ValueError("GLM_BASE_URL must be an HTTP or HTTPS endpoint")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("GLM_BASE_URL cannot contain credentials, query parameters, or a fragment")
    base_url = base_url.rstrip("/")
    return base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"


def suggest_preprocessing(image: Image.Image, options: GLMOptions | None = None) -> GLMAssistance:
    """Make one bounded API request; errors preserve the caller's local fallback.

    No retries are made, and the model is never silently substituted. The request
    timeout is an HTTP socket timeout, separate from the local decoding budget.
    """
    options = options or GLMOptions.from_env()
    started = time.monotonic()

    def outcome(status: str, error: str | None = None, hints: tuple[GLMHint, ...] = ()) -> GLMAssistance:
        return GLMAssistance(status, options.model, hints, time.monotonic() - started, error)

    if options.configuration_error:
        return outcome("error", options.configuration_error)
    if not options.base_url.strip() or not options.model.strip():
        return outcome("disabled", "GLM_BASE_URL and GLM_MODEL must be configured")
    try:
        endpoint = _endpoint(options.base_url)
        if not options.model.strip():
            raise ValueError("GLM_MODEL cannot be empty")
        if not math.isfinite(options.timeout_seconds) or not 0 < options.timeout_seconds <= 120:
            raise ValueError("GLM timeout must be greater than 0 and at most 120 seconds")
        if options.reasoning_effort is not None and options.reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("GLM reasoning_effort must be low, high, or max")
        if not 256 <= options.max_tokens <= 16384:
            raise ValueError("GLM max_tokens must be between 256 and 16384")
        if not isinstance(options.extra_headers, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            and key.strip() and not any(char in key + value for char in "\r\n")
            for key, value in options.extra_headers.items()
        ):
            raise ValueError("GLM custom headers must contain valid string names and values")
    except (TypeError, ValueError) as exc:
        return outcome("error", str(exc))

    uploaded = image.convert("RGB")
    uploaded.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    uploaded.save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    body = {
        "model": options.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": (
                    f"Suggest preprocessing for this Data Matrix image. The original resolution is "
                    f"{image.width}x{image.height} pixels; the uploaded preview is "
                    f"{uploaded.width}x{uploaded.height}. Specify blur_sigma in original-image pixels. "
                    "All coordinates must be normalized. Output JSON only."
                )},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": options.max_tokens,
        "stream": False,
    }
    if options.reasoning_effort is not None:
        body["reasoning_effort"] = options.reasoning_effort
    headers = {"Content-Type": "application/json"}
    if options.api_key:
        headers["Authorization"] = f"Bearer {options.api_key}"
    headers.update(options.extra_headers)
    request = Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with build_opener(_NoRedirect()).open(request, timeout=options.timeout_seconds) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            return outcome("error", "GLM response exceeded the size limit")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
            raise ValueError("Invalid completion object")
        choice = payload["choices"][0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ValueError("Invalid completion choice")
        if choice.get("finish_reason") == "length":
            return outcome("error", "GLM response was truncated; increase the output-token budget")
        content = choice["message"]["content"]
        hints = parse_hints(content)
    except HTTPError as exc:
        return outcome("error", f"GLM API returned HTTP {exc.code}")
    except (TimeoutError, socket.timeout):
        return outcome("error", "GLM API request timed out")
    except URLError:
        return outcome("error", "GLM API connection failed")
    except (KeyError, IndexError, TypeError, ValueError, UnicodeError):
        return outcome("error", "GLM returned an invalid preprocessing response")
    except (OSError, HTTPException):
        return outcome("error", "GLM API connection failed")
    return outcome("ok", hints=hints)


def _number(value: object, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a number")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError("Number outside the allowed range")
    return number


def parse_hints(content: str) -> tuple[GLMHint, ...]:
    """Validate untrusted model output without interpreting any payload text."""
    if not isinstance(content, str) or len(content) > 32_768:
        raise ValueError("Invalid hint response")
    text = content.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    elif text.startswith("```\n") and text.endswith("```"):
        text = text[4:-3].strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or set(parsed) != {"hints"}:
        raise ValueError("Expected only a hints object")
    raw_hints = parsed["hints"]
    if not isinstance(raw_hints, list) or len(raw_hints) > MAX_HINTS:
        raise ValueError("Invalid hints array")
    hints = []
    allowed = {"bbox", "channel", "blur_sigma", "threshold", "invert", "grid_size"}
    for raw in raw_hints:
        if not isinstance(raw, dict) or set(raw) - allowed:
            raise ValueError("Unknown hint fields")
        box = raw.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Invalid bounding box")
        bbox = tuple(_number(value, 0, 1) for value in box)
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError("Empty bounding box")
        channel = raw.get("channel")
        if not isinstance(channel, str) or channel not in CHANNELS:
            raise ValueError("Unsupported channel")
        sigma = _number(raw.get("blur_sigma", 0), 0, 6)
        threshold = raw.get("threshold")
        if threshold is not None and (type(threshold) is not int or not 0 <= threshold <= 255):
            raise ValueError("Invalid threshold")
        invert = raw.get("invert", False)
        if not isinstance(invert, bool):
            raise ValueError("Invalid inversion flag")
        size = raw.get("grid_size")
        if size is not None and (type(size) is not int or size not in SQUARE_GRID_SIZES):
            raise ValueError("Invalid Data Matrix grid size")
        hints.append(GLMHint(bbox, channel, sigma, threshold, invert, size))
    return tuple(hints)


def iter_glm_candidates(image: Image.Image, assistance: GLMAssistance) -> Iterable[tuple[str, np.ndarray]]:
    """Render up to four image candidates per hint from real input pixels only.

    These remain *candidates*. The caller must require a successful local Data
    Matrix decode before returning any code value to its user.
    """
    if assistance.status != "ok":
        return
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    for index, hint in enumerate(assistance.hints[:MAX_HINTS]):
        left = max(0, math.floor(hint.bbox[0] * width))
        top = max(0, math.floor(hint.bbox[1] * height))
        right = min(width, math.ceil(hint.bbox[2] * width))
        bottom = min(height, math.ceil(hint.bbox[3] * height))
        crop = rgb[top:bottom, left:right]
        if min(crop.shape[:2]) < 8:
            continue
        name = f"glm/hint-{index + 1}/{hint.channel}/sigma-{hint.blur_sigma:g}"
        yield f"{name}/rgb", crop
        if hint.channel in {"red", "green", "blue"}:
            source = crop[:, :, {"red": 0, "green": 1, "blue": 2}[hint.channel]].astype(np.float32)
        elif hint.channel == "yellow_blue":
            values = crop.astype(np.float32)
            source = (values[:, :, 0] + values[:, :, 1]) / 2 - values[:, :, 2]
        elif hint.channel == "saturation":
            source = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
        elif hint.channel in {"lab_a", "lab_b"}:
            source = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)[:, :, {"lab_a": 1, "lab_b": 2}[hint.channel]].astype(np.float32)
        else:
            source = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if hint.blur_sigma > 0:
            source = cv2.GaussianBlur(source, (0, 0), hint.blur_sigma)
        source = cv2.normalize(source, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if hint.invert:
            source = 255 - source
        yield f"{name}/normalized", source
        if hint.threshold is None:
            _, binary = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(source, hint.threshold, 255, cv2.THRESH_BINARY)
        pad = max(8, round(min(source.shape) * 0.12))
        yield f"{name}/threshold", cv2.copyMakeBorder(binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        yield f"{name}/threshold-inverse", cv2.copyMakeBorder(255 - binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
