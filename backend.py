"""Data Matrix integration for callers that use OpenCV image arrays.

The result keeps the dictionary fields used by ``org_code.py``. This is a
Data Matrix entry point, not a replacement for that module's generic barcode
API or its selectable preprocessor/decoder implementations.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from PIL import Image

from reader import ReaderOptions, read_datamatrix


def decode_datamatrix_from_array(
    image_array: np.ndarray | None,
    *,
    use_glm: bool = False,
    max_seconds: float = 12.0,
    mode: str = "aggressive",
) -> dict[str, Any]:
    """Read Data Matrix codes from uint8 BGR or grayscale pixels.

    ``cv2.imread`` produces the expected BGR ordering. Convert RGB arrays from
    PIL explicitly before calling this function. Grayscale arrays can have
    shape ``(height, width)`` or ``(height, width, 1)``. None and empty arrays
    return a failed result; unsupported nonempty inputs raise ValueError.

    GLM is opt-in and uses the endpoint/model/credentials configured through
    the GLM environment variables. Returned code text always comes from the
    local decoder. ``max_seconds`` controls the local search budget; GLM has
    the separate timeout configured by ReaderOptions.

    ``debug_variants`` lists the reader's selected/diagnostic variants, not
    every attempted transformation. ``decoder_engine`` reports engines of
    successful hits (or "auto" when none succeeded). ``preprocessing_engine``
    identifies the configured pipeline. Optional ``assistance`` and ``error``
    fields carry reader diagnostics when present.
    """
    if mode not in {"fast", "balanced", "aggressive"}:
        raise ValueError("mode must be 'fast', 'balanced', or 'aggressive'")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be a finite positive number")

    result: dict[str, Any] = {
        "barcodes": [],
        "status": "failed",
        "debug_variants": [],
        "preprocessing_engine": "adaptive+glm" if use_glm else "adaptive",
        "decoder_engine": "auto",
    }
    if image_array is None:
        return result
    if not isinstance(image_array, np.ndarray):
        raise ValueError("image_array must be a numpy array")
    if image_array.size == 0:
        return result
    if image_array.dtype != np.uint8:
        raise ValueError("image_array must contain uint8 pixels")

    if image_array.ndim == 2:
        rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.ndim == 3 and image_array.shape[2] == 1:
        rgb = cv2.cvtColor(image_array[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("image_array must have shape (H, W), (H, W, 1), or (H, W, 3)")

    decoded = read_datamatrix(
        Image.fromarray(rgb),
        ReaderOptions(mode=mode, max_seconds=max_seconds, use_glm=use_glm),
    )
    result["barcodes"] = [
        {
            "text": hit.text,
            "format": hit.format,
            "preprocessing_method": hit.variant,
            "engine": hit.engine,
        }
        for hit in decoded.hits
    ]
    result["status"] = "success" if decoded.ok and decoded.hits else "failed"
    variants = [hit.variant for hit in decoded.hits]
    if decoded.best_variant:
        variants.append(decoded.best_variant)
    result["debug_variants"] = list(dict.fromkeys(variants))
    if decoded.hits:
        result["decoder_engine"] = ", ".join(dict.fromkeys(hit.engine for hit in decoded.hits))
    if decoded.error is not None:
        result["error"] = decoded.error
    if decoded.assistance is not None:
        result["assistance"] = decoded.assistance
    return result
