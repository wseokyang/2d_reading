"""Data Matrix adapter for the original OpenCV-array import path.

The default read call now uses the maintained adaptive reader and retains the
original result dictionary fields. Generic barcode support, the Classic/VE2
method catalog, and forcing an individual decoder are not part of this adapter.
Unsupported legacy selectors fail explicitly; their original implementation is
preserved unchanged in reference/org_code.py. The UI is maintained in app.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend import decode_datamatrix_from_array

__all__ = ["decode_datamatrix_from_array", "decode_image_from_array", "decode_multiple_barcodes"]


def decode_image_from_array(
    image_array: np.ndarray | None,
    selected_methods: set[str] | None = None,
    preprocessor_type: str = "adaptive",
    decoder_type: str = "auto",
    *,
    use_glm: bool = False,
    max_seconds: float = 12.0,
    mode: str = "aggressive",
) -> dict[str, Any]:
    """Read Data Matrix from BGR/grayscale pixels using the shared pipeline.

    The four original positional argument names remain available, but only
    ``selected_methods=None`` (or empty), ``preprocessor_type='adaptive'`` and
    ``decoder_type='auto'`` are supported. Use ``mode`` to select search strength
    and ``use_glm`` to enable API-assisted preprocessing. Code text is always
    confirmed by a local barcode decoder.
    """
    if selected_methods:
        raise ValueError("selected_methods is not supported by the adaptive reader; select mode instead")
    if preprocessor_type != "adaptive":
        raise ValueError("preprocessor_type must be 'adaptive'; Classic/VE2 selectors are preserved only in reference/org_code.py")
    if decoder_type != "auto":
        raise ValueError("decoder_type must be 'auto'; forcing one legacy decoder is not supported")
    return decode_datamatrix_from_array(
        image_array, use_glm=use_glm, max_seconds=max_seconds, mode=mode
    )


def decode_multiple_barcodes(image: np.ndarray | None) -> list[dict[str, str]]:
    """Retain the old list-returning entry point for Data Matrix codes only."""
    return decode_image_from_array(image)["barcodes"]
