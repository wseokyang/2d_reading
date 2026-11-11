#!/usr/bin/env python3
"""
Barcode Recognition Engine - Unified Backend Logic
Image preprocessing (Classic/VE2) and barcode decoding (ZXingCPP/PyLibDMTX).

Architecture:
  Preprocessor Layer
    ├── ClassicPreprocessor (current methods) ← Default
    └── VE2Preprocessor (VE2 balanced mode)
  
  Decoder Layer
    ├── ZXingCPPDecoder ← Default
    └── PyLibDMTXDecoder
  
  Orchestrator
    └── decode_image() ← Combines preprocessor + decoder
"""

from __future__ import annotations

import csv
import io
import base64
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Iterable

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    from PIL import Image as PILImage, ImageEnhance, ImageOps
    _HAS_PIL = True
except ImportError:
    PILImage = None
    _HAS_PIL = False

try:
    from pylibdmtx.pylibdmtx import decode as dmtx_decode
    from pylibdmtx.wrapper import DmtxSymbolSize
    _HAS_PYLIBDMTX = True
except ImportError:
    dmtx_decode = None
    DmtxSymbolSize = None
    _HAS_PYLIBDMTX = False

try:
    from zxingcpp import read_barcodes, Binarizer, BarcodeFormat
    _HAS_ZXINGCPP = True
except ImportError:
    _HAS_ZXINGCPP = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PREPROCESS_CATEGORIES: Dict[str, List[Any]] = {
    "기본 (Basic)": [
        "original",
        "inverted",
        "histeq",
        "unsharp_mask",
    ],
    "확대 (Resize)": [
        ("resize_1.5x", {"scale": 1.5}),
        ("resize_2.0x", {"scale": 2.0}),
        ("resize_2.5x", {"scale": 2.5}),
        ("resize_3.0x", {"scale": 3.0}),
        ("resize_4.0x", {"scale": 4.0}),
        ("resize_5.0x", {"scale": 5.0}),
        ("resize_6.0x", {"scale": 6.0}),
        ("resize_8.0x", {"scale": 8.0}),
        ("resize_10.0x", {"scale": 10.0}),
    ],
    "CLAHE (대비)": [
        ("clahe_c1.5_t16x16", {"clipLimit": 1.5, "tileGridSize": [16, 16]}),
        ("clahe_c2.0_t8x8", {"clipLimit": 2.0, "tileGridSize": [8, 8]}),
        ("clahe_c2.5_t8x8", {"clipLimit": 2.5, "tileGridSize": [8, 8]}),
        ("clahe_c3.0_t4x4", {"clipLimit": 3.0, "tileGridSize": [4, 4]}),
        ("clahe_c4.0_t6x6", {"clipLimit": 4.0, "tileGridSize": [6, 6]}),
    ],
    "블러/노이즈 (Blur)": [
        ("medianBlur_3", {"kernel": 3}),
        ("medianBlur_5", {"kernel": 5}),
    ],
    "오투 이진화 (Otsu)": [
        ("otsu_noblur", {"threshold": "binary_otsu"}),
        ("otsu_noblur_inv", {"threshold": "binary_inv_otsu"}),
        ("otsu_gauss3", {"blur_kernel": (3, 3)}),
        ("otsu_gauss5", {"blur_kernel": (5, 5)}),
        ("otsu_gauss7", {"blur_kernel": (7, 7)}),
        ("otsu_gauss9", {"blur_kernel": (9, 9)}),
        ("otsu_inv_gauss3", {"blur_kernel": (3, 3), "inv": True}),
        ("otsu_inv_gauss5", {"blur_kernel": (5, 5), "inv": True}),
        ("otsu_inv_gauss7", {"blur_kernel": (7, 7), "inv": True}),
        ("otsu_inv_gauss9", {"blur_kernel": (9, 9), "inv": True}),
    ],
    "적응 이진화 (Adaptive)": [
        "adaptive_gauss_binary_11",
        "adaptive_gauss_binary_15",
        "adaptive_gauss_binary_21",
        "adaptive_gauss_binary_31",
        "adaptive_gauss_binary_41",
        "adaptive_gauss_binary_51",
        "adaptive_gauss_inv_11",
        "adaptive_gauss_inv_15",
        "adaptive_gauss_inv_21",
        "adaptive_gauss_inv_31",
        "adaptive_gauss_inv_41",
        "adaptive_gauss_inv_51",
    ],
    "임계값 (Threshold)": [
        "thresh_binary_80",
        "thresh_binary_100",
        "thresh_binary_127",
        "thresh_binary_150",
        "thresh_binary_180",
        "thresh_binary_200",
        "thresh_inv_80",
        "thresh_inv_100",
        "thresh_inv_127",
        "thresh_inv_150",
        "thresh_inv_180",
        "thresh_inv_200",
    ],
    "회전 (Rotation)": [
        "rotate_90",
        "rotate_180",
        "rotate_270",
    ],
    "감마 (Gamma)": [
        ("gamma_0.3", {"gamma": 0.3}),
        ("gamma_0.5", {"gamma": 0.5}),
        ("gamma_0.7", {"gamma": 0.7}),
        ("gamma_1.5", {"gamma": 1.5}),
        ("gamma_2.0", {"gamma": 2.0}),
        ("gamma_2.5", {"gamma": 2.5}),
        ("gamma_3.0", {"gamma": 3.0}),
    ],
    "감마+확대 (Gamma+Resize)": [
        ("gamma_0.5_resize_2.0x", {"gamma": 0.5, "scale": 2.0}),
        ("gamma_0.5_resize_3.0x", {"gamma": 0.5, "scale": 3.0}),
        ("gamma_0.5_resize_4.0x", {"gamma": 0.5, "scale": 4.0}),
        ("gamma_0.7_resize_2.0x", {"gamma": 0.7, "scale": 2.0}),
        ("gamma_0.7_resize_3.0x", {"gamma": 0.7, "scale": 3.0}),
        ("gamma_0.7_resize_4.0x", {"gamma": 0.7, "scale": 4.0}),
        ("gamma_1.5_resize_2.0x", {"gamma": 1.5, "scale": 2.0}),
        ("gamma_1.5_resize_3.0x", {"gamma": 1.5, "scale": 3.0}),
        ("gamma_1.5_resize_4.0x", {"gamma": 1.5, "scale": 4.0}),
        ("gamma_2.0_resize_2.0x", {"gamma": 2.0, "scale": 2.0}),
        ("gamma_2.0_resize_3.0x", {"gamma": 2.0, "scale": 3.0}),
        ("gamma_2.0_resize_4.0x", {"gamma": 2.0, "scale": 4.0}),
    ],
    "밝기 (Brightness)": [
        ("brightness_x1.2", {"alpha": 1.2}),
        ("brightness_x1.5", {"alpha": 1.5}),
        ("brightness_x2.0", {"alpha": 2.0}),
        ("brightness_x2.5", {"alpha": 2.5}),
        ("brightness_x3.0", {"alpha": 3.0}),
    ],
    "밝기+확대 (Brightness+Resize)": [
        ("brightness_x1.5_resize_2.0x", {"alpha": 1.5, "scale": 2.0}),
        ("brightness_x1.5_resize_3.0x", {"alpha": 1.5, "scale": 3.0}),
        ("brightness_x2.0_resize_2.0x", {"alpha": 2.0, "scale": 2.0}),
        ("brightness_x2.0_resize_3.0x", {"alpha": 2.0, "scale": 3.0}),
        ("brightness_x3.0_resize_3.0x", {"alpha": 3.0, "scale": 3.0}),
    ],
    "복합 (Combined)": [
        ("resize_2.0x_clahe", {"scale": 2.0, "clahe": 2.0}),
        ("resize_3.0x_clahe", {"scale": 3.0, "clahe": 2.0}),
        ("resize_4.0x_clahe", {"scale": 4.0, "clahe": 2.0}),
        ("gamma_0.5_clahe", {"gamma": 0.5}),
        ("gamma_0.7_clahe", {"gamma": 0.7}),
        ("gamma_1.5_clahe", {"gamma": 1.5}),
        ("gamma_2.0_clahe", {"gamma": 2.0}),
        ("brightness_x1.5_clahe", {"alpha": 1.5}),
        ("brightness_x2.0_clahe", {"alpha": 2.0}),
        ("brightness_x3.0_clahe", {"alpha": 3.0}),
        ("histeq_resize_2.0x", {"scale": 2.0}),
        ("histeq_resize_3.0x", {"scale": 3.0}),
    ],
    "ROI/그리드 (Advanced)": [
        "roi_crop",
        "grid_search",
    ],
}

# Recommended methods for Classic preprocessor
RECOMMENDED_METHODS: List[str] = [
    "original",
    "resize_1.5x",
    "resize_2.0x",
    "resize_2.5x",
    "resize_6.0x",
    "gamma_1.5_resize_2.0x",
    "clahe_c3.0_t4x4",
    "otsu_gauss5",
    "inverted",
    "medianBlur_3",
    "roi_crop",
]
RECOMMENDED_SET: set = set(RECOMMENDED_METHODS)

# Preprocessor options
PREPROCESSOR_CLASSIC = "classic"
PREPROCESSOR_VE2_BALANCED = "ve2_balanced"
PREPROCESSOR_VE2_AGGRESSIVE = "ve2_aggressive"
PREPROCESSOR_ALL = "all"

PREPROCESSOR_OPTIONS = [PREPROCESSOR_CLASSIC, PREPROCESSOR_VE2_BALANCED, PREPROCESSOR_VE2_AGGRESSIVE, PREPROCESSOR_ALL]

# Decoder options
DECODER_ZXINGCPP = "zxingcpp"
DECODER_PYLIBDMTX = "pylibdmtx"

DECODER_OPTIONS = [DECODER_ZXINGCPP, DECODER_PYLIBDMTX]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class BarcodeHit:
    """Represents a single barcode detection result."""
    text: str
    format: str
    preprocessing_method: str
    engine: str = ""


@dataclass
class DecodeResult:
    """Result of barcode decoding."""
    barcodes: List[Dict[str, str]] = field(default_factory=list)
    status: str = "failed"
    debug_variants: List[str] = field(default_factory=list)
    preprocessing_engine: str = ""
    decoder_engine: str = ""


# ---------------------------------------------------------------------------
# Helper: Get all method names with recommended flags
# ---------------------------------------------------------------------------

def get_all_methods_with_recommended() -> List[Dict[str, Any]]:
    """Get flat list of all Classic methods with recommended flag."""
    methods = []
    for category_methods in PREPROCESS_CATEGORIES.values():
        for method in category_methods:
            if isinstance(method, str):
                methods.append({"id": method, "recommended": method in RECOMMENDED_SET})
            else:
                methods.append({"id": method[0], "recommended": method[0] in RECOMMENDED_SET})
    return methods


def get_all_method_ids() -> List[str]:
    """Get flat list of all Classic method IDs."""
    return [m["id"] for m in get_all_methods_with_recommended()]


def get_recommended_method_ids() -> List[str]:
    """Get list of recommended Classic method IDs."""
    return list(RECOMMENDED_SET)


# ===========================================================================
# PREPROCESSOR LAYER
# ===========================================================================

class BasePreprocessor:
    """Base class for image preprocessors."""
    
    def generate_variants(
        self,
        image: np.ndarray,
        selected_methods: Optional[set] = None
    ) -> List[Tuple[str, np.ndarray]]:
        """Generate preprocessing variants. Returns list of (name, image)."""
        raise NotImplementedError


class ClassicPreprocessor(BasePreprocessor):
    """Classic preprocessor - uses the original method selection approach.
    
    Generates variants from predefined methods (resize, CLAHE, Otsu, etc.)
    User can select specific methods or use all.
    """
    
    def generate_variants(
        self,
        image: np.ndarray,
        selected_methods: Optional[set] = None
    ) -> List[Tuple[str, np.ndarray]]:
        variants: List[Tuple[str, np.ndarray]] = []
        h, w = image.shape[:2]

        def _add(name: str, img: np.ndarray):
            if selected_methods is None or name in selected_methods:
                variants.append((name, img))

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Original
        _add("original", gray)

        # Resize scales
        for scale in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w > 4000 or new_h > 4000:
                continue
            resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            _add(f"resize_{scale}x", resized)

        # CLAHE
        for clip, tx, ty in [(2.0, 8, 8), (3.0, 4, 4), (1.5, 16, 16), (2.5, 8, 8), (4.0, 6, 6)]:
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tx, ty))
            _add(f"clahe_c{clip}_t{tx}x{ty}", clahe.apply(gray))

        # Denoising
        _add("medianBlur_3", cv2.medianBlur(gray, 3))
        _add("medianBlur_5", cv2.medianBlur(gray, 5))

        # Gaussian + Otsu
        for ksize in [(3, 3), (5, 5), (7, 7), (9, 9)]:
            blurred = cv2.GaussianBlur(gray, ksize, 0)
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _add(f"otsu_gauss{ksize[0]}", otsu)
            _, otsu_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            _add(f"otsu_inv_gauss{ksize[0]}", otsu_inv)

        # Otsu on original
        _, otsu_noblur = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _add("otsu_noblur", otsu_noblur)
        _, otsu_noblur_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _add("otsu_noblur_inv", otsu_noblur_inv)

        # Adaptive threshold
        for bs in [11, 15, 21, 31, 41, 51]:
            if bs > min(w, h) // 2:
                continue
            adj = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, bs, 10)
            _add(f"adaptive_gauss_binary_{bs}", adj)
            adj_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, bs, 10)
            _add(f"adaptive_gauss_inv_{bs}", adj_inv)

        # Simple binary threshold
        for thresh in [80, 100, 127, 150, 180, 200]:
            _, simple_bin = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            _add(f"thresh_binary_{thresh}", simple_bin)
            _, simple_bin_inv = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
            _add(f"thresh_inv_{thresh}", simple_bin_inv)

        # Inverted
        _add("inverted", cv2.bitwise_not(gray))

        # Sharpening
        blur_sharp = cv2.GaussianBlur(gray, (0, 0), 2.0)
        sharpened = cv2.addWeighted(gray, 1.5, blur_sharp, -0.5, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        _add("unsharp_mask", sharpened)

        # Rotations
        for angle in [90, 180, 270]:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            _add(f"rotate_{angle}", cv2.warpAffine(gray, M, (w, h)))

        # Combined: resize + CLAHE
        for scale in [2.0, 3.0, 4.0]:
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w > 4000 or new_h > 4000:
                continue
            resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            _add(f"resize_{scale}x_clahe", c.apply(resized))

        # Histogram equalization
        eq = cv2.equalizeHist(gray)
        _add("histeq", eq)
        for scale in [2.0, 3.0]:
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w > 4000 or new_h > 4000:
                continue
            _add(f"histeq_resize_{scale}x", cv2.resize(eq, (new_w, new_h), interpolation=cv2.INTER_CUBIC))

        # Gamma correction
        for gamma in [0.3, 0.5, 0.7, 1.5, 2.0, 2.5, 3.0]:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
            _add(f"gamma_{gamma}", cv2.LUT(gray, table))

        # Gamma + resize
        for gamma in [0.5, 0.7, 1.5, 2.0]:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
            gamma_img = cv2.LUT(gray, table)
            for scale in [2.0, 3.0, 4.0]:
                new_w, new_h = int(w * scale), int(h * scale)
                if new_w > 4000 or new_h > 4000:
                    continue
                _add(f"gamma_{gamma}_resize_{scale}x",
                     cv2.resize(gamma_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC))

        # Brightness boost
        for alpha in [1.2, 1.5, 2.0, 2.5, 3.0]:
            _add(f"brightness_x{alpha}", cv2.convertScaleAbs(gray, alpha=alpha, beta=0))

        # Brightness + resize
        for alpha in [1.5, 2.0, 3.0]:
            bright_img = cv2.convertScaleAbs(gray, alpha=alpha, beta=0)
            for scale in [2.0, 3.0]:
                new_w, new_h = int(w * scale), int(h * scale)
                if new_w > 4000 or new_h > 4000:
                    continue
                _add(f"brightness_x{alpha}_resize_{scale}x",
                     cv2.resize(bright_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC))

        # Combined: gamma + CLAHE
        for gamma in [0.5, 0.7, 1.5, 2.0]:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
            gamma_img = cv2.LUT(gray, table)
            c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            _add(f"gamma_{gamma}_clahe", c.apply(gamma_img))

        # Combined: brightness + CLAHE
        for alpha in [1.5, 2.0, 3.0]:
            bright_img = cv2.convertScaleAbs(gray, alpha=alpha, beta=0)
            c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            _add(f"brightness_x{alpha}_clahe", c.apply(bright_img))

        return variants


class VE2Preprocessor(BasePreprocessor):
    """VE2 preprocessor - uses advanced channel-based techniques.
    
    Modes:
        balanced: Channel separation (RGB, HSV, LAB), gamma/contrast, binarization
                  ~50-100 variants
        aggressive: All balanced + grid reconstruction, various rotations (-14°~+14°)
                    ~500-650 variants (up to 12 seconds)
    """
    
    def __init__(self, mode: str = "balanced"):
        if mode not in ("balanced", "aggressive"):
            raise ValueError(f"VE2 mode must be 'balanced' or 'aggressive', got: {mode}")
        self.mode = mode
        if not _HAS_PIL:
            raise ImportError("PIL (Pillow) is required for VE2 preprocessor")
    
    def generate_variants(
        self,
        image: np.ndarray,
        selected_methods: Optional[set] = None
    ) -> List[Tuple[str, np.ndarray]]:
        variants: List[Tuple[str, np.ndarray]] = []
        
        # Convert to PIL and back
        rgb = np.array(PILImage.fromarray(image).convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        def _add(name: str, img: np.ndarray):
            variants.append((name, img))

        # Basic variants (always included)
        _add("ve2/original-rgb", rgb)
        _add("ve2/gray", gray)
        _add("ve2/gray-median5", cv2.medianBlur(gray, 5))
        _add("ve2/gray-normalized", normalize(gray))
        _add("ve2/gray-clahe", clahe.apply(gray))
        _add("ve2/gray-sharpened", sharpen(gray))

        # Channel-based variants (balanced mode)
        base_channels = (
            ("red", bgr[:, :, 2]),
            ("green", bgr[:, :, 1]),
            ("blue", bgr[:, :, 0]),
            ("hsv-value", hsv[:, :, 2]),
            ("hsv-saturation", hsv[:, :, 1]),
            ("lab-lightness", lab[:, :, 0]),
        )
        
        for name, channel in base_channels:
            _add(f"ve2/{name}", channel)
            _add(f"ve2/{name}-normalized", normalize(channel))
            _add(f"ve2/{name}-clahe", clahe.apply(channel))

        # Gamma and contrast adjustments
        for name, channel in (("gray", gray), ("lab-lightness", lab[:, :, 0]), ("green", bgr[:, :, 1])):
            for gamma in (0.45, 0.65, 0.85, 1.25, 1.6):
                _add(f"ve2/{name}-gamma-{gamma}", gamma_correct(channel, gamma))
            for alpha, beta in ((1.4, 0), (1.8, -30), (2.2, -70), (1.8, 35), (1.2, 60)):
                _add(f"ve2/{name}-contrast-{alpha}-{beta}", cv2.convertScaleAbs(channel, alpha=alpha, beta=beta))

        # Threshold variants
        threshold_sources = [
            ("gray", gray),
            ("gray-clahe", clahe.apply(gray)),
            ("gray-normalized", normalize(gray)),
        ]
        
        if self.mode == "aggressive":
            # Additional channels for aggressive mode
            threshold_sources.extend([
                ("red", bgr[:, :, 2]),
                ("green", bgr[:, :, 1]),
                ("lab-lightness", lab[:, :, 0]),
                ("hsv-value", hsv[:, :, 2]),
            ])
        
        for name, source in threshold_sources:
            for variant_name, variant_img in threshold_variants(name, source):
                _add(f"ve2/{variant_name}", variant_img)

        return variants


# ===========================================================================
# VE2 Helper Functions
# ===========================================================================

def normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to 0-255."""
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)


def sharpen(arr: np.ndarray) -> np.ndarray:
    """Sharpen array using unsharp mask."""
    blur = cv2.GaussianBlur(arr, (0, 0), 1.0)
    return cv2.addWeighted(arr, 1.7, blur, -0.7, 0)


def gamma_correct(arr: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction."""
    table = (np.power(np.arange(256) / 255.0, gamma) * 255).astype("uint8")
    return cv2.LUT(arr, table)


def threshold_variants(name: str, source: np.ndarray) -> Iterable[Tuple[str, np.ndarray]]:
    """Generate threshold variants for a source image."""
    _, otsu = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield f"{name}-otsu", otsu
    yield f"{name}-otsu-inverted", 255 - otsu

    for block in (15, 25, 35, 51):
        if min(source.shape[:2]) <= block:
            continue
        adaptive = cv2.adaptiveThreshold(
            source,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            3,
        )
        yield f"{name}-adaptive-{block}", adaptive
        yield f"{name}-adaptive-{block}-inverted", 255 - adaptive


# ===========================================================================
# ROI Helper (Classic)
# ===========================================================================

def _crop_roi_candidates(image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Detect ROI candidates from image."""
    crops: List[Tuple[str, np.ndarray]] = []
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = min(400, w * h * 0.001)
    max_area = w * h * 0.8

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / float(ch)
        if aspect < 0.2 or aspect > 5.0:
            continue
        pad = min(20, cw // 5, ch // 5)
        crop = gray[max(0, y - pad):min(h, y + ch + pad),
                    max(0, x - pad):min(w, x + cw + pad)]
        crops.append((f"roi_contour_{area}", crop))

    return crops


# ===========================================================================
# DECODER LAYER
# ===========================================================================

class BaseDecoder:
    """Base class for barcode decoders."""
    
    def decode(self, image: np.ndarray, variant_name: str) -> List[BarcodeHit]:
        """Decode barcodes from image. Returns list of BarcodeHit."""
        raise NotImplementedError


class ZXingCPPDecoder(BaseDecoder):
    """ZXing-CPP decoder - supports multiple binarizers."""
    
    def __init__(self):
        if not _HAS_ZXINGCPP:
            raise ImportError("zxing-cpp is required for ZXingCPP decoder")
    
    def decode(self, image: np.ndarray, variant_name: str) -> List[BarcodeHit]:
        if image is None or image.size == 0:
            return []
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if gray.shape[0] < 10 or gray.shape[1] < 10:
            return []

        hits: List[BarcodeHit] = []
        binarizers = [
            Binarizer.LocalAverage,
            Binarizer.GlobalHistogram,
            Binarizer.FixedThreshold,
        ]

        for binarizer in binarizers:
            try:
                results = read_barcodes(gray, binarizer=binarizer)
                for r in results:
                    if getattr(r, "error", None):
                        continue
                    text = r.text
                    if not text:
                        continue
                    fmt = str(r.format).split(".")[-1] if hasattr(r, 'format') else 'unknown'
                    hits.append(BarcodeHit(
                        text=text,
                        format=fmt,
                        preprocessing_method=variant_name,
                        engine=f"zxing-cpp/{binarizer.name}",
                    ))
            except Exception:
                continue
            
            if hits:
                return dedupe_hits(hits)
        
        return []


class PyLibDMTXDecoder(BaseDecoder):
    """PyLibDMTX decoder - for DataMatrix barcodes."""
    
    def __init__(self):
        if not _HAS_PYLIBDMTX or dmtx_decode is None:
            raise ImportError("pylibdmtx is required for PyLibDMTX decoder")
    
    def decode(self, image: np.ndarray, variant_name: str) -> List[BarcodeHit]:
        if image is None or image.size == 0:
            return []

        # Convert to PIL for pylibdmtx
        if not _HAS_PIL:
            return []
        
        if len(image.shape) == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb, mode="RGB")
        else:
            pil_img = PILImage.fromarray(image, mode="L")

        hits: List[BarcodeHit] = []
        timeout = 650
        
        try:
            decoded = dmtx_decode(pil_img, timeout=timeout, max_count=10)
            for item in decoded:
                text = item.data.decode("utf-8", errors="replace")
                if text and len(text.strip()) >= 3:
                    hits.append(BarcodeHit(
                        text=text,
                        format="DataMatrix",
                        preprocessing_method=variant_name,
                        engine="pylibdmtx",
                    ))
        except Exception:
            pass
        
        return dedupe_hits(hits)


# ===========================================================================
# Decoder Factory
# ===========================================================================

def create_decoder(decoder_type: str) -> BaseDecoder:
    """Create a decoder instance based on type."""
    if decoder_type == DECODER_ZXINGCPP:
        return ZXingCPPDecoder()
    elif decoder_type == DECODER_PYLIBDMTX:
        return PyLibDMTXDecoder()
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")


def get_available_decoders() -> List[str]:
    """Get list of available decoders."""
    available = []
    if _HAS_ZXINGCPP:
        available.append(DECODER_ZXINGCPP)
    if _HAS_PYLIBDMTX and dmtx_decode is not None:
        available.append(DECODER_PYLIBDMTX)
    return available


# ===========================================================================
# Helper Functions
# ===========================================================================

def dedupe_hits(hits: List[BarcodeHit]) -> List[BarcodeHit]:
    """Remove duplicate hits by text."""
    seen: set[str] = set()
    result: List[BarcodeHit] = []
    for hit in hits:
        if hit.text in seen:
            continue
        seen.add(hit.text)
        result.append(hit)
    return result


def is_plausible_text(text: str) -> bool:
    """Check if text is plausible barcode content."""
    if not text:
        return False
    if len(text.strip()) < 6:
        return False
    if any(ch.isalnum() for ch in text):
        return True
    printable = sum(ch.isprintable() and not ch.isspace() for ch in text)
    return printable >= 4


# ===========================================================================
# Reader.py Integration: Utility Functions
# ===========================================================================

def resize_for_scan(arr: np.ndarray, scale: int) -> np.ndarray:
    """Resize array for scanning, choosing interpolation based on content."""
    if scale == 1:
        return arr
    h, w = arr.shape[:2]
    unique_count = len(np.unique(arr)) if arr.ndim == 2 else 999
    interpolation = cv2.INTER_NEAREST if unique_count <= 8 else cv2.INTER_CUBIC
    return cv2.resize(arr, (w * scale, h * scale), interpolation=interpolation)


def add_quiet_zone(arr: np.ndarray, value: int = 255) -> np.ndarray:
    """Add quiet zone (border) around array."""
    h, w = arr.shape[:2]
    pad = max(8, int(max(h, w) * 0.18))
    return cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=value)


def rotate_image(arr: np.ndarray, degrees: float, border_value: int = 255) -> np.ndarray:
    """Rotate array by given degrees."""
    h, w = arr.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_w = int((h * sine) + (w * cosine))
    new_h = int((h * cosine) + (w * sine))
    matrix[0, 2] += new_w / 2 - w / 2
    matrix[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(
        arr, matrix, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def pil_resize(image, scale: int):
    """Resize PIL image by scale factor."""
    width, height = image.size
    return image.resize((width * scale, height * scale), Image.Resampling.BICUBIC)


def pil_pad(image, fraction: float, color: str):
    """Add padding to PIL image."""
    width, height = image.size
    pad = max(8, int(max(width, height) * fraction))
    if image.mode == "L":
        fill = 255 if color == "white" else 0
    else:
        fill = (255, 255, 255) if color == "white" else (0, 0, 0)
    return ImageOps.expand(image, border=pad, fill=fill)


def pil_rotate(image, degrees: float, color: str):
    """Rotate PIL image by degrees."""
    if image.mode == "L":
        fill = 255 if color == "white" else 0
    else:
        fill = (255, 255, 255) if color == "white" else (0, 0, 0)
    return image.rotate(degrees, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=fill)


def cv_resize_rgb(arr: np.ndarray, scale: int) -> np.ndarray:
    """Resize RGB array by scale factor."""
    height, width = arr.shape[:2]
    return cv2.resize(arr, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)


def cv_rotate_rgb(arr: np.ndarray, degrees: float, border_value: int) -> np.ndarray:
    """Rotate RGB array by degrees."""
    height, width = arr.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = int((height * sine) + (width * cosine))
    new_height = int((height * cosine) + (width * sine))
    matrix[0, 2] += new_width / 2 - width / 2
    matrix[1, 2] += new_height / 2 - height / 2
    return cv2.warpAffine(
        arr, matrix, (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_value, border_value, border_value),
    )


def cv_rgb_to_pil(arr: np.ndarray):
    """Convert numpy RGB array to PIL Image."""
    return PILImage.fromarray(arr, mode="RGB")


def image_to_png(arr: np.ndarray) -> bytes:
    """Convert numpy array to PNG bytes."""
    if arr.ndim == 2:
        image = PILImage.fromarray(arr, mode="L")
    else:
        image = PILImage.fromarray(arr)
    return pil_to_png(image)


def pil_to_png(image) -> bytes:
    """Convert PIL Image to PNG bytes."""
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


# ===========================================================================
# Reader.py Integration: ZXing Priority Candidates
# ===========================================================================

def zxing_priority_candidates(image) -> Iterable[Tuple[str, np.ndarray]]:
    """Generate prioritized candidates for ZXing decoding.
    
    Applies scale + quiet zone + small rotation combinations
    optimized for DataMatrix detection.
    """
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    bases = (
        ("zxing-priority-gray", gray),
        ("zxing-priority-gray-median5", cv2.medianBlur(gray, 5)),
        ("zxing-priority-gray-normalized-median5", cv2.medianBlur(normalize(gray), 5)),
        ("zxing-priority-gray-clahe", clahe.apply(gray)),
        ("zxing-priority-gray-sharp", sharpen(gray)),
    )

    for name, arr in bases:
        for scale in zxing_priority_scales(arr):
            scaled = resize_for_scan(arr, scale)
            for pad_value in (255, 0):
                padded = add_quiet_zone(scaled, value=pad_value)
                padded_name = f"{name}/x{scale}/pad-{pad_value}"
                yield padded_name, padded
                for angle in (-1, 1, -3, 3):
                    yield f"{padded_name}/rotate-{angle}", rotate_image(padded, angle, pad_value)


def zxing_priority_scales(arr: np.ndarray) -> Tuple[int, ...]:
    """Determine scale factors based on image dimensions."""
    max_dim = max(arr.shape[:2])
    if max_dim >= 450:
        return (1, 2)
    if max_dim >= 320:
        return (1, 2, 3)
    return (2, 3, 4)


def expand_candidate(name: str, arr: np.ndarray, mode: str = "balanced") -> Iterable[Tuple[str, np.ndarray]]:
    """Expand a candidate with scale + padding + rotation combinations."""
    scales = (1, 2) if mode == "fast" else (1, 2, 3)
    if mode == "aggressive":
        scales = (1, 2, 3, 4)

    pad_values: Tuple[Optional[int], ...] = (None, 255)
    if mode in {"balanced", "aggressive"}:
        pad_values = (None, 255, 0, 128)

    angles = (0,)
    if mode == "aggressive":
        angles = (0, -1, 1, -3, 3, -7, 7, -14, 14)

    for scale in scales:
        scaled = resize_for_scan(arr, scale)
        for pad_value in pad_values:
            if pad_value is None:
                padded = scaled
                padded_name = f"{name}/x{scale}"
            else:
                padded = add_quiet_zone(scaled, value=pad_value)
                padded_name = f"{name}/x{scale}/pad-{pad_value}"

            for angle in angles:
                if angle == 0:
                    yield padded_name, padded
                else:
                    yield f"{padded_name}/rotate-{angle}", rotate_image(padded, angle, pad_value or 255)


# ===========================================================================
# Reader.py Integration: Grid Reconstruction
# ===========================================================================

def decode_rendered_grid(arr: np.ndarray, variant: str) -> List[BarcodeHit]:
    """Decode barcodes from a rendered grid matrix."""
    if not _HAS_ZXINGCPP:
        return []
    hits: List[BarcodeHit] = []
    grid_binarizers = [
        Binarizer.BoolCast,
        Binarizer.FixedThreshold,
        Binarizer.GlobalHistogram,
    ]
    for binarizer in grid_binarizers:
        try:
            decoded = read_barcodes(
                arr,
                formats=BarcodeFormat.DataMatrix,
                try_rotate=True,
                try_downscale=False,
                try_invert=True,
                binarizer=binarizer,
                is_pure=True,
            )
        except Exception:
            continue
        for barcode in decoded:
            if getattr(barcode, "error", None):
                continue
            text = barcode.text
            if is_plausible_text(text):
                hits.append(BarcodeHit(
                    text=text,
                    format=str(barcode.format).split(".")[-1],
                    preprocessing_method=variant,
                    engine=f"zxing-cpp/{binarizer.name}/grid",
                ))
        if hits:
            return dedupe_hits(hits)
    return []


def grid_boxes(width: int, height: int) -> List[Tuple[int, int, int, int]]:
    """Generate grid margin box candidates sorted by priority."""
    candidates: List[Tuple[float, int, int, int, int]] = []
    x_margins = grid_margin_values(width)
    y_margins = grid_margin_values(height)
    expected_margin = max(width, height) * 0.11
    for left in x_margins:
        for right in x_margins:
            roi_width = width - left - right
            if roi_width < 30:
                continue
            for top in y_margins:
                for bottom in y_margins:
                    roi_height = height - top - bottom
                    if roi_height < 30:
                        continue
                    shape_cost = abs((roi_width / max(1, roi_height)) - 1.0) * 25.0
                    margin_cost = abs((left + right + top + bottom) / 4.0 - expected_margin) * 0.2
                    balance_cost = abs(left - right) * 0.05 + abs(top - bottom) * 0.05
                    candidates.append((shape_cost + margin_cost + balance_cost, left, top, right, bottom))
    return [(left, top, right, bottom) for _, left, top, right, bottom in sorted(candidates)]


def valid_grid_box(box: Tuple[int, int, int, int], width: int, height: int) -> bool:
    """Check if a grid box is valid."""
    left, top, right, bottom = box
    if left < 0 or top < 0 or right < 0 or bottom < 0:
        return False
    if width - left - right < 30 or height - top - bottom < 30:
        return False
    return True


def prioritized_grid_boxes(source: np.ndarray) -> Iterable[Tuple[int, int, int, int]]:
    """Generate prioritized grid box candidates."""
    height, width = source.shape[:2]
    seen: set = set()
    for boxes in (grid_ratio_boxes(width, height), grid_foreground_boxes(source)):
        for box in boxes:
            if not valid_grid_box(box, width, height):
                continue
            if box in seen:
                continue
            seen.add(box)
            yield box
    for box in grid_boxes(width, height):
        if not valid_grid_box(box, width, height):
            continue
        if box in seen:
            continue
        seen.add(box)
        yield box


def grid_ratio_boxes(width: int, height: int) -> List[Tuple[int, int, int, int]]:
    """Generate grid boxes based on ratio offsets."""
    offsets = tuple(range(-5, 6))
    fractions = (0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28, 0.32)
    priority_offsets = (
        (0, 0, 0, 0),
        (0, 1, 0, -1),
        (-1, 1, 1, -1),
        (-2, 2, 2, -2),
        (-3, 2, 3, -2),
        (-3, 3, 3, -3),
        (-4, 4, 4, -4),
        (2, 2, 2, 0),
        (3, 3, 3, 1),
        (-2, 0, 2, 0),
        (0, 2, 0, -2),
    )
    result: List[Tuple[int, int, int, int]] = []
    seen_result: set = set()

    def add_box(box: Tuple[int, int, int, int]) -> None:
        left, top, right, bottom = box
        roi_width = width - left - right
        roi_height = height - top - bottom
        if roi_width < 30 or roi_height < 30:
            return
        if box in seen_result:
            return
        seen_result.add(box)
        result.append(box)

    for fraction in fractions:
        x_margin = round(width * fraction)
        y_margin = round(height * fraction)

        for left_offset, top_offset, right_offset, bottom_offset in priority_offsets:
            add_box(
                (
                    max(0, x_margin + left_offset),
                    max(0, y_margin + top_offset),
                    max(0, x_margin + right_offset),
                    max(0, y_margin + bottom_offset),
                )
            )

        candidates: List[Tuple[float, Tuple[int, int, int, int]]] = []
        for left_offset in offsets:
            for top_offset in offsets:
                for right_offset in offsets:
                    for bottom_offset in offsets:
                        left = max(0, x_margin + left_offset)
                        top = max(0, y_margin + top_offset)
                        right = max(0, x_margin + right_offset)
                        bottom = max(0, y_margin + bottom_offset)
                        roi_width = width - left - right
                        roi_height = height - top - bottom
                        if roi_width < 30 or roi_height < 30:
                            continue
                        shape_cost = abs((roi_width / max(1, roi_height)) - 1.0)
                        balance_cost = (abs(left - right) + abs(top - bottom)) * 0.01
                        candidates.append((shape_cost + balance_cost, (left, top, right, bottom)))

        seen: set = set()
        added = 0
        for _, box in sorted(candidates):
            if box in seen:
                continue
            seen.add(box)
            add_box(box)
            added += 1
            if added >= 25:
                break
    return result


def grid_foreground_boxes(source: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Generate grid boxes based on foreground detection."""
    arr = source if source.ndim == 2 else cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    arr = normalize(arr)
    height, width = arr.shape[:2]
    min_side = min(width, height)
    kernel_size = max(3, int(round(min_side * 0.025)) | 1)
    kernel = np.ones((kernel_size, kernel_size), dtype="uint8")
    threshold_names = ("otsu", "p35", "p40", "p45", "p50", "p55", "p60", "p65")

    candidates: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for threshold_name in threshold_names:
        threshold = foreground_threshold_value(arr, threshold_name)
        for invert in (False, True):
            mask = arr < threshold if not invert else arr > threshold
            fill_ratio = float(mask.mean())
            if fill_ratio < 0.01 or fill_ratio > 0.75:
                continue
            binary = (mask.astype("uint8")) * 255
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            dilated = cv2.dilate(closed, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = [cv2.boundingRect(contour) for contour in contours]

            y_coords, x_coords = np.where(mask)
            if len(x_coords):
                x0, x1 = int(x_coords.min()), int(x_coords.max()) + 1
                y0, y1 = int(y_coords.min()), int(y_coords.max()) + 1
                boxes.append((x0, y0, x1 - x0, y1 - y0))

            for x, y, box_width, box_height in boxes:
                if box_width < min_side * 0.28 or box_height < min_side * 0.28:
                    continue
                if box_width > width * 0.96 or box_height > height * 0.96:
                    continue
                aspect = box_width / max(1, box_height)
                if not 0.65 <= aspect <= 1.35:
                    continue
                for delta in (-4, -2, 0, 2, 4, 6, 8):
                    left = max(0, x - delta)
                    top = max(0, y - delta)
                    right = max(0, width - min(width, x + box_width + delta))
                    bottom = max(0, height - min(height, y + box_height + delta))
                    roi_width = width - left - right
                    roi_height = height - top - bottom
                    shape_cost = abs((roi_width / max(1, roi_height)) - 1.0)
                    area_cost = abs((roi_width * roi_height) / max(1, width * height) - 0.5)
                    candidates.append((shape_cost + area_cost, (left, top, right, bottom)))

    seen: set = set()
    result: List[Tuple[int, int, int, int]] = []
    for _, box in sorted(candidates):
        if box in seen:
            continue
        seen.add(box)
        result.append(box)
    return result[:80]


def foreground_threshold_value(arr: np.ndarray, name: str) -> float:
    """Get threshold value for foreground detection."""
    if name == "otsu":
        threshold, _ = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float(threshold)
    return float(np.percentile(arr, int(name[1:])))


def grid_margin_values(length: int) -> List[int]:
    """Generate grid margin values sorted by priority."""
    values = {
        0, 4, 8, 10, 12, 15, 18, 22,
        int(length * 0.05),
        int(length * 0.08),
        int(length * 0.10),
        int(length * 0.12),
        int(length * 0.15),
    }
    values.update(range(int(length * 0.08), int(length * 0.18) + 1, 2))
    center = length * 0.11
    return sorted((value for value in values if 0 <= value < length // 3),
                  key=lambda value: (abs(value - center), value))


def grid_cell_means(roi: np.ndarray, size: int) -> np.ndarray:
    """Sample cell means from ROI for grid reconstruction."""
    resized = cv2.resize(roi, (size * 9, size * 9), interpolation=cv2.INTER_CUBIC)
    return resized.reshape(size, 9, size, 9).mean(axis=(1, 3))


def grid_threshold_value(sampled: np.ndarray, name: str) -> float:
    """Get threshold value for grid cell sampling."""
    if name == "otsu":
        normalized = cv2.normalize(sampled, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        threshold, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        span = float(sampled.max() - sampled.min())
        return float(threshold) / 255.0 * span + float(sampled.min())
    if name == "mean":
        return float(sampled.mean())
    return float(np.percentile(sampled, int(name[1:])))


def grid_finder_score(matrix: np.ndarray) -> float:
    """Calculate finder pattern score for grid matrix."""
    size = matrix.shape[0]
    top = np.array([index % 2 == 0 for index in range(size)], dtype=bool)
    right = np.array([index % 2 == 1 for index in range(size)], dtype=bool)
    right[-1] = True
    checks = [
        matrix[:, 0] == True,
        matrix[-1, :] == True,
        matrix[0, :] == top,
        matrix[:, -1] == right,
    ]
    return float(np.concatenate(checks).mean())


def force_grid_finder(matrix: np.ndarray) -> np.ndarray:
    """Force correct finder patterns on grid matrix."""
    result = matrix.copy()
    size = result.shape[0]
    result[:, 0] = True
    result[-1, :] = True
    result[0, :] = np.array([index % 2 == 0 for index in range(size)], dtype=bool)
    right = np.array([index % 2 == 1 for index in range(size)], dtype=bool)
    right[-1] = True
    result[:, -1] = right
    result[-1, -1] = True
    return result


def render_grid_matrix(matrix: np.ndarray, module_size: int = 14, quiet_modules: int = 4) -> np.ndarray:
    """Render a boolean grid matrix as a scannable image."""
    size = matrix.shape[0]
    image_size = (size + quiet_modules * 2) * module_size
    rendered = np.full((image_size, image_size), 255, dtype="uint8")
    for y in range(size):
        for x in range(size):
            if matrix[y, x]:
                y0 = (y + quiet_modules) * module_size
                x0 = (x + quiet_modules) * module_size
                rendered[y0 : y0 + module_size, x0 : x0 + module_size] = 0
    return rendered


def decode_with_grid_reconstruction(
    image,
    mode: str = "balanced",
    started: float = 0.0,
    max_seconds: float = 12.0,
) -> Dict[str, Any]:
    """Attempt DataMatrix decoding via grid reconstruction.
    
    Reconstructs grid matrices from image regions and decodes them.
    Returns DecodeResult-like dict.
    """
    if not _HAS_ZXINGCPP:
        return {"barcodes": [], "status": "grid-reconstruction-unavailable",
                "debug_variants": [], "preprocessing_engine": "grid", "decoder_engine": "zxing-cpp"}

    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    sources: List[Tuple[str, np.ndarray]] = [
        ("grid-gray", gray),
        ("grid-gray-normalized", normalize(gray).astype("uint8")),
        ("grid-gray-clahe", clahe.apply(gray)),
        ("grid-gray-median3", cv2.medianBlur(gray, 3)),
    ]
    if mode == "aggressive":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
        sources.extend([
            ("grid-green-red", normalize(g.astype("int16") - r.astype("int16")).astype("uint8")),
            ("grid-blue-red", normalize(b.astype("int16") - r.astype("int16")).astype("uint8")),
            ("grid-saturation", hsv[:, :, 1]),
            ("grid-lab-a", lab[:, :, 1]),
        ])

    attempts = 0
    max_grid_attempts = 30000 if mode == "aggressive" else 9000
    angles = (0, -1, 1, -2, 2, -3, 3)
    sizes = (14, 16) if mode == "balanced" else (14, 16, 18, 20)
    thresholds = ("otsu", "mean", "p42", "p45", "p50", "p55", "p56", "p58", "p59", "p62")
    first_variant = ""
    first_preview: Optional[bytes] = None

    for source_name, source in sources:
        for angle in angles:
            if time.monotonic() - started > max_seconds:
                break
            rotated = rotate_image(source, angle, 255) if angle else source
            height, width = rotated.shape[:2]
            for size in sizes:
                if time.monotonic() - started > max_seconds or attempts >= max_grid_attempts:
                    break
                for left, top, right, bottom in prioritized_grid_boxes(rotated):
                    if time.monotonic() - started > max_seconds or attempts >= max_grid_attempts:
                        break
                    roi = rotated[top : height - bottom, left : width - right]
                    if min(roi.shape[:2]) < size * 3:
                        continue
                    sampled = grid_cell_means(roi, size)
                    for threshold_name in thresholds:
                        threshold = grid_threshold_value(sampled, threshold_name)
                        for invert in (False, True):
                            matrix = sampled < threshold if not invert else sampled >= threshold
                            for rotation in range(4):
                                attempts += 1
                                oriented = np.rot90(matrix, rotation)
                                finder_score = grid_finder_score(oriented)
                                if finder_score < 0.89:
                                    continue
                                for corrected in (False, True):
                                    clean = force_grid_finder(oriented) if corrected else oriented
                                    rendered = render_grid_matrix(clean)
                                    variant = (
                                        f"{source_name}/angle-{angle}/size-{size}/box-{left}-{top}-{right}-{bottom}/"
                                        f"{threshold_name}/invert-{invert}/rot-{rotation}/finder-{finder_score:.3f}/"
                                        f"corrected-{corrected}"
                                    )
                                    if first_variant == "":
                                        first_variant = variant
                                        first_preview = image_to_png(rendered)
                                    hits = decode_rendered_grid(rendered, variant)
                                    if hits:
                                        return {
                                            "barcodes": [_hit_to_dict(h) for h in hits],
                                            "status": "success",
                                            "debug_variants": [variant],
                                            "preprocessing_engine": "grid",
                                            "decoder_engine": "zxing-cpp",
                                        }
        if time.monotonic() - started > max_seconds or attempts >= max_grid_attempts:
            break

    return {
        "barcodes": [],
        "status": "grid-reconstruction-none",
        "debug_variants": [],
        "preprocessing_engine": "grid",
        "decoder_engine": "zxing-cpp",
    }


# ===========================================================================
# ORCHESTRATOR
# ===========================================================================

def decode_image_from_array(
    image_array: np.ndarray,
    selected_methods: Optional[set] = None,
    preprocessor_type: str = PREPROCESSOR_CLASSIC,
    decoder_type: str = DECODER_ZXINGCPP,
) -> Dict[str, Any]:
    """Decode ALL barcodes from image array.
    
    Args:
        image_array: Input image as numpy array (BGR format from cv2)
        selected_methods: Set of method names to apply (for Classic preprocessor).
                         If None or empty, all Classic methods are applied.
                         For VE2 preprocessor, this parameter is ignored.
        preprocessor_type: "classic", "ve2_balanced", "ve2_aggressive", or "all"
        decoder_type: "zxingcpp" or "pylibdmtx"
    
    Returns:
        Result dict with barcodes list, status, and metadata.
    """
    result = DecodeResult(
        preprocessing_engine=preprocessor_type,
        decoder_engine=decoder_type,
    )

    if image_array is None or image_array.size == 0:
        return _result_to_dict(result)

    # Create decoder
    decoder = create_decoder(decoder_type)

    # "all" mode: try all preprocessors in order
    if preprocessor_type == PREPROCESSOR_ALL:
        preprocessors = [
            (PREPROCESSOR_CLASSIC, ClassicPreprocessor()),
            (PREPROCESSOR_VE2_BALANCED, VE2Preprocessor(mode="balanced")),
            (PREPROCESSOR_VE2_AGGRESSIVE, VE2Preprocessor(mode="aggressive")),
        ]
        for ptype, preprocessor in preprocessors:
            result_temp = DecodeResult(
                preprocessing_engine=ptype,
                decoder_engine=decoder_type,
            )

            # Priority candidates (ZXing priority) - try first for fast detection
            if _HAS_PIL and PILImage is not None:
                pil_img = PILImage.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
                for cand_name, cand in zxing_priority_candidates(pil_img):
                    hits = decoder.decode(cand, f"[{ptype}] priority/{cand_name}")
                    if hits:
                        result.barcodes = [_hit_to_dict(h) for h in hits]
                        result.status = "success"
                        result.debug_variants = [cand_name]
                        return _result_to_dict(result)

            # ROI crop for Classic preprocessor
            if ptype == PREPROCESSOR_CLASSIC:
                if selected_methods is None or "roi_crop" in selected_methods:
                    roi_crops = _crop_roi_candidates(image_array)
                    for crop_name, crop in roi_crops:
                        hits = decoder.decode(crop, f"[{ptype}] roi_crop:{crop_name}")
                        if hits:
                            result.barcodes = [_hit_to_dict(h) for h in hits]
                            result.status = "success"
                            result.debug_variants = [v[0] for v in preprocessor.generate_variants(crop, selected_methods)]
                            return _result_to_dict(result)

                        # ROI + variants
                        variants = preprocessor.generate_variants(crop, selected_methods)
                        for vname, v in variants:
                            hits = decoder.decode(v, f"[{ptype}] roi_crop:{crop_name}+{vname}")
                            if hits:
                                result.barcodes = [_hit_to_dict(h) for h in hits]
                                result.status = "success"
                                result.debug_variants = [v[0] for v in variants]
                                return _result_to_dict(result)

            # Generate variants and try each
            variants = preprocessor.generate_variants(image_array, selected_methods)
            result_temp.debug_variants = [v[0] for v in variants]

            for name, var in variants:
                hits = decoder.decode(var, f"[{ptype}] {name}")
                if hits:
                    result.barcodes = [_hit_to_dict(h) for h in hits]
                    result.status = "success"
                    result.debug_variants = [v[0] for v in variants]
                    return _result_to_dict(result)

        # Grid reconstruction fallback (all mode)
        if _HAS_PIL and PILImage is not None:
            pil_img = PILImage.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
            grid_result = decode_with_grid_reconstruction(pil_img, mode="aggressive", started=time.monotonic())
            if grid_result.get("status") == "success":
                result.barcodes = grid_result["barcodes"]
                result.status = "success"
                return _result_to_dict(result)

        return _result_to_dict(result)

    # Single preprocessor mode
    if preprocessor_type == PREPROCESSOR_CLASSIC:
        preprocessor = ClassicPreprocessor()
    elif preprocessor_type == PREPROCESSOR_VE2_BALANCED:
        preprocessor = VE2Preprocessor(mode="balanced")
    elif preprocessor_type == PREPROCESSOR_VE2_AGGRESSIVE:
        preprocessor = VE2Preprocessor(mode="aggressive")
    else:
        raise ValueError(f"Unknown preprocessor type: {preprocessor_type}")

    # Priority candidates (ZXing priority) - try first for fast detection
    if _HAS_PIL and PILImage is not None:
        pil_img = PILImage.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        for cand_name, cand in zxing_priority_candidates(pil_img):
            hits = decoder.decode(cand, f"priority/{cand_name}")
            if hits:
                result.barcodes = [_hit_to_dict(h) for h in hits]
                result.status = "success"
                result.debug_variants = [cand_name]
                return _result_to_dict(result)

    # ROI crop for Classic preprocessor
    if preprocessor_type == PREPROCESSOR_CLASSIC:
        if selected_methods is None or "roi_crop" in selected_methods:
            roi_crops = _crop_roi_candidates(image_array)
            for crop_name, crop in roi_crops:
                hits = decoder.decode(crop, f"roi_crop:{crop_name}")
                if hits:
                    result.barcodes = [_hit_to_dict(h) for h in hits]
                    result.status = "success"
                    return _result_to_dict(result)

                # ROI + variants
                variants = preprocessor.generate_variants(crop, selected_methods)
                for vname, v in variants:
                    hits = decoder.decode(v, f"roi_crop:{crop_name}+{vname}")
                    if hits:
                        result.barcodes = [_hit_to_dict(h) for h in hits]
                        result.status = "success"
                        return _result_to_dict(result)

    # Generate variants and try each
    variants = preprocessor.generate_variants(image_array, selected_methods)
    result.debug_variants = [v[0] for v in variants]

    for name, var in variants:
        hits = decoder.decode(var, name)
        if hits:
            result.barcodes = [_hit_to_dict(h) for h in hits]
            result.status = "success"
            return _result_to_dict(result)

    # Grid reconstruction fallback
    if _HAS_PIL and PILImage is not None:
        pil_img = PILImage.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        ve2_mode = "aggressive" if preprocessor_type == PREPROCESSOR_VE2_AGGRESSIVE else "balanced"
        grid_result = decode_with_grid_reconstruction(pil_img, mode=ve2_mode, started=time.monotonic())
        if grid_result.get("status") == "success":
            result.barcodes = grid_result["barcodes"]
            result.status = "success"
            return _result_to_dict(result)

    return _result_to_dict(result)


def _hit_to_dict(hit: BarcodeHit) -> Dict[str, str]:
    """Convert BarcodeHit to dict."""
    return {
        "text": hit.text,
        "format": hit.format,
        "preprocessing_method": hit.preprocessing_method,
    }


def _result_to_dict(result: DecodeResult) -> Dict[str, Any]:
    """Convert DecodeResult to dict."""
    return {
        "barcodes": result.barcodes,
        "status": result.status,
        "debug_variants": result.debug_variants,
        "preprocessing_engine": result.preprocessing_engine,
        "decoder_engine": result.decoder_engine,
    }


# ===========================================================================
# Legacy Compatibility Functions
# ===========================================================================

def decode_multiple_barcodes(image: np.ndarray) -> List[Dict[str, str]]:
    """Legacy function - decode barcodes using default settings."""
    result = decode_image_from_array(image)
    return result.get("barcodes", [])


def image_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy array to base64 string for display."""
    _, buffer = cv2.imencode('.png', image_array)
    return base64.b64encode(buffer).decode('utf-8')


def results_to_csv_rows(results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert recognition results to CSV rows.
    
    Note: decoded_text is prefixed with \t to prevent Excel from stripping leading zeros.
    """
    csv_rows = []
    for r in results:
        barcodes = r.get("barcodes", [])
        if barcodes:
            for bc_idx, bc in enumerate(barcodes):
                csv_rows.append({
                    "image_file": r.get("image_file", ""),
                    "barcode_index": bc_idx + 1,
                    "decoded_text": "\t" + bc.get("text", ""),
                    "format": bc.get("format", ""),
                    "preprocessing_method": bc.get("preprocessing_method", ""),
                    "status": r.get("status", "failed"),
                })
        else:
            csv_rows.append({
                "image_file": r.get("image_file", ""),
                "barcode_index": "-",
                "decoded_text": "",
                "format": "",
                "preprocessing_method": "",
                "status": r.get("status", "failed"),
            })
    return csv_rows


def csv_rows_to_stringio(csv_rows: List[Dict[str, str]]) -> str:
    """Convert CSV rows to StringIO content."""
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=[
        "image_file", "barcode_index", "decoded_text", "format",
        "preprocessing_method", "status"
    ])
    writer.writeheader()
    writer.writerows(csv_rows)
    return csv_buffer.getvalue()
