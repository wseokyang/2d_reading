from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import zxingcpp

try:
    from pylibdmtx.pylibdmtx import decode as dmtx_decode
    from pylibdmtx.wrapper import DmtxSymbolSize
except Exception:  # pragma: no cover - optional runtime fallback
    dmtx_decode = None
    DmtxSymbolSize = None


@dataclass(frozen=True)
class ReaderOptions:
    mode: str = "balanced"
    brightness: int = 0
    contrast: float = 1.0
    saturation: float = 1.0
    sharpness: float = 1.0
    max_seconds: float = 12.0
    max_attempts: int = 650
    use_libdmtx: bool = True
    use_glm: bool = False
    glm_timeout_seconds: float | None = None


@dataclass(frozen=True)
class CodeHit:
    text: str
    engine: str
    format: str
    variant: str
    confidence: str


@dataclass(frozen=True)
class ReadResult:
    ok: bool
    hits: list[CodeHit]
    attempts: int
    elapsed_seconds: float
    best_variant: str
    preview_png: bytes | None
    error: str | None = None
    assistance: dict[str, object] | None = None


BINARIZERS = (
    zxingcpp.Binarizer.LocalAverage,
    zxingcpp.Binarizer.GlobalHistogram,
    zxingcpp.Binarizer.FixedThreshold,
)

GRID_BINARIZERS = (
    zxingcpp.Binarizer.BoolCast,
    zxingcpp.Binarizer.FixedThreshold,
    zxingcpp.Binarizer.GlobalHistogram,
)


def read_image_bytes(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")


def read_datamatrix(image: Image.Image, options: ReaderOptions | None = None) -> ReadResult:
    options = options or ReaderOptions()
    if options.use_glm:
        return read_with_glm(image, options)
    started = time.monotonic()
    attempts = 0
    seen: set[str] = set()
    hits: list[CodeHit] = []
    first_preview: bytes | None = None
    first_variant = ""

    adjusted = apply_manual_adjustments(image.convert("RGB"), options)

    first_hits = decode_with_zxing(np.array(adjusted), "original-rgb")
    if first_hits:
        return ReadResult(
            ok=True,
            hits=first_hits,
            attempts=1,
            elapsed_seconds=time.monotonic() - started,
            best_variant=first_hits[0].variant,
            preview_png=pil_to_png(adjusted),
        )

    attempts = 1
    if options.mode in {"balanced", "aggressive"}:
        for name, candidate in color_denoise_candidates(adjusted):
            if time.monotonic() - started > min(options.max_seconds, 2.0) or attempts >= options.max_attempts:
                break
            attempts += 1
            channel_hits = decode_with_zxing(candidate, name)
            if channel_hits:
                return ReadResult(True, channel_hits, attempts, time.monotonic() - started,
                                  channel_hits[0].variant, image_to_png(candidate))

    if options.mode in {"balanced", "aggressive"}:
        for priority_name, priority_candidate in zxing_priority_candidates(adjusted):
            if time.monotonic() - started > min(options.max_seconds, 6.0):
                break
            attempts += 1
            priority_hits = decode_with_zxing(priority_candidate, priority_name)
            if priority_hits:
                return ReadResult(
                    ok=True,
                    hits=priority_hits,
                    attempts=attempts,
                    elapsed_seconds=time.monotonic() - started,
                    best_variant=priority_hits[0].variant,
                    preview_png=image_to_png(priority_candidate),
                )

    if options.mode in {"balanced", "aggressive"}:
        from grid_recovery import recover_grid

        remaining = options.max_seconds - (time.monotonic() - started)
        recovered = recover_grid(adjusted, max_seconds=min(3.0, max(0.0, remaining)))
        if recovered:
            hit = CodeHit(recovered.text, "zxing-cpp/center-grid", "DataMatrix",
                          recovered.variant, "decoded-reconstructed")
            return ReadResult(True, [hit], attempts + recovered.attempts,
                              time.monotonic() - started, recovered.variant,
                              image_to_png(recovered.rendered))

        grid_fallback = decode_with_grid_reconstruction(adjusted, options, started)
        attempts += grid_fallback.attempts
        if grid_fallback.hits:
            return ReadResult(
                ok=True,
                hits=grid_fallback.hits,
                attempts=attempts,
                elapsed_seconds=time.monotonic() - started,
                best_variant=grid_fallback.best_variant,
                preview_png=grid_fallback.preview_png,
            )

    if options.use_libdmtx and dmtx_decode is not None:
        fallback = decode_with_libdmtx(adjusted, options, started)
        attempts += fallback.attempts
        for hit in fallback.hits:
            if hit.text not in seen:
                seen.add(hit.text)
                hits.append(hit)
        if hits:
            return ReadResult(
                ok=True,
                hits=hits,
                attempts=attempts,
                elapsed_seconds=time.monotonic() - started,
                best_variant=fallback.best_variant,
                preview_png=fallback.preview_png,
            )

    for variant_name, arr in iter_variants(adjusted, options.mode):
        if time.monotonic() - started > options.max_seconds:
            break

        for expanded_name, candidate in expand_candidate(variant_name, arr, options.mode):
            if time.monotonic() - started > options.max_seconds or attempts >= options.max_attempts:
                break
            attempts += 1
            if first_preview is None:
                first_preview = image_to_png(candidate)
                first_variant = expanded_name

            zxing_hits = decode_with_zxing(candidate, expanded_name)
            for hit in zxing_hits:
                if hit.text not in seen:
                    seen.add(hit.text)
                    hits.append(hit)

            if hits:
                return ReadResult(
                    ok=True,
                    hits=hits,
                    attempts=attempts,
                    elapsed_seconds=time.monotonic() - started,
                    best_variant=expanded_name,
                    preview_png=image_to_png(candidate),
                )
        if time.monotonic() - started > options.max_seconds or attempts >= options.max_attempts:
            break

    return ReadResult(
        ok=False,
        hits=[],
        attempts=attempts,
        elapsed_seconds=time.monotonic() - started,
        best_variant=first_variant,
        preview_png=first_preview,
    )


def apply_manual_adjustments(image: Image.Image, options: ReaderOptions) -> Image.Image:
    result = ImageEnhance.Brightness(image).enhance(1.0 + options.brightness / 100.0)
    result = ImageEnhance.Contrast(result).enhance(options.contrast)
    result = ImageEnhance.Color(result).enhance(options.saturation)
    result = ImageEnhance.Sharpness(result).enhance(options.sharpness)
    return result


def read_with_glm(image: Image.Image, options: ReaderOptions) -> ReadResult:
    """Use vision hints as an execution plan, then validate with local ECC decoding.

    The API has its own timeout. Local candidate processing and fallback share
    max_seconds; a failed API call cannot consume the local decoding allowance.
    """
    from glm_assist import GLMOptions, iter_glm_candidates, suggest_preprocessing

    started = time.monotonic()
    adjusted = apply_manual_adjustments(image.convert("RGB"), options)
    direct_hits = decode_with_zxing(np.asarray(adjusted), "original-rgb")
    if direct_hits:
        return ReadResult(True, direct_hits, 1, time.monotonic() - started,
                          direct_hits[0].variant, pil_to_png(adjusted),
                          assistance={"status": "not-needed", "used": False})

    config = GLMOptions.from_env()
    if options.glm_timeout_seconds is not None:
        config = replace(config, timeout_seconds=options.glm_timeout_seconds)
    plan = suggest_preprocessing(adjusted, config)
    details: dict[str, object] = {
        "status": plan.status,
        "model": plan.model,
        "elapsed_seconds": round(plan.elapsed_seconds, 3),
        "hints": [asdict(hint) for hint in plan.hints],
        "error": plan.error,
        "used": False,
    }
    local_started = time.monotonic()
    attempts = 1
    # Preserve most of the local allowance for deterministic fallback if the
    # visual suggestion is inaccurate. Never execute code returned by a model.
    hint_budget = min(3.0, options.max_seconds * 0.25)
    for name, candidate in iter_glm_candidates(adjusted, plan):
        if time.monotonic() - local_started >= hint_budget:
            break
        attempts += 1
        hits = decode_with_zxing(candidate, name)
        if hits:
            details["used"] = True
            return ReadResult(True, hits, attempts, time.monotonic() - started,
                              hits[0].variant, image_to_png(candidate), assistance=details)

    if plan.hints:
        from grid_recovery import recover_grid

        for index, hint in enumerate(plan.hints):
            remaining = hint_budget - (time.monotonic() - local_started)
            if remaining <= 0:
                break
            if hint.grid_size is None:
                continue
            left, top, right, bottom = hint.bbox
            crop = adjusted.crop((int(left * adjusted.width), int(top * adjusted.height),
                                  int(right * adjusted.width), int(bottom * adjusted.height)))
            # A model's ROI can touch the finder edges. The geometry search
            # needs some margin around the symbol to fit module centers.
            crop = ImageOps.expand(crop, border=max(4, round(min(crop.size) * 0.12)), fill="white")
            recovered = recover_grid(crop, max_seconds=min(1.0, remaining), grid_sizes=(hint.grid_size,))
            if recovered:
                details["used"] = True
                variant = f"glm/hint-{index + 1}/" + recovered.variant
                hit = CodeHit(recovered.text, "zxing-cpp/center-grid", "DataMatrix",
                              variant, "decoded-reconstructed")
                return ReadResult(True, [hit], attempts + recovered.attempts,
                                  time.monotonic() - started, variant,
                                  image_to_png(recovered.rendered), assistance=details)

    remaining = max(0.0, options.max_seconds - (time.monotonic() - local_started))
    fallback = read_datamatrix(image, replace(options, use_glm=False, max_seconds=remaining))
    return replace(fallback, attempts=attempts + fallback.attempts,
                   elapsed_seconds=time.monotonic() - started, assistance=details)


def color_denoise_candidates(image: Image.Image) -> Iterable[tuple[str, np.ndarray]]:
    """Separate marking color from metal texture before geometry-heavy searches.

    Work in floating point for opponent channels: uint8 subtraction would wrap.
    Normalize *before* smoothing so fixed thresholds share a predictable scale.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    red, green, blue = cv2.split(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sources = (
        ("blue", blue),
        ("yellow-blue", (red + green) * 0.5 - blue),
        ("gray", gray),
        ("green-red", green - red),
        ("red", red),
        ("green", green),
    )
    # Interleave channels so an unsuccessful grayscale search cannot consume
    # the budget before the color information is examined.
    normalized = [(name, normalize(source).astype(np.uint8)) for name, source in sources]
    for sigma in (3.0, 2.0, 1.0, 4.0):
        for name, source in normalized:
            blurred = cv2.GaussianBlur(source, (0, 0), sigma)
            prefix = f"color-{name}/gaussian-{sigma:g}"
            yield prefix, blurred
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            yield f"{prefix}/otsu", binary
            for threshold in (110, 130, 90, 150, 70, 170, 190):
                yield f"{prefix}/threshold-{threshold}", np.uint8(blurred > threshold) * 255


def iter_variants(image: Image.Image, mode: str) -> Iterable[tuple[str, np.ndarray]]:
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    yield "original-rgb", rgb
    yield "gray", gray
    yield "gray-median5", cv2.medianBlur(gray, 5)
    yield "gray-normalized", normalize(gray)
    yield "gray-clahe", clahe.apply(gray)
    yield "gray-sharpened", sharpen(gray)

    if mode in {"balanced", "aggressive"}:
        base_channels = (
            ("red", bgr[:, :, 2]),
            ("green", bgr[:, :, 1]),
            ("blue", bgr[:, :, 0]),
            ("hsv-value", hsv[:, :, 2]),
            ("hsv-saturation", hsv[:, :, 1]),
            ("lab-lightness", lab[:, :, 0]),
        )
        for name, channel in base_channels:
            yield name, channel
            yield f"{name}-normalized", normalize(channel)
            yield f"{name}-clahe", clahe.apply(channel)

        for name, channel in (("gray", gray), ("lab-lightness", lab[:, :, 0]), ("green", bgr[:, :, 1])):
            for gamma in (0.45, 0.65, 0.85, 1.25, 1.6):
                yield f"{name}-gamma-{gamma}", gamma_correct(channel, gamma)
            for alpha, beta in ((1.4, 0), (1.8, -30), (2.2, -70), (1.8, 35), (1.2, 60)):
                yield f"{name}-contrast-{alpha}-{beta}", cv2.convertScaleAbs(channel, alpha=alpha, beta=beta)

    threshold_sources = [("gray", gray), ("gray-clahe", clahe.apply(gray)), ("gray-normalized", normalize(gray))]
    if mode == "aggressive":
        threshold_sources.extend(
            [
                ("red", bgr[:, :, 2]),
                ("green", bgr[:, :, 1]),
                ("lab-lightness", lab[:, :, 0]),
                ("hsv-value", hsv[:, :, 2]),
            ]
        )

    for name, source in threshold_sources:
        yield from threshold_variants(name, source)


def threshold_variants(name: str, source: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
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


def zxing_priority_candidates(image: Image.Image) -> Iterable[tuple[str, np.ndarray]]:
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


def expand_candidate(name: str, arr: np.ndarray, mode: str) -> Iterable[tuple[str, np.ndarray]]:
    scales = (1, 2) if mode == "fast" else (1, 2, 3)
    if mode == "aggressive":
        scales = (1, 2, 3, 4)

    pad_values: tuple[int | None, ...] = (None, 255)
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


def decode_with_zxing(arr: np.ndarray, variant: str) -> list[CodeHit]:
    # This module's color arrays are RGB; ZXing's numpy color contract is BGR.
    # Explicit grayscale conversion keeps the channel interpretation consistent.
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hits: list[CodeHit] = []
    for binarizer in BINARIZERS:
        for pure in (False,):
            for downscale in (False, True):
                try:
                    decoded = zxingcpp.read_barcodes(
                        arr,
                        formats=zxingcpp.BarcodeFormat.DataMatrix,
                        try_rotate=True,
                        try_downscale=downscale,
                        try_invert=True,
                        binarizer=binarizer,
                        is_pure=pure,
                    )
                except Exception:
                    continue
                for barcode in decoded:
                    if getattr(barcode, "error", None):
                        continue
                    text = barcode.text
                    if is_plausible_text(text):
                        hits.append(
                            CodeHit(
                                text=text,
                                engine=f"zxing-cpp/{binarizer.name}",
                                format=str(barcode.format).split(".")[-1],
                                variant=f"{variant}/pure-{pure}/downscale-{downscale}",
                                confidence="decoded",
                            )
                        )
                if hits:
                    return dedupe_hits(hits)
    return []


def decode_with_grid_reconstruction(image: Image.Image, options: ReaderOptions, started: float) -> ReadResult:
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    sources: list[tuple[str, np.ndarray]] = [
        ("grid-gray", gray),
        ("grid-gray-normalized", normalize(gray).astype("uint8")),
        ("grid-gray-clahe", clahe.apply(gray)),
        ("grid-gray-median3", cv2.medianBlur(gray, 3)),
    ]
    if options.mode == "aggressive":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
        sources.extend(
            [
                ("grid-green-red", normalize(g.astype("int16") - r.astype("int16")).astype("uint8")),
                ("grid-blue-red", normalize(b.astype("int16") - r.astype("int16")).astype("uint8")),
                ("grid-saturation", hsv[:, :, 1]),
                ("grid-lab-a", lab[:, :, 1]),
            ]
        )

    attempts = 0
    max_grid_attempts = 30000 if options.mode == "aggressive" else 9000
    angles = (0, -1, 1, -2, 2, -3, 3)
    sizes = (14, 16) if options.mode == "balanced" else (14, 16, 18, 20)
    thresholds = ("otsu", "mean", "p42", "p45", "p50", "p55", "p56", "p58", "p59", "p62")

    for source_name, source in sources:
        for angle in angles:
            if time.monotonic() - started > options.max_seconds or attempts >= max_grid_attempts:
                break
            rotated = rotate_image(source, angle, 255) if angle else source
            height, width = rotated.shape[:2]
            for size in sizes:
                if time.monotonic() - started > options.max_seconds or attempts >= max_grid_attempts:
                    break
                for left, top, right, bottom in prioritized_grid_boxes(rotated):
                    if time.monotonic() - started > options.max_seconds or attempts >= max_grid_attempts:
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
                                    hits = decode_rendered_grid(rendered, variant)
                                    if hits:
                                        return ReadResult(
                                            True,
                                            hits,
                                            attempts,
                                            time.monotonic() - started,
                                            variant,
                                            image_to_png(rendered),
                                        )
            if time.monotonic() - started > options.max_seconds or attempts >= max_grid_attempts:
                break

    return ReadResult(False, [], attempts, time.monotonic() - started, "grid-reconstruction-none", None)


def decode_rendered_grid(arr: np.ndarray, variant: str) -> list[CodeHit]:
    hits: list[CodeHit] = []
    for binarizer in GRID_BINARIZERS:
        try:
            decoded = zxingcpp.read_barcodes(
                arr,
                formats=zxingcpp.BarcodeFormat.DataMatrix,
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
                hits.append(
                    CodeHit(
                        text=text,
                        engine=f"zxing-cpp/{binarizer.name}/grid",
                        format=str(barcode.format).split(".")[-1],
                        variant=variant,
                        confidence="decoded-reconstructed",
                    )
                )
        if hits:
            return dedupe_hits(hits)
    return []


def decode_with_libdmtx(image: Image.Image, options: ReaderOptions, started: float) -> ReadResult:
    if dmtx_decode is None or time.monotonic() - started > options.max_seconds:
        return ReadResult(False, [], 0, time.monotonic() - started, "libdmtx-unavailable", None)

    attempts = 0
    hits: list[CodeHit] = []
    timeout = 650 if options.mode == "aggressive" else 450

    for name, candidate, params_list in dmtx_priority_candidates(image, options.mode):
        if time.monotonic() - started > options.max_seconds:
            break
        for params in params_list:
            if time.monotonic() - started > options.max_seconds:
                break
            attempts += 1
            try:
                decoded = dmtx_decode(candidate, timeout=timeout, max_count=1, **params)
            except Exception:
                continue
            for item in decoded:
                text = item.data.decode("utf-8", errors="replace")
                if is_plausible_text(text):
                    hits.append(
                        CodeHit(
                            text=text,
                            engine=f"pylibdmtx/{format_dmtx_params(params)}",
                            format="DataMatrix",
                            variant=name,
                            confidence="decoded",
                        )
                    )
            if hits:
                return ReadResult(True, dedupe_hits(hits), attempts, time.monotonic() - started, name, pil_to_png(candidate))

    for name, candidate in dmtx_candidates(image, options.mode):
        if time.monotonic() - started > options.max_seconds:
            break
        for params in dmtx_params(options.mode):
            if time.monotonic() - started > options.max_seconds:
                break
            attempts += 1
            try:
                decoded = dmtx_decode(candidate, timeout=timeout, max_count=1, **params)
            except Exception:
                continue
            for item in decoded:
                text = item.data.decode("utf-8", errors="replace")
                if is_plausible_text(text):
                    hits.append(
                        CodeHit(
                            text=text,
                            engine=f"pylibdmtx/{format_dmtx_params(params)}",
                            format="DataMatrix",
                            variant=name,
                            confidence="decoded",
                        )
                    )
            if hits:
                return ReadResult(True, dedupe_hits(hits), attempts, time.monotonic() - started, name, pil_to_png(candidate))

    return ReadResult(False, [], attempts, time.monotonic() - started, "libdmtx-none", None)


def dmtx_priority_candidates(image: Image.Image, mode: str) -> Iterable[tuple[str, Image.Image, list[dict[str, object]]]]:
    square_auto = DmtxSymbolSize.DmtxSymbolSquareAuto if DmtxSymbolSize is not None else None
    rgb = image.convert("RGB")
    gray = rgb.convert("L")
    rgb_arr = np.array(rgb)

    common = dmtx_named_params(
        [
            {},
            {"shrink": 2},
            {"shrink": 3},
            {"shrink": 4},
            {"shrink": 5},
            {"shrink": 2, "shape": square_auto},
            {"shrink": 3, "shape": square_auto},
            {"shrink": 4, "shape": square_auto},
            {"shrink": 5, "shape": square_auto},
        ]
    )
    shrink3 = dmtx_named_params([{"shrink": 3}, {"shrink": 3, "shape": square_auto}])
    shrink4 = dmtx_named_params([{"shrink": 4}, {"shrink": 4, "shape": square_auto}])
    shrink5 = dmtx_named_params([{"shrink": 5}, {"shrink": 5, "shape": square_auto}])

    yield "libdmtx-priority-original", rgb, common
    yield "libdmtx-priority-original-cv-rotate--9", cv_rgb_to_pil(cv_rotate_rgb(rgb_arr, -9, 255)), shrink5
    yield "libdmtx-priority-original-cv-x2-rotate-3", cv_rgb_to_pil(cv_rotate_rgb(cv_resize_rgb(rgb_arr, 2), 3, 255)), shrink5
    yield "libdmtx-priority-gray", gray, shrink4
    yield "libdmtx-priority-gray-autocontrast", ImageOps.autocontrast(gray), shrink3
    yield "libdmtx-priority-original-pad-white", pil_pad(rgb, 0.18, "white"), shrink3

    if mode == "aggressive":
        yield "libdmtx-priority-original-x2", pil_resize(rgb, 2), shrink5


def dmtx_named_params(params: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for item in params:
        if any(value is None for value in item.values()):
            continue
        result.append(item)
    return result


def dmtx_candidates(image: Image.Image, mode: str) -> Iterable[tuple[str, Image.Image]]:
    rgb = image.convert("RGB")
    gray = rgb.convert("L")
    rgb_arr = np.array(rgb)

    yield "libdmtx-original", rgb
    yield "libdmtx-gray", gray
    yield "libdmtx-gray-autocontrast", ImageOps.autocontrast(gray)
    yield "libdmtx-original-pad-white", pil_pad(rgb, 0.18, "white")
    yield "libdmtx-original-pad-black", pil_pad(rgb, 0.18, "black")
    yield "libdmtx-original-cv-rotate--9", cv_rgb_to_pil(cv_rotate_rgb(rgb_arr, -9, 255))
    yield "libdmtx-original-cv-x2-rotate-3", cv_rgb_to_pil(cv_rotate_rgb(cv_resize_rgb(rgb_arr, 2), 3, 255))
    yield "libdmtx-original-rotate--9", pil_rotate(rgb, -9, "white")
    yield "libdmtx-original-rotate-3", pil_rotate(rgb, 3, "white")
    yield "libdmtx-original-x2-rotate-3", pil_rotate(pil_resize(rgb, 2), 3, "white")

    if mode in {"balanced", "aggressive"}:
        yield "libdmtx-original-x2", pil_resize(rgb, 2)
        yield "libdmtx-original-x2-pad-white", pil_pad(pil_resize(rgb, 2), 0.18, "white")
        yield "libdmtx-gray-x2", pil_resize(gray, 2)
        yield "libdmtx-gray-x2-rotate--9", pil_rotate(pil_resize(gray, 2), -9, "white")

    if mode == "aggressive":
        bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
        channel_candidates = (
            ("green-red", normalize(g.astype("int16") - r.astype("int16"))),
            ("blue-red", normalize(b.astype("int16") - r.astype("int16"))),
            ("lab-a", lab[:, :, 1]),
            ("cr", ycrcb[:, :, 1]),
            ("cb", ycrcb[:, :, 2]),
        )
        for name, arr in channel_candidates:
            candidate = Image.fromarray(arr, mode="L")
            yield f"libdmtx-{name}", candidate
            yield f"libdmtx-{name}-autocontrast", ImageOps.autocontrast(candidate)
            yield f"libdmtx-{name}-x2", pil_resize(candidate, 2)


def dmtx_params(mode: str) -> list[dict[str, object]]:
    square_auto = DmtxSymbolSize.DmtxSymbolSquareAuto if DmtxSymbolSize is not None else None
    params: list[dict[str, object]] = [{}]
    for shrink in (2, 3, 4, 5):
        params.append({"shrink": shrink})
        if square_auto is not None:
            params.append({"shrink": shrink, "shape": square_auto})
    if mode == "aggressive":
        for shrink in (6, 7, 8):
            params.append({"shrink": shrink})
            if square_auto is not None:
                params.append({"shrink": shrink, "shape": square_auto})
        if square_auto is not None:
            for threshold in (16, 32, 48, 64, 96, 128, 160, 192):
                params.append({"threshold": threshold, "shape": square_auto})
    return params


def format_dmtx_params(params: dict[str, object]) -> str:
    if not params:
        return "default"
    parts = []
    for key, value in params.items():
        if key == "shape":
            value = getattr(value, "name", value)
        parts.append(f"{key}={value}")
    return ",".join(parts)


def is_plausible_text(text: str) -> bool:
    if not text:
        return False
    if len(text.strip()) < 6:
        return False
    if any(ch.isalnum() for ch in text):
        return True
    printable = sum(ch.isprintable() and not ch.isspace() for ch in text)
    return printable >= 4


def dedupe_hits(hits: list[CodeHit]) -> list[CodeHit]:
    seen: set[str] = set()
    result: list[CodeHit] = []
    for hit in hits:
        if hit.text in seen:
            continue
        seen.add(hit.text)
        result.append(hit)
    return result


def grid_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    candidates: list[tuple[float, int, int, int, int]] = []
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


def zxing_priority_scales(arr: np.ndarray) -> tuple[int, ...]:
    max_dim = max(arr.shape[:2])
    if max_dim >= 450:
        return (1, 2)
    if max_dim >= 320:
        return (1, 2, 3)
    return (2, 3, 4)


def prioritized_grid_boxes(source: np.ndarray) -> Iterable[tuple[int, int, int, int]]:
    height, width = source.shape[:2]
    seen: set[tuple[int, int, int, int]] = set()
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


def valid_grid_box(box: tuple[int, int, int, int], width: int, height: int) -> bool:
    left, top, right, bottom = box
    if left < 0 or top < 0 or right < 0 or bottom < 0:
        return False
    if width - left - right < 30 or height - top - bottom < 30:
        return False
    return True


def grid_ratio_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
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
    result: list[tuple[int, int, int, int]] = []
    seen_result: set[tuple[int, int, int, int]] = set()

    def add_box(box: tuple[int, int, int, int]) -> None:
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

        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
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

        seen: set[tuple[int, int, int, int]] = set()
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


def grid_foreground_boxes(source: np.ndarray) -> list[tuple[int, int, int, int]]:
    arr = source if source.ndim == 2 else cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    arr = normalize(arr)
    height, width = arr.shape[:2]
    min_side = min(width, height)
    kernel_size = max(3, int(round(min_side * 0.025)) | 1)
    kernel = np.ones((kernel_size, kernel_size), dtype="uint8")
    threshold_names = ("otsu", "p35", "p40", "p45", "p50", "p55", "p60", "p65")

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
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

    seen: set[tuple[int, int, int, int]] = set()
    result: list[tuple[int, int, int, int]] = []
    for _, box in sorted(candidates):
        if box in seen:
            continue
        seen.add(box)
        result.append(box)
    return result[:80]


def foreground_threshold_value(arr: np.ndarray, name: str) -> float:
    if name == "otsu":
        threshold, _ = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float(threshold)
    return float(np.percentile(arr, int(name[1:])))


def grid_margin_values(length: int) -> list[int]:
    values = {
        0,
        4,
        8,
        10,
        12,
        15,
        18,
        22,
        int(length * 0.05),
        int(length * 0.08),
        int(length * 0.10),
        int(length * 0.12),
        int(length * 0.15),
    }
    values.update(range(int(length * 0.08), int(length * 0.18) + 1, 2))
    center = length * 0.11
    return sorted((value for value in values if 0 <= value < length // 3), key=lambda value: (abs(value - center), value))


def grid_cell_means(roi: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(roi, (size * 9, size * 9), interpolation=cv2.INTER_CUBIC)
    return resized.reshape(size, 9, size, 9).mean(axis=(1, 3))


def grid_threshold_value(sampled: np.ndarray, name: str) -> float:
    if name == "otsu":
        normalized = cv2.normalize(sampled, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        threshold, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        span = float(sampled.max() - sampled.min())
        return float(threshold) / 255.0 * span + float(sampled.min())
    if name == "mean":
        return float(sampled.mean())
    return float(np.percentile(sampled, int(name[1:])))


def grid_finder_score(matrix: np.ndarray) -> float:
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


def normalize(arr: np.ndarray) -> np.ndarray:
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)


def sharpen(arr: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(arr, (0, 0), 1.0)
    return cv2.addWeighted(arr, 1.7, blur, -0.7, 0)


def gamma_correct(arr: np.ndarray, gamma: float) -> np.ndarray:
    table = (np.power(np.arange(256) / 255.0, gamma) * 255).astype("uint8")
    return cv2.LUT(arr, table)


def resize_for_scan(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return arr
    h, w = arr.shape[:2]
    unique_count = len(np.unique(arr)) if arr.ndim == 2 else 999
    interpolation = cv2.INTER_NEAREST if unique_count <= 8 else cv2.INTER_CUBIC
    return cv2.resize(arr, (w * scale, h * scale), interpolation=interpolation)


def add_quiet_zone(arr: np.ndarray, value: int = 255) -> np.ndarray:
    h, w = arr.shape[:2]
    pad = max(8, int(max(h, w) * 0.18))
    return cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=value)


def rotate_image(arr: np.ndarray, degrees: float, border_value: int = 255) -> np.ndarray:
    h, w = arr.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_w = int((h * sine) + (w * cosine))
    new_h = int((h * cosine) + (w * sine))
    matrix[0, 2] += new_w / 2 - w / 2
    matrix[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(
        arr,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def pil_resize(image: Image.Image, scale: int) -> Image.Image:
    width, height = image.size
    return image.resize((width * scale, height * scale), Image.Resampling.BICUBIC)


def pil_pad(image: Image.Image, fraction: float, color: str) -> Image.Image:
    width, height = image.size
    pad = max(8, int(max(width, height) * fraction))
    if image.mode == "L":
        fill = 255 if color == "white" else 0
    else:
        fill = (255, 255, 255) if color == "white" else (0, 0, 0)
    return ImageOps.expand(image, border=pad, fill=fill)


def pil_rotate(image: Image.Image, degrees: float, color: str) -> Image.Image:
    if image.mode == "L":
        fill = 255 if color == "white" else 0
    else:
        fill = (255, 255, 255) if color == "white" else (0, 0, 0)
    return image.rotate(degrees, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=fill)


def cv_resize_rgb(arr: np.ndarray, scale: int) -> np.ndarray:
    height, width = arr.shape[:2]
    return cv2.resize(arr, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)


def cv_rotate_rgb(arr: np.ndarray, degrees: float, border_value: int) -> np.ndarray:
    height, width = arr.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = int((height * sine) + (width * cosine))
    new_height = int((height * cosine) + (width * sine))
    matrix[0, 2] += new_width / 2 - width / 2
    matrix[1, 2] += new_height / 2 - height / 2
    return cv2.warpAffine(
        arr,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_value, border_value, border_value),
    )


def cv_rgb_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr, mode="RGB")


def image_to_png(arr: np.ndarray) -> bytes:
    if arr.ndim == 2:
        image = Image.fromarray(arr, mode="L")
    else:
        image = Image.fromarray(arr)
    return pil_to_png(image)


def pil_to_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
