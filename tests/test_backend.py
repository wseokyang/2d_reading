from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np
import zxingcpp

from backend import decode_datamatrix_from_array
from reader import CodeHit, ReadResult


def decoded_result(*, assistance=None):
    return ReadResult(
        ok=True,
        hits=[CodeHit("001234567890", "zxing-cpp/LocalAverage", "Data Matrix", "blue/blur-3", "decoder")],
        attempts=2,
        elapsed_seconds=0.1,
        best_variant="blue/blur-3",
        preview_png=None,
        assistance=assistance,
    )


class ArrayBackendTests(unittest.TestCase):
    @patch("backend.read_datamatrix", return_value=decoded_result())
    def test_opencv_bgr_is_converted_once_without_changing_input(self, read):
        bgr = np.array([[[17, 51, 203], [211, 100, 29]]], dtype=np.uint8)
        original = bgr.copy()
        decode_datamatrix_from_array(bgr)
        pil_image = read.call_args.args[0]
        self.assertEqual("RGB", pil_image.mode)
        self.assertEqual((203, 51, 17), pil_image.getpixel((0, 0)))
        self.assertEqual((29, 100, 211), pil_image.getpixel((1, 0)))
        np.testing.assert_array_equal(bgr, original)

    @patch("backend.read_datamatrix", return_value=decoded_result())
    def test_grayscale_intensities_are_preserved(self, read):
        gray = np.array([[0, 31], [127, 255]], dtype=np.uint8)
        for input_array in (gray, gray[:, :, None]):
            with self.subTest(shape=input_array.shape):
                decode_datamatrix_from_array(input_array)
                rgb = np.asarray(read.call_args.args[0])
                np.testing.assert_array_equal(rgb, np.repeat(gray[:, :, None], 3, axis=2))

    @patch("backend.read_datamatrix")
    def test_missing_or_empty_input_does_not_run_reader_or_api(self, read):
        for input_array in (None, np.empty((0, 0, 3), dtype=np.uint8)):
            with self.subTest(input=input_array):
                result = decode_datamatrix_from_array(input_array, use_glm=True)
                self.assertEqual("failed", result["status"])
                self.assertEqual([], result["barcodes"])
                self.assertEqual([], result["debug_variants"])
        read.assert_not_called()

    @patch("backend.read_datamatrix", return_value=decoded_result())
    def test_result_preserves_legacy_fields_and_hit_engine(self, read):
        result = decode_datamatrix_from_array(np.zeros((20, 20, 3), dtype=np.uint8))
        self.assertEqual(
            {"barcodes", "status", "debug_variants", "preprocessing_engine", "decoder_engine"},
            set(result),
        )
        self.assertEqual("success", result["status"])
        self.assertEqual("001234567890", result["barcodes"][0]["text"])
        self.assertEqual("blue/blur-3", result["barcodes"][0]["preprocessing_method"])
        self.assertEqual("zxing-cpp/LocalAverage", result["barcodes"][0]["engine"])
        self.assertEqual(["blue/blur-3"], result["debug_variants"])
        self.assertFalse(read.call_args.args[1].use_glm)

    @patch("backend.read_datamatrix", return_value=decoded_result(assistance={"status": "error"}))
    def test_glm_selection_and_api_diagnostics_are_forwarded(self, read):
        result = decode_datamatrix_from_array(
            np.zeros((20, 20, 3), dtype=np.uint8), use_glm=True, max_seconds=3, mode="fast"
        )
        options = read.call_args.args[1]
        self.assertTrue(options.use_glm)
        self.assertEqual(3, options.max_seconds)
        self.assertEqual("fast", options.mode)
        self.assertEqual({"status": "error"}, result["assistance"])

    @patch("backend.read_datamatrix")
    def test_unsupported_inputs_raise_before_decoder_runs(self, read):
        for input_array in (
            np.ones((20, 20), dtype=np.float32),
            np.zeros((20, 20, 4), dtype=np.uint8),
            np.zeros((20,), dtype=np.uint8),
            [[0]],
        ):
            with self.subTest(input_type=type(input_array)), self.assertRaises(ValueError):
                decode_datamatrix_from_array(input_array)
        read.assert_not_called()

    def test_generated_datamatrix_round_trip_through_bgr_and_grayscale(self):
        expected = "001234567890"
        code = zxingcpp.create_barcode(expected, zxingcpp.BarcodeFormat.DataMatrix)
        pixels = np.asarray(zxingcpp.write_barcode_to_image(code, scale=8))
        for input_array in (pixels, cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)):
            with self.subTest(shape=input_array.shape):
                result = decode_datamatrix_from_array(input_array, mode="fast", max_seconds=1)
                self.assertEqual("success", result["status"])
                self.assertEqual([expected], [hit["text"] for hit in result["barcodes"]])
                self.assertTrue(result["barcodes"][0]["engine"].startswith("zxing-cpp/"))


if __name__ == "__main__":
    unittest.main()
