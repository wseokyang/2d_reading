from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image
import zxingcpp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grid_recovery import recover_grid


class GridRecoveryTests(unittest.TestCase):
    def test_damaged_and_colored_images_decode_from_observed_pixels(self):
        references = json.loads((ROOT / "test_cases/manifest.json").read_text())["cases"]
        for name in (
            "{E7CA298A-DC03-45A1-BABC-BEF2C7D89DFB}.png",
            "{0783A71B-ACC9-4179-A3AF-25FC9042B4FC}.png",
            "user_yellow_mark_blue_metal.png",
            "user_yellow_mark_dark_metal.png",
        ):
            with self.subTest(file=name), Image.open(ROOT / "test_cases" / name) as image:
                result = recover_grid(image, max_seconds=3)
                self.assertIsNotNone(result)
                self.assertEqual(result.text, references[name]["expected_text"])
                self.assertGreaterEqual(result.finder_score, 0.90)
                # Recheck the actual returned, unmodified sampled matrix.
                decoded = zxingcpp.read_barcode(
                    result.rendered,
                    formats=zxingcpp.BarcodeFormat.DataMatrix,
                    is_pure=True,
                    binarizer=zxingcpp.Binarizer.BoolCast,
                )
                self.assertTrue(decoded.valid)
                self.assertEqual(decoded.bytes, result.raw_bytes)

    def test_unseen_alphanumeric_payload_all_rotations_and_polarities(self):
        payload = "A12B34C56D78"
        barcode = zxingcpp.create_barcode(payload, zxingcpp.BarcodeFormat.DataMatrix)
        symbol = np.asarray(zxingcpp.write_barcode_to_image(barcode, scale=5, add_quiet_zones=False))
        pixels = np.pad(symbol, 25, constant_values=255)
        for rotation in range(4):
            for inverted in (False, True):
                with self.subTest(rotation=rotation, inverted=inverted):
                    sample = np.rot90(pixels, rotation)
                    if inverted:
                        sample = 255 - sample
                    result = recover_grid(Image.fromarray(sample), max_seconds=2)
                    self.assertIsNotNone(result)
                    self.assertEqual(result.text, payload)

    def test_blank_noise_and_finder_without_valid_data_are_rejected(self):
        rng = np.random.default_rng(37)
        random_grid = rng.integers(0, 2, (14, 14), dtype=np.uint8)
        random_grid[:, 0] = 1
        random_grid[-1, :] = 1
        random_grid[0, :] = np.arange(14) % 2 == 0
        random_grid[:, -1] = np.arange(14) % 2 == 1
        invalid_symbol = np.repeat(np.repeat(255 - random_grid * 255, 5, axis=0), 5, axis=1)
        cases = (
            np.full((132, 143, 3), 128, dtype=np.uint8),
            rng.integers(0, 256, (160, 160, 3), dtype=np.uint8),
            np.pad(invalid_symbol, 25, constant_values=255),
        )
        for index, pixels in enumerate(cases):
            with self.subTest(index=index):
                self.assertIsNone(recover_grid(Image.fromarray(pixels), max_seconds=1))

    def test_zero_budget_skips_recovery(self):
        self.assertIsNone(recover_grid(Image.new("RGB", (100, 100)), max_seconds=0))


if __name__ == "__main__":
    unittest.main()
