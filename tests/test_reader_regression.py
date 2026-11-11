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

from reader import ReaderOptions, read_datamatrix, read_image_bytes


class ReaderRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((ROOT / "test_cases/manifest.json").read_text())["cases"]

    def test_user_images_and_recovered_hard_cases_match_exact_references(self):
        names = [
            "user_yellow_mark_blue_metal.png",
            "user_yellow_mark_dark_metal.png",
            "{E7CA298A-DC03-45A1-BABC-BEF2C7D89DFB}.png",
            "{0783A71B-ACC9-4179-A3AF-25FC9042B4FC}.png",
        ]
        for name in names:
            with self.subTest(file=name):
                result = read_datamatrix(
                    read_image_bytes((ROOT / "test_cases" / name).read_bytes()),
                    ReaderOptions(mode="aggressive", max_seconds=30),
                )
                self.assertTrue(result.ok, result.error)
                self.assertEqual({hit.text for hit in result.hits}, {self.cases[name]["expected_text"]})

    def test_generated_unseen_payload_retains_leading_zeroes(self):
        payload = "0011223344556677"
        barcode = zxingcpp.create_barcode(payload, zxingcpp.BarcodeFormat.DataMatrix)
        pixels = np.asarray(zxingcpp.write_barcode_to_image(barcode, scale=8))
        image = Image.fromarray(pixels).convert("RGB")
        for variant in (image, image.rotate(3, expand=True, fillcolor="white")):
            with self.subTest(size=variant.size):
                result = read_datamatrix(variant, ReaderOptions(mode="balanced", max_seconds=5))
                self.assertTrue(result.ok)
                self.assertEqual({hit.text for hit in result.hits}, {payload})

    def test_blank_and_noise_images_do_not_produce_payloads(self):
        rng = np.random.default_rng(20261111)
        samples = {
            "white": np.full((96, 112, 3), 255, dtype=np.uint8),
            "gray": np.full((96, 112, 3), 127, dtype=np.uint8),
            "black": np.zeros((96, 112, 3), dtype=np.uint8),
            "rgb_noise": rng.integers(0, 256, (96, 112, 3), dtype=np.uint8),
            "monochrome_noise": np.repeat(rng.integers(0, 256, (96, 112, 1), dtype=np.uint8), 3, axis=2),
        }
        for name, pixels in samples.items():
            with self.subTest(image=name):
                result = read_datamatrix(
                    Image.fromarray(pixels),
                    ReaderOptions(mode="aggressive", max_seconds=1.5, max_attempts=100, use_libdmtx=False),
                )
                self.assertFalse(result.ok)
                self.assertEqual(result.hits, [])

    def test_manifest_covers_all_fixture_images(self):
        images = {path.name for path in (ROOT / "test_cases").glob("*.png")}
        self.assertEqual(images, set(self.cases))
        self.assertEqual(len(images), 26)
        for name, case in self.cases.items():
            with self.subTest(file=name):
                self.assertIsInstance(case["expected_text"], str)
                self.assertTrue(case["provenance"])

    def test_center_grid_recovery_rejects_blank_and_noise(self):
        # Exercise the new grid search directly: the reader's short negative-case
        # budget may otherwise expire before it reaches this fallback.
        from grid_recovery import recover_grid

        rng = np.random.default_rng(4343)
        for pixels in (
            np.full((100, 120, 3), 127, dtype=np.uint8),
            rng.integers(0, 256, (100, 120, 3), dtype=np.uint8),
        ):
            self.assertIsNone(recover_grid(Image.fromarray(pixels), max_seconds=1.0))


if __name__ == "__main__":
    unittest.main()
