from __future__ import annotations

import runpy
import sys
import types
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import zxingcpp
from PIL import Image

import org_code
import org_code_2
import reader


class LegacyEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = "0012345678901234"
        code = zxingcpp.create_barcode(cls.text, zxingcpp.BarcodeFormat.DataMatrix)
        cls.gray = np.asarray(zxingcpp.write_barcode_to_image(code, scale=8))
        cls.bgr = cv2.cvtColor(cls.gray, cv2.COLOR_GRAY2BGR)

    def test_original_array_read_name_decodes_with_new_backend(self):
        result = org_code.decode_image_from_array(self.bgr)
        self.assertEqual("success", result["status"])
        self.assertEqual(self.text, result["barcodes"][0]["text"])
        self.assertTrue(result["barcodes"][0]["engine"].startswith("zxing-cpp/"))
        self.assertEqual("adaptive", result["preprocessing_engine"])

    def test_original_list_entrypoint_retains_return_shape(self):
        hits = org_code.decode_multiple_barcodes(self.bgr)
        self.assertEqual([self.text], [hit["text"] for hit in hits])

    @patch("org_code.decode_datamatrix_from_array")
    def test_legacy_selectors_are_rejected_instead_of_ignored(self, decode):
        for kwargs in (
            {"selected_methods": {"original"}},
            {"preprocessor_type": "classic"},
            {"preprocessor_type": "ve2_aggressive"},
            {"decoder_type": "pylibdmtx"},
            {"decoder_type": "zxingcpp"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                org_code.decode_image_from_array(self.bgr, **kwargs)
        decode.assert_not_called()

    @patch("org_code.decode_datamatrix_from_array")
    def test_glm_and_new_search_options_are_forwarded(self, decode):
        decode.return_value = {"status": "failed", "barcodes": []}
        result = org_code.decode_image_from_array(
            self.bgr, use_glm=True, max_seconds=3, mode="balanced"
        )
        self.assertIs(result, decode.return_value)
        self.assertIs(decode.call_args.args[0], self.bgr)
        self.assertEqual({"use_glm": True, "max_seconds": 3, "mode": "balanced"}, decode.call_args.kwargs)

    def test_original_pil_reader_imports_are_the_shared_api(self):
        for name in org_code_2.__all__:
            self.assertIs(getattr(org_code_2, name), getattr(reader, name))
        result = org_code_2.read_datamatrix(Image.fromarray(self.gray))
        self.assertTrue(result.ok)
        self.assertEqual(self.text, result.hits[0].text)

    def test_legacy_streamlit_filename_launches_maintained_app(self):
        app = types.ModuleType("app")
        app.main = Mock()
        with patch.dict(sys.modules, {"app": app}):
            runpy.run_module("org_app", run_name="__main__")
        app.main.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
