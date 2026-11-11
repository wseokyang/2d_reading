"""Exercise the Streamlit entrypoint, including replacing a selected image."""
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    def test_selected_image_change_updates_decoded_detail(self):
        with patch.dict("os.environ", {"GLM_BASE_URL": "", "GLM_MODEL": "", "GLM_API_KEY": "", "ZAI_API_KEY": ""}):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
            self.assertFalse(app.exception)
            self.assertEqual(["업로드", "샘플"], [tab.label for tab in app.tabs])
            for filename, payload in (
                ("user_yellow_mark_blue_metal.png", "0540234986250871"),
                ("user_yellow_mark_dark_metal.png", "0523413046220941"),
            ):
                app.multiselect[0].set_value([filename]).run()
                app.button(key="run-samples-test_cases").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(payload, app.text_area[0].value)
                self.assertTrue(app.text_area[0].disabled)
                self.assertEqual("1", app.metric[1].value)


if __name__ == "__main__":
    unittest.main()
