from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_cases import compare_payloads, load_manifest


class BenchmarkVerificationTests(unittest.TestCase):
    def test_exact_comparison_keeps_zeroes_and_rejects_extra_payloads(self):
        self.assertEqual(compare_payloads(True, ["00123"], "00123"), (True, "match"))
        self.assertEqual(compare_payloads(True, ["123"], "00123"), (False, "mismatch"))
        self.assertEqual(compare_payloads(True, ["00123", "other"], "00123"), (False, "mismatch"))
        self.assertEqual(compare_payloads(False, [], "00123"), (False, "unread"))

    def test_missing_reference_does_not_count_as_verified(self):
        self.assertEqual(compare_payloads(True, ["00123"], None), (None, "unverified"))
        self.assertEqual(compare_payloads(False, [], None), (None, "unread"))

    def test_manifest_rejects_numeric_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"schema_version": 1, "cases": {"image.png": {"expected_text": 123}}}))
            with self.assertRaisesRegex(ValueError, "nonempty string or null"):
                load_manifest(path)

    def test_worker_hard_timeout_is_recorded_and_run_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "timeout.json"
            completed = subprocess.run([
                sys.executable, str(ROOT / "scripts/benchmark_cases.py"),
                "--only", "user_yellow_mark_blue_metal.png", "--case-timeout", "0.001",
                "--json", str(report), "--csv", str(report.with_suffix(".csv")),
            ], capture_output=True, text=True, timeout=15)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            rows = json.loads(report.read_text())
            self.assertFalse(rows[0]["ok"])
            self.assertFalse(rows[0]["verified"])
            self.assertIn("Hard per-case timeout", rows[0]["error"])

    def test_wrong_reference_is_a_failure_even_when_decoder_succeeds(self):
        name = "{22A13104-DC65-43D3-8835-CC1B9A259244}.png"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "cases": {name: {"expected_text": "WRONG-REFERENCE"}}}))
            report = directory / "mismatch.json"
            completed = subprocess.run([
                sys.executable, str(ROOT / "scripts/benchmark_cases.py"),
                "--only", name, "--manifest", str(manifest), "--max-seconds", "2",
                "--json", str(report), "--csv", str(report.with_suffix(".csv")),
            ], capture_output=True, text=True, timeout=15)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            rows = json.loads(report.read_text())
            self.assertTrue(rows[0]["ok"])
            self.assertFalse(rows[0]["verified"])
            self.assertEqual(rows[0]["verification_status"], "mismatch")


if __name__ == "__main__":
    unittest.main()
