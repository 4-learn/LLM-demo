"""Tests for the generated PDF fixture; no third-party parser required."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).with_name("demo_document_parsing.py")
spec = importlib.util.spec_from_file_location("demo_document_parsing", MODULE)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class PdfFixtureTests(unittest.TestCase):
    def test_fixture_has_valid_pdf_structure_and_expected_text(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.pdf"
            size = demo.make_pdf(path)
            data = path.read_bytes()
            self.assertEqual(size, len(data))
            self.assertTrue(data.startswith(b"%PDF-1.4"))
            self.assertIn(b"xref\n", data)
            self.assertIn(b"startxref\n", data)
            self.assertIn(b"Work order WO-2048", data)
            self.assertIn(b"Asset: Pump A", data)


if __name__ == "__main__":
    unittest.main()
