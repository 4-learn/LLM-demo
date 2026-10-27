"""Tests for the generated PDF fixture and real pypdf text extraction."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).with_name("demo_document_parsing.py")
spec = importlib.util.spec_from_file_location("demo_document_parsing", MODULE)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
LESSON_FILE = MODULE.parents[1] / "20_document_parsing.py"
lesson_spec = importlib.util.spec_from_file_location("lesson_document_parsing", LESSON_FILE)
lesson = importlib.util.module_from_spec(lesson_spec)
lesson_spec.loader.exec_module(lesson)


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

    def test_pypdf_reads_fixture_and_preserves_page_number(self):
        try:
            import pypdf  # noqa: F401
        except ImportError:
            self.skipTest("optional dependency pypdf is not installed")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.pdf"
            demo.make_pdf(path)
            pages = lesson.extract_pdf_pages(path)
            self.assertEqual([number for number, _ in pages], [1])
            joined = "\n".join(pages[0][1])
            self.assertIn("Work order WO-2048", joined)
            self.assertIn("Asset: Pump A", joined)

    def test_missing_optional_dependency_has_clear_error(self):
        original = __import__
        def without_pypdf(name, *args, **kwargs):
            if name == "pypdf":
                raise ImportError("simulated missing optional dependency")
            return original(name, *args, **kwargs)
        import builtins
        builtins.__import__ = without_pypdf
        try:
            with self.assertRaisesRegex(RuntimeError, "pip install pypdf"):
                lesson.extract_pdf_pages("unused.pdf")
        finally:
            builtins.__import__ = original

    def test_empty_text_layer_routes_to_injected_ocr_and_requires_review(self):
        calls = []
        def fake_ocr(page_no):
            calls.append(page_no)
            return ["工單編號：WO-2048"]
        result = lesson.route_pages_for_review([(1, ["文字層"]), (2, [])], fake_ocr)
        self.assertEqual(calls, [2])
        self.assertEqual(result[0]["source"], "text_layer")
        self.assertFalse(result[0]["needs_review"])
        self.assertEqual(result[1]["source"], "ocr_fixture")
        self.assertTrue(result[1]["needs_review"])
        extracted = lesson.extract_ticket([(result[1]["page"], result[1]["lines"])])
        self.assertEqual(extracted["page"], 2)


if __name__ == "__main__":
    unittest.main()
