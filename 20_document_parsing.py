"""Extract ticket evidence from page text; PDF support is limited to pypdf text layer.

This is a small teaching example, not a general document parser. It does not
understand layout/tables or perform OCR.
"""
import re
from pathlib import Path

PAGES = [
    (1, ["維修紀錄", "工單編號：WO-2048", "日期：2026-04-02"]),
    (2, ["處理結果：已完成"]),
]


def extract_ticket(pages):
    if not isinstance(pages, (list, tuple)):
        raise ValueError("pages 必須是頁面序列")
    candidates = []
    for page in pages:
        if not isinstance(page, (list, tuple)) or len(page) != 2:
            raise ValueError("每頁必須是 (page_no, lines)")
        page_no, lines = page
        if type(page_no) is not int or page_no < 1:
            raise ValueError("page_no 必須是正整數")
        if not isinstance(lines, (list, tuple)) or not all(isinstance(line, str) for line in lines):
            raise ValueError("lines 必須是字串序列")
        for line in lines:
            match = re.fullmatch(r"工單編號：\s*([A-Z]{2}-\d{4})", line)
            if match:
                candidates.append({"value": match.group(1), "page": page_no, "evidence": line})
    if not candidates:
        return {"status": "missing", "value": None, "evidence": []}
    if len(candidates) != 1:
        return {"status": "ambiguous", "value": None, "evidence": candidates}
    return {"status": "ok", **candidates[0]}


def extract_pdf_pages(path):
    """Return [(1-based page number, [text lines])]; requires optional pypdf.

    Empty text is kept as an empty page so callers can route it to OCR. Encrypted,
    malformed, image-only and layout-heavy PDFs require separate handling.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF 文字抽取需要可選依賴 pypdf：pip install pypdf") from exc
    reader = PdfReader(str(Path(path)))
    pages = []
    for page_no, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        pages.append((page_no, text.splitlines()))
    return pages


def route_pages_for_review(pages, ocr):
    """Keep text-layer pages; send empty pages to an injected OCR callable.

    `ocr(page_no)` is only a seam for a provider/fixture. No OCR engine is
    bundled and OCR output must be reviewed against the original page image.
    """
    routed = []
    for page_no, lines in pages:
        if any(line.strip() for line in lines):
            routed.append({"page": page_no, "source": "text_layer", "lines": list(lines), "needs_review": False})
        else:
            ocr_lines = ocr(page_no)
            routed.append({"page": page_no, "source": "ocr_fixture", "lines": list(ocr_lines), "needs_review": True})
    return routed


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf_pages = extract_pdf_pages(sys.argv[1])
        routed = route_pages_for_review(pdf_pages, lambda page_no: [f"OCR stub: page {page_no} requires human review"] )
        text_pages = [(item["page"], item["lines"]) for item in routed]
        print({"pages": routed, "ticket": extract_ticket(text_pages)})
    else:
        print(extract_ticket(PAGES))
        print(extract_ticket([(1, ["維修紀錄", "日期：2026-04-02"])]))
