# 20｜文件證據抽取（文字型 PDF、OCR 分流與人工覆核示例）

> **正文改版：2026-10-28｜對應新版 LLM 第 20 節：文件解析入門。**本節用 `pypdf` 對合成文字型 PDF 做實際文字抽取並保留 1-based 頁碼；不宣稱通用 PDF parser。空文字頁可交給注入的 OCR stub 展示分流，但未包含 OCR 引擎；OCR 結果標記為需人工覆核。版面表格、掃描品質與真實文件準確率仍未驗證。

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先備：能讀懂 Python 字典、字串與條件判斷；不需模型或網路。建議 50 分鐘：概念 30 分鐘（頁面文字與來源 10、抽取 10、OCR/覆核界線 10），Workshop 20 分鐘。標準函式庫測試可直接執行；實際讀取 PDF 文字需安裝可選依賴 `python3 -m pip install pypdf`。程式遇到缺少 `pypdf` 時會給明確安裝錯誤；單元測試會對該依賴測試標記 skip。

## 一、頁面不是一串欄位

真實文件可能有標題、表格、頁首頁尾、欄位標籤、跨頁內容與文字區塊。解析的第一件事是把可觀察的原始內容整理成頁面證據，保留頁碼、區塊順序及原文；若只保留最後的「日期」值，就無法分辨它來自表頭、簽核欄或另一筆紀錄。實務上 OCR 只是一種取得文字的方式，不能代替後續結構判讀，也可能把 8 誤成 3、漏掉欄位或打亂閱讀順序。文件處理宜分為取得原始頁面、切出區塊、辨識欄位、驗證型別、保存證據等步驟；各步驟分開記錄失敗，可避免把辨識錯字誤當成欄位語意錯誤。欄位與頁碼應可回指原始頁面，才方便人工抽查。

## 二、抽取結果須帶來源

本例將合成頁面表達為頁碼與文字行。每個欄位透過明確標籤附近的文字抽取，輸出值同時帶頁碼、原文證據和可信狀態。這是可追查的抽取，不是任意猜值：若欄位缺失、多個候選或證據格式不符，就回報需人工覆核，不以第一個看似合理的值冒充正確答案。

## 三、文字相同不代表語意相同

「日期：2026-04-02」可作為日期候選，但同頁也可能有「修訂日期」或頁尾列印日期。欄位名稱、所在區域、格式驗證及文件版本都會影響判讀。欄位抽取應依資料契約逐項定義，將缺失、歧義、驗證失敗分開記錄；不要把空值改成預設值，也不要用生成模型補齊不存在的證據。

## 四、可執行 Demo

同源 Demo 保留了手動輸入頁面行的欄位抽取案例；另外現在有可執行 PDF 路徑，會用 `pypdf` 逐頁抽取文字並保留 1-based 頁碼。執行：

```bash
python3 tools/demo_document_parsing.py /tmp/work-order.pdf
python3 20_document_parsing.py /tmp/work-order.pdf
# The CLI supplies an explicitly labelled OCR stub for pages with no text layer.
python3 -m unittest tools.test_demo_document_parsing -v
```

若缺少 `pypdf`，第二個命令會以明確錯誤提示安裝方式；測試中的 parser integration case 會標記 skip，其餘測試仍可跑。解析僅限 PDF 文字層可抽取的內容，並非通用 parser，不會推斷複雜閱讀順序或表格結構。空文字頁可以送到注入的 OCR callable；本教材測試只用 stub 回傳固定行，`source=ocr_fixture` 且 `needs_review=True`，**不是 OCR 結果，也不是準確率證據**。真實 OCR 要保留原頁影像、辨識引擎/版本/設定與逐頁輸出，核對欄位、頁碼、否定詞、數字與表格後才可採用；低信心、缺頁、歧義或無法回指原文時須人工覆核，不可讓模型補造證據。

<!-- demo: 20_document_parsing.py; full file is shown here, PDF CLI above -->
```python
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
```

關鍵輸出（本次實際執行；標準函式庫、合成 fixtures）：

```text
{'status': 'ok', 'value': 'WO-2048', 'page': 1, 'evidence': '工單編號：WO-2048'}
{'status': 'missing', 'value': None, 'evidence': []}
```

## Workshop

20 分鐘：確認文字型 PDF 逐頁保留頁碼；再用注入的固定 OCR stub 處理空文字頁，確認該頁標記 `source=ocr_fixture` 與 `needs_review=True`，明白它不是辨識結果。單筆、缺失與多筆工單案例分別應回傳 `ok`、`missing`、`ambiguous`。

## 教師紀錄與下一步

本次驗證範圍：由標準函式庫產生的單頁文字型 PDF fixture；實際 pypdf parser integration 測試在未安裝 pypdf 的環境標記 skip，缺依賴錯誤提示另有測試。無 OCR 引擎、掃描文件、版面表格或真實文件準確率測試；OCR 僅以注入 stub 示範分流，需人工對照原頁覆核。不得外推為通用 PDF/OCR 能力。尚未實班試教，沒有學生實測結果。
