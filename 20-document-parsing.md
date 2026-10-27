# 20｜文件證據抽取（PDF 解析／OCR／版面表格待補）

> **正文改版：2026-10-24｜對應新版 LLM 第 20 節：真實文件解析。**本節以合成頁面文字教授欄位抽取與證據定位，並可產生含可選取文字的 PDF fixture；未使用 PDF parser、未辨識版面表格或 OCR，也不模擬掃描辨識率。完整實際文件流程仍待補。

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先備：能讀懂 Python 字典、字串與條件判斷；不需 OCR、模型或網路。建議 50 分鐘：概念 35 分鐘（頁面結構 10、證據抽取 15、限制與錯誤 10），Workshop 15 分鐘。若要執行 PDF 解析片段，另需教師或學生自行安裝 `pypdf`；核心同源 Demo 仍只依賴標準函式庫。

## 一、頁面不是一串欄位

真實文件可能有標題、表格、頁首頁尾、欄位標籤、跨頁內容與文字區塊。解析的第一件事是把可觀察的原始內容整理成頁面證據，保留頁碼、區塊順序及原文；若只保留最後的「日期」值，就無法分辨它來自表頭、簽核欄或另一筆紀錄。實務上 OCR 只是一種取得文字的方式，不能代替後續結構判讀，也可能把 8 誤成 3、漏掉欄位或打亂閱讀順序。文件處理宜分為取得原始頁面、切出區塊、辨識欄位、驗證型別、保存證據等步驟；各步驟分開記錄失敗，可避免把辨識錯字誤當成欄位語意錯誤。欄位與頁碼應可回指原始頁面，才方便人工抽查。

## 二、抽取結果須帶來源

本例將合成頁面表達為頁碼與文字行。每個欄位透過明確標籤附近的文字抽取，輸出值同時帶頁碼、原文證據和可信狀態。這是可追查的抽取，不是任意猜值：若欄位缺失、多個候選或證據格式不符，就回報需人工覆核，不以第一個看似合理的值冒充正確答案。

## 三、文字相同不代表語意相同

「日期：2026-04-02」可作為日期候選，但同頁也可能有「修訂日期」或頁尾列印日期。欄位名稱、所在區域、格式驗證及文件版本都會影響判讀。欄位抽取應依資料契約逐項定義，將缺失、歧義、驗證失敗分開記錄；不要把空值改成預設值，也不要用生成模型補齊不存在的證據。

## 四、可執行 Demo

同源 Demo 只用合成文字證據，示範「工單編號」欄標籤抽取、來源保留與候選歧義拒答；執行：`python3 20_document_parsing.py`。另有兩段**可選擇性 PDF 操作（不屬同源測試涵蓋）**：先執行 `python3 tools/demo_document_parsing.py /tmp/work-order.pdf` 產生可選取文字的合成 PDF，再以 `pypdf` 讀取頁面文字並交給同一欄位抽取函式。這只驗證文字型 PDF 可抽出文字，不是 OCR、版面表格或任意 PDF 解析；安裝/版本差異需另行記錄。

```python
# 可選操作片段；同源 test_lessons.py 不會執行本區塊。
from pypdf import PdfReader
reader = PdfReader("/tmp/work-order.pdf")
pages = [(i, (page.extract_text() or "").splitlines())
         for i, page in enumerate(reader.pages, 1)]
print(extract_ticket(pages))
```

<!-- demo: 20_document_parsing.py -->
```python
import re

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


if __name__ == "__main__":
    print(extract_ticket(PAGES))
    print(extract_ticket([(1, ["維修紀錄", "日期：2026-04-02"])]))
```

關鍵輸出（本次實際執行；標準函式庫、合成 fixtures）：

```text
{'status': 'ok', 'value': 'WO-2048', 'page': 1, 'evidence': '工單編號：WO-2048'}
{'status': 'missing', 'value': None, 'evidence': []}
```

## Workshop

15 分鐘：前 5 分鐘把第二個頁面增加一行工單編號；接著 5 分鐘再增加另一個不同編號；最後 5 分鐘可選執行 PDF fixture 產生器與 `pypdf` 抽取片段，觀察原文行交給同一抽取函式的結果。參考判讀：單筆應為 `ok`，沒有標籤為 `missing`，多筆不得任選其一，應為 `ambiguous`；PDF 解析片段只適用含可抽取文字的此 fixture。可執行驗收：複製 Demo 後新增 `assert extract_ticket([(3, ["工單編號：WO-3001"])])["value"] == "WO-3001"`、`assert extract_ticket([])["status"] == "missing"`，並確認兩筆候選回傳 `ambiguous` 且 evidence 長度為 2；執行 `python3` 檔案時三項斷言皆通過。

## 教師紀錄與下一步

證據級別：本次實際執行教材內嵌程式，日期 2026-10-24；環境 Linux Ubuntu 20.04 x86_64、Python 3.11.2、standard library；模型不適用；量測方式為直接執行並抄錄 stdout。合成 fixtures 僅測結構化頁面輸入的規則抽取；另有標準函式庫產生之文字型 PDF fixture，可選用 `pypdf` 解析其可選取文字，但不是通用 PDF 抽取驗證。未涵蓋掃描 PDF、OCR、版面配置、表格合併、手寫辨識、文件語言變異或準確率評估；不得外推至真文件。尚未實班試教，沒有學生實測結果。下一節可銜接切塊與證據索引；舊教材連結僅供歷史參照，未宣稱同步。
