# Day 28｜微調入門：準備資料、訓練與評估

> **定位**：理解監督式微調的資料流程與風險，使用合成小型 JSONL 範例認識格式；實際提交訓練前需確認當前供應商支援情況和費用。

## 學習目標
- 說明微調是用示例調整模型行為，不等於補上可即時更新的知識庫。
- 辨識 JSONL 對話訓練資料的基本結構。
- 列出訓練前資料清理、授權、隱私與評估要求。

## 概念
微調（fine-tuning）是用訓練範例進一步調整模型行為；JSON 是常見的鍵值資料格式，JSONL 則是一行一筆 JSON 的文字檔。Schema 是描述欄位及型別的資料規格，各服務訓練 API 要求可能不同。微調可讓模型更貼近特定任務的格式或風格，但不保證事實正確，也不能取代檢索最新資料。需準備高品質、獲准使用且不含不必要個資的訓練範例，並以獨立測試資料比較微調前後效果。

## JSONL 示意資料

以下只作格式說明，不表示此內容符合任何當前 API 的訓練 schema：

```json
{"messages":[{"role":"user","content":"請把這句話整理成分類：產品故障"},{"role":"assistant","content":"故障回報"}]}
{"messages":[{"role":"user","content":"請把這句話整理成分類：詢問營業時間"},{"role":"assistant","content":"一般詢問"}]}
```

每一行是一筆 JSON 記錄。訓練前應檢查 JSON 可解析、欄位格式正確、資料權利清楚、重複/洩漏已處理，且保留獨立評估集。

## 動手做：逐行驗證 JSONL 結構

```python
import json
from io import StringIO

jsonl = '''{"messages":[{"role":"user","content":"請分類：故障"},{"role":"assistant","content":"故障回報"}]}
{"messages":[{"role":"user","content":"請分類：營業時間"},{"role":"assistant","content":"一般詢問"}]}'''

for line_no, line in enumerate(StringIO(jsonl), start=1):
    row = json.loads(line)
    valid = isinstance(row.get("messages"), list) and all(
        isinstance(m, dict) and m.get("role") in {"user", "assistant"}
        and isinstance(m.get("content"), str)
        for m in row["messages"]
    )
    print(f"第 {line_no} 行：{'OK' if valid else '格式錯誤'}")
```

**實際輸出：**
```text
第 1 行：OK
第 2 行：OK
```

這只驗證本例採用的簡化格式，不代表符合任何供應商的訓練 schema。

## 練習
設計三筆合成範例和兩筆獨立測試案例。寫下預期評估指標；不呼叫任何付費 API、不上傳真實資料。

## 自我檢查
1. 微調和 RAG 分別適合處理什麼需求？
2. 為什麼不能用訓練資料本身作唯一評估？

## 原文與版本註記
[原文 Day 28：讓 gpt-4o-mini 變成你的女僕：用 JSONL 微調 gpt-4o-mini，並在 Open WebUI 中使用](https://ithelp.ithome.com.tw/articles/10366902?sc=rss.iron)

原文以 2024 年 OpenAI 模型、微調 API 和 Open WebUI Pipeline 為背景。模型支援、資料 schema、價格、排隊時間與介面都可能已變；提交資料或付費前須查當前官方文件及帳務。不要上傳私人對話、個資或機密資料。
