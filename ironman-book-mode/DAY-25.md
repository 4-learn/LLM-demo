# Day 25｜把網頁轉成可供 LLM 閱讀的文字

> **定位**：理解網頁擷取與內容轉換的用途及限制；確認來源合法性、內容品質與外部服務資料邊界。

## 學習目標
- 說明網頁轉 Markdown 是內容抽取，不等於摘要或事實驗證。
- 檢查來源、擷取結果與失敗狀態。
- 知道第三方轉換服務可能收到目標網址或頁面內容。

## 概念
網頁常含導覽列、廣告與正文。Markdown 是以純文字標記標題、清單和連結的格式；抽取工具可將可讀內容轉成較簡潔文字，供 LLM（大型語言模型）搜尋或處理。抽取可能漏掉表格、動態內容或圖片文字，也不能代表有權擷取或重用該資料。API 是程式依規格呼叫外部服務的介面；呼叫前確認頁面網址或內容會送至哪個服務。

## 動手做：離線抽取標題與正文

用標準函式庫解析已取得的 HTML 字串，不連線抓網站：

```python
from html.parser import HTMLParser

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.keep = False
    def handle_starttag(self, tag, attrs):
        if tag in ("title", "p", "h1"):
            self.keep = True
    def handle_endtag(self, tag):
        if tag in ("title", "p", "h1"):
            self.keep = False
    def handle_data(self, data):
        if self.keep and data.strip():
            self.parts.append(data.strip())

sample = "<html><title>公告</title><nav>導覽</nav><h1>活動通知</h1><p>週五舉辦講座。</p></html>"
parser = TextExtractor()
parser.feed(sample)
print("\\n".join(parser.parts))
```

**實際輸出：**
```text
公告
活動通知
週五舉辦講座。
```

這只是小型示意抽取器，不是完整 HTML 轉 Markdown 工具；仍需人工對照來源，動態內容也不會由此取得。

## 練習
挑出一段抽取文字，標記其原始網址和擷取日期。指出任何遺失的上下文，說明若把片段直接交給模型可能造成什麼誤解。

## 自我檢查
1. 網頁轉 Markdown 後，為何仍要回到來源核對？
2. 使用外部轉換 API 前要確認哪些隱私及服務條款？

## 原文與版本註記
[原文 Day 25：LLM 時代下的爬蟲 (1)：利用 Firecrawl 或 Jina AI 把網頁變成 LLM 讀得懂的 Markdown 格式](https://ithelp.ithome.com.tw/articles/10365557?sc=rss.iron)

Firecrawl、Jina Reader、Dify 的方案、API、介面、額度與費用可能變動；授課前查官方文件。僅處理有權存取的頁面並遵守網站條款，不繞過登入、驗證、robots 或其他存取限制。
