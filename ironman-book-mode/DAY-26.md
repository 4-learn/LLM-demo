# Day 26｜LLM 輔助爬蟲：代理不是存取許可

> **定位**：了解 LLM/Agent 如何協助規劃資料擷取步驟，同時把網站允許範圍、停止條件與資料驗證放在控制流程中。

## 學習目標
- 區分「能自動化瀏覽」與「獲准存取」兩件事。
- 設計唯讀、限量且可停止的擷取任務。
- 對擷取資料保留來源並檢查抽取錯誤。

## 概念
LLM（大型語言模型）可處理及生成文字；Agent（代理流程）則在受限工具範圍內依觀察選擇下一步。API 是程式呼叫外部服務的介面。Agent 可以把自然語言目標轉成工具操作，但不能替代網站條款、robots 規則、存取控制或使用者授權。當遇到登入、CAPTCHA、403、速率限制或明確拒絕時，應停止；不得用代理繞過。

## 活動：安全爬取前檢查表

對一個教師核准的公開測試頁逐項回答：
1. 網站是否允許自動存取？
2. 是否含個資、登入內容或受限資料？
3. 請求頻率及頁數上限是多少？
4. 遇到拒絕或驗證時如何停止？
5. 每段結果如何記錄來源 URL 與擷取時間？

本章不執行真實網站爬取；實作工具須先經教師審核，並採人工設定的網域 allowlist、請求上限及逾時。

## 動手做：安全檢查合成頁面

下面只處理程式內的 HTML 文字，不連線抓取網站；遇空白正文就明確回報失敗：

```python
from html.parser import HTMLParser

class Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_p = False
        self.text = []
    def handle_starttag(self, tag, attrs):
        self.in_p = tag == "p"
    def handle_endtag(self, tag):
        if tag == "p":
            self.in_p = False
    def handle_data(self, data):
        if self.in_p and data.strip():
            self.text.append(data.strip())

def extract(html):
    parser = Paragraphs()
    parser.feed(html)
    if not parser.text:
        return "失敗：沒有可用正文"
    return "成功：" + " ".join(parser.text)

for page in ("<nav>Menu</nav><p>公開公告</p>", "<nav>Menu</nav>"):
    print(extract(page))
```

**實際輸出：**
```text
成功：公開公告
失敗：沒有可用正文
```

這是離線解析示意，不會自動驗證權限或網站條款；真實擷取前仍須取得授權並設定網域白名單、請求上限與停止條件。

## 自我檢查
1. Agent 能打開頁面，是否代表有權存取？
2. 什麼情況下必須停止而不是改用另一個工具？

## 原文與版本註記
[原文 Day 26：LLM 時代下的爬蟲 (2)：透過 LLM + Agent 的方式來聰明的爬蟲吧！](https://ithelp.ithome.com.tw/articles/10365779?sc=rss.iron)

ScrapeGraphAI、Skyvern 及其支援能力、API 和費用可能變更；本章以概念和安全規範為主，不保證原文程式在目前版本可執行。不得繞過網站存取控制或防護措施。
