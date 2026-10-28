# Day 18｜Dify Chatflow 與 Workflow：依任務安排節點

> **主題：** 對話型流程與任務型流程可有不同入口與結束方式；先用一般 Python 理解「輸入—處理—回應」，再映射到圖形化流程工具。

Workflow（工作流程）是預先安排節點與轉移的任務流程；Chatflow 是帶有對話輸入/回應的互動流程。UI（使用者介面）是人操作系統的畫面或入口；本例只用 Python 模擬，不依賴 Dify 當前 UI。

## 動手做：拆分處理步驟

```python
def normalize(text):
    return " ".join(text.strip().split())

def route(text):
    clean = normalize(text)
    if not clean:
        return "請輸入問題"
    if "退貨" in clean:
        return "退貨流程"
    return "一般問答流程"

for message in ["  想辦理退貨  ", "你好", "   "]:
    print(f"{message!r} -> {route(message)}")
```

預期輸出：

```text
'  想辦理退貨  ' -> 退貨流程
'你好' -> 一般問答流程
'   ' -> 請輸入問題
```

Chatflow 常用於互動式對話，Workflow 可用於明確的任務步驟；具體功能與命名須按實際產品版本確認。無論用哪種流程，節點間都應定義資料格式、錯誤路徑和結束條件。Day 4 的加權混合 Mermaid 圖屬課程增補，非原文內容，另見延伸頁。

## 小練習

畫出上述三條路徑的流程草圖，替「退貨流程」加上人工確認節點。說明空輸入為何不能直接送進模型。

## 原文與版本

原文：[Day 18｜Dify 工作流 (1)：工作流基本介紹 (Chatflow, Workflow)](https://ithelp.ithome.com.tw/articles/10361999?sc=rss.iron)

原文 UI、節點名稱和功能對應發文時的 Dify 版本。授課前請核對當前官方文件；本例僅示範流程拆解，未宣稱特定節點在目前版本的位置或行為。
