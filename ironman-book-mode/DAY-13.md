# Day 13｜圖形化 LLM 工具：理解流程抽象與取捨

> **主題：** Dify、Langflow、Flowise、Coze 等圖形化工具，讓使用者以節點組成應用流程；選工具前仍要理解節點背後的資料與控制流程。

Dify、Langflow 等圖形化開發介面可將步驟建成節點式 Workflow（工作流程），讓資料按預先設計的路徑移動；fallback（後備路徑）是在無法分類或執行失敗時採用的替代處理。

## 動手做：用 Python 描述一條流程

```python
def classify(question):
    if "退貨" in question:
        return "退貨說明"
    if "訂單" in question:
        return "訂單查詢"
    return "人工確認"

for question in ["退貨要幾天？", "我的訂單到哪了？", "嗨"]:
    print(f"{question} -> {classify(question)}")
```

預期輸出：

```text
退貨要幾天？ -> 退貨說明
我的訂單到哪了？ -> 訂單查詢
嗨 -> 人工確認
```

將此流程畫成節點圖：輸入 → 分類 → 對應路徑。圖形工具可以降低串接步驟的門檻，但不會自動保證分類正確、權限安全或部署設定妥當。每個節點都應有可觀察的輸入、輸出和失敗處理。

## 小練習

增加「取消訂單」路徑，再設計一個無法分類時的 fallback。列出至少一個同義問法，檢查關鍵字規則的限制。

## 原文與版本

原文：[Day 13｜告別繁瑣程式碼：初探 Dify / Langflow 等圖形化介面工具來開發 LLM 應用](https://ithelp.ithome.com.tw/articles/10359521?sc=rss.iron)

原文比較 Dify、Langflow、Flowise、Coze 等工具，並以當時介面說明。產品功能、授權與部署方式會更新；本 Python 練習不依賴特定平台。若使用工具，請授課前核對其官方文件與授權條款。
