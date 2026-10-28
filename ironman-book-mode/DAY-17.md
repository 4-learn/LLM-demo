# Day 17｜知識庫 RAG 與 Notion：資料來源要可追溯

> **主題：** RAG 先從知識來源取回相關內容，再讓模型根據檢索結果回答。接入 Notion 等外部來源前，先處理授權、同步與來源追蹤。

RAG（檢索增強生成）先找相關資料再組成回答；Embedding（嵌入向量）是文字的數值表示，可用於相似度檢索。本章用關鍵字示範，不連接 Dify 或 Notion；Notion 是文件與資料庫協作服務，接入前需限定資料權限。

## 動手做：保留文件來源

```python
documents = [
    {"id": "KB-01", "source": "退貨政策", "text": "商品到貨七日內可申請退貨。"},
    {"id": "KB-02", "source": "配送說明", "text": "訂單成立後可至會員頁查詢配送狀態。"},
]
query = "退貨"
hits = [doc for doc in documents if query in doc["text"]]
for doc in hits:
    print(f"{doc['id']}｜{doc['source']}｜{doc['text']}")
```

預期輸出：

```text
KB-01｜退貨政策｜商品到貨七日內可申請退貨。
```

這是簡化的字串檢索，不是 embedding 或完整 RAG。任何回答都應保留可回查的文件識別資訊；沒有找到相關證據時應說明查無資料，而不是自行補答案。連接 Notion 或知識庫服務時，只授予必要頁面權限，並確認更新/刪除是否同步。

## 小練習

新增一條配送文件，搜尋「配送」，列出命中文件 id。再搜尋不存在的主題，讓程式輸出「查無資料」。

## 原文與版本

原文：[Day 17｜Dify 入門 (3)：使用知識庫來做 RAG 並結合 Notion database](https://ithelp.ithome.com.tw/articles/10361713?sc=rss.iron)

Dify 知識庫與 Notion connector 的設定、權限和同步行為可能隨版本變更。本例僅用合成資料作本地示範；上課前核對服務官方文件及資料授權，不把私人或機密頁面送入未核准的服務。
