# Day 8｜RAG：先找資料，再請模型回答

> **定位**：用簡單 Python 做字詞檢索，理解 RAG 的兩階段；另以 NVIDIA Build embedding API 為選做，實際印出向量。

## 學習目標
- 分辨檢索與生成兩個階段。
- 保留檢索來源，避免把相似度當成事實。
- 知道真實 embedding 呼叫需設定服務憑證。

## 概念

RAG（Retrieval-Augmented Generation，檢索增強生成）先從資料找相關內容，再把內容作為上下文交給 LLM；它可提供證據，但不保證答案正確。Embedding（嵌入向量）是把文字轉為數值向量，使程式可依向量相似度找候選內容；API（程式呼叫服務的介面）回傳的向量應由實際服務回應，不可杜撰。OpenAI-compatible（OpenAI 相容）表示部分 API 請求格式相似，不保證不同供應商支援相同模型或功能。

## 延伸閱讀：兩跳關係圖

E04 的二跳關係檢索指從起點經一個中間節點到目標節點；此為課程補充概念，不是本篇 RAG 原文操作。

```mermaid
flowchart LR
    S[問題中的起點] -->|第一跳：關係| M[中間實體]
    M -->|第二跳：關係| T[答案候選]
```

圖中的箭頭代表關係連結，不代表資料已查證；每一跳都需保留來源與證據。

## 動手做：本地關鍵字檢索

```python
docs = ["蘋果是水果。", "茶可以熱飲或冷飲。", "香蕉通常是黃色。"]
query = "蘋果"
hits = [doc for doc in docs if query in doc]
print("候選資料：", hits)
```

**實際輸出（以 Python 執行上述程式）：**

```text
候選資料： ['蘋果是水果。']
```

關鍵字檢索要求字面相同；embedding 檢索則把文字映射成向量，再依向量相似度找候選。相似不等於事實，回答仍應附來源供查核。

## 選做：NVIDIA Build embedding

NVIDIA Build 提供可選的免費使用方案，但啟用情況、限額及條款可能變動；執行前到官方模型頁查證是否仍免費。依該頁選定可用的 embedding 模型與 API URL。安裝 `requests` 後，先在本機環境變數設定 `NVIDIA_API_KEY`；不要把 key 放入程式或筆記。本文不保證模型、URL、額度或價格現況。合成文字範例：

```python
import os
import requests

url = os.environ["NVIDIA_EMBEDDINGS_URL"]  # 使用 NVIDIA Build 當前模型頁提供的完整 endpoint
headers = {"Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}"}
payload = {
    "input": ["退貨需要在收到商品七天內申請。"],
    "input_type": "passage",
}
response = requests.post(url, headers=headers, json=payload, timeout=30)
response.raise_for_status()
data = response.json()
vector = data["data"][0]["embedding"]
print(vector)
```

`input` 是待轉換文字；`input_type` 指明用途，文件內容用 `passage`、使用者問題用 `query`（前提是所選模型文件支援這些值）。程式印出 API 真正回傳的完整向量，不會用固定假資料取代服務回應。

**實際輸出：需金鑰，輸出依實際執行為準。** 本環境未設定 `NVIDIA_API_KEY`，因此沒有執行真實 API 呼叫，也不提供捏造的向量。執行前確認方案、額度與資料政策；金鑰不應貼入對話，只傳送合成文字。

**查詢文字版本**：把 `input` 換成使用者問題，並將 `input_type` 改成 `query`；文件段落使用 `passage`。請依選用模型的當前文件確認可接受值。

## 版本提醒

原文描述 RAG、向量資料庫及模型例子為 2024 年脈絡。embedding 模型、端點、參數、向量維度與服務條款會變；不猜測現行型號、免費額度或價格。真實 LLM 生成延續 OpenAI 相容 chat API 格式，仍須核對供應商文件。

## 原文

[原文 Day 8：RAG 是什麼？如何應用於我們的 LLM 應用？](https://ithelp.ithome.com.tw/articles/10355335?sc=rss.iron)
