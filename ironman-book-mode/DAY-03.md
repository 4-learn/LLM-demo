# Day 3｜第一次呼叫 LLM API

> **定位**：認識 API request/response，先用標準函式庫建立安全的離線示意；真實呼叫是選做，且需自行確認帳務與權限。

## 學習目標

- 說明 API、JSON、API key 的用途。
- 看懂 OpenAI 相容 chat completion 的基本欄位。
- 知道金鑰與外送資料的風險。

## 概念

API（應用程式介面）是程式依規格向服務提出請求並接收回應的方式。JSON 是常見的文字資料格式；SDK 是封裝 API 呼叫的程式套件；endpoint 是服務接收請求的網址。OpenAI 相容表示部分請求格式相似，不代表模型、功能、價格或資料政策相同。

## 動手做：離線 request/response

```python
import json
request = {
    "model": "MODEL_ID_FROM_PROVIDER_DOCS",
    "messages": [{"role": "user", "content": "請用一句話問候。"}],
}
print(json.dumps(request, ensure_ascii=False, indent=2))
```

**實際輸出（以 Python 執行上述程式）：**

```text
{
  "model": "MODEL_ID_FROM_PROVIDER_DOCS",
  "messages": [
    {
      "role": "user",
      "content": "請用一句話問候。"
    }
  ]
}
```

這是本機產生 JSON 的離線示意，沒有送出網路請求，也沒有模型回覆。

## 選做：OpenAI 相容 API

先安裝 `openai` Python 套件，並由服務商官方文件取得當前 HTTPS base URL 與模型 ID。金鑰放在環境變數 `OPENAI_API_KEY`，不可寫在程式或貼到聊天、Git、截圖。

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)
response = client.chat.completions.create(
    model=os.environ["OPENAI_MODEL"],
    messages=[{"role": "user", "content": "請用一句話問候。"}],
)
print(response.choices[0].message.content)
```

**實際輸出：需金鑰，輸出依實際執行為準。** 本環境未設定 `OPENAI_API_KEY`，因此沒有執行真實 API 呼叫，也不捏造模型回覆。呼叫可能產生費用並把訊息傳給服務商；只用合成文字，先確認帳務、權限與資料政策。

## 版本提醒

原文示範 OpenAI API、GPT 型號與當時的計價說明。SDK、endpoint、模型 ID、可用額度和費率會變；本章不提供價格猜測，執行前查服務商最新官方文件。

## 原文

[Day 3 原文：第一次使用 LLM API：用 OpenAI API 和 LLM 說聲嗨（還有免費的 Groq API）](https://ithelp.ithome.com.tw/articles/10351845?sc=rss.iron)
