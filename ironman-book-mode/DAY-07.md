# Day 7｜工具呼叫：資料要驗證，動作由程式控制

> **定位**：用固定的工具請求示範函式呼叫的資料流，不依賴模型輸出或第三方套件。

## 學習目標

- 說明工具呼叫只是請求，不是已執行動作。
- 用白名單和型別檢查驗證工具參數。
- 認識 OpenAI 相容 API 的工具呼叫需查當前 schema。

## 概念

白名單是程式明確允許的有限選項集合；未知工具或不合規參數應拒絕，不可直接執行。Schema（資料結構規格）描述欄位、型別與限制；各服務的工具呼叫 schema 可能不同，需按當前文件確認。

## 動手做：安全執行一個工具

```python
def count_letter(text, letter):
    if not isinstance(text, str) or not isinstance(letter, str) or len(letter) != 1:
        raise ValueError("需要文字和單一字元")
    return text.count(letter)

request = {"name": "count_letter", "arguments": {"text": "banana", "letter": "a"}}
allowed_tools = {"count_letter": count_letter}
if request["name"] not in allowed_tools:
    raise ValueError("未知工具")
result = allowed_tools[request["name"]](**request["arguments"])
print("工具結果：", result)
```

**實際輸出（以 Python 執行上述程式）：**

```text
工具結果： 3
```

模型即使透過 OpenAI 相容 API 提出類似的 tool call，也不代表請求可信；應用程式要自行驗證名稱、參數、權限和副作用。

## 練習

把 `letter` 改成兩個字元，確認程式會拒絕。請勿用 `eval` 執行模型提供的字串。

## 版本提醒

原文的 OpenAI function calling 介面及模型行為可能已變。不同供應商雖可能採 OpenAI 相容格式，欄位支援仍不完全相同；依官方文件核對，不猜測模型或價格。

## 原文

[Day 7 原文：讓模型使用工具（2）：透過 tool use（function calling）API 提升模型輸出穩定性](https://ithelp.ithome.com.tw/articles/10355329?sc=rss.iron)
