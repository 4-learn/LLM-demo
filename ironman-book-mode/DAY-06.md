# Day 6｜讓程式做精確計算

> **定位**：理解模型可提出工具需求，精確計算交給普通 Python 函式。

## 學習目標

- 用 Python 計算字母出現次數。
- 認識工具呼叫是模型提出結構化請求、程式執行工具並回傳結果的流程。
- 知道不可直接執行未驗證的模型文字。

## 概念

工具呼叫（tool calling）是模型依約定提出工具名稱和參數，由應用程式驗證後執行，而不是模型自行取得電腦權限。

## 動手做：計算 Strawberry 的 r

```python
word = "Strawberry"
count = word.lower().count("r")
print(f"{word} 裡有 {count} 個 r")
```

**實際輸出（以 Python 執行上述程式）：**

```text
Strawberry 裡有 3 個 r
```

這種明確計算比請生成模型猜測可靠。後續若模型透過 OpenAI 相容 API 要求工具，應用程式仍要檢查工具名稱、參數型別、長度及使用者權限，再呼叫白名單中的函式。

## 練習

將 `word` 改成 `banana`、目標字母改成 `a`，預測並執行確認結果。

## 版本提醒

原文以 2024 年模型例子說明字母計數失誤。模型版本及 function/tool calling API 欄位可能改變；不要依賴特定模型必然選對工具。費用、模型與端點請查當前官方文件。

## 原文

[Day 6 原文：讓模型使用工具（1）：連 Strawberry 有幾個 r 都不會算？那就用程式算吧！](https://ithelp.ithome.com.tw/articles/10354715?sc=rss.iron)
