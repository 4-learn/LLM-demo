# Day 5｜對話記憶就是保留上下文

> **定位**：用 Python 保存一段對話歷史，觀察每次請求要帶上相關訊息。

## 學習目標

- 說明上下文是模型本次可見的輸入內容。
- 用 list 保存 user/assistant 訊息。
- 知道歷史訊息會增加輸入量且可能包含敏感內容。

## 概念

上下文（context）是模型在一次生成時可讀取的訊息與資料；模型不會因為呼叫結束就自動知道程式先前的對話。訊息常用 list（列表）依序保存；每則訊息以 dict（字典）記錄角色與內容。API 的 token（詞元）是模型處理文字的單位，歷史越長通常會增加輸入量；它不等同密碼或 API key。

## 動手做：建立最小對話紀錄

```python
messages = []
messages.append({"role": "user", "content": "我喜歡茶。"})
messages.append({"role": "assistant", "content": "你喜歡茶。"})
messages.append({"role": "user", "content": "我剛才說喜歡什麼？"})
print("訊息筆數：", len(messages))
print("最後一則：", messages[-1]["content"])
```

**實際輸出（以 Python 執行上述程式）：**

```text
訊息筆數： 3
最後一則： 我剛才說喜歡什麼？
```

真實 API 呼叫時通常要把需要的歷史訊息一併送出，使用 OpenAI 相容 `messages` 格式；這段程式只是本機保存，並沒有模型記憶或網路傳輸。

## 練習

新增一則 assistant 回覆，再印出訊息筆數。思考若只保留最後兩則，模型還能否看到「我喜歡茶」？

## 版本提醒

原文的 GPT-4o-mini 上下文限制與 token 數是當時資料，不作為現況保證。模型上下文、API 欄位與費用各有差異，使用前查官方文件；不要把私人對話送至未核准服務。

## 原文

[Day 5 原文：初探 AI 記憶功能：讓你的 LLM 不再失憶](https://ithelp.ithome.com.tw/articles/10354011?sc=rss.iron)
