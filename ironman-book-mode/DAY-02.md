# Day 2｜AI、LLM 與文字接龍

> **定位**：用小程式觀察「依前文預測下一個字詞」的概念，不下載模型、不連網。

## 學習目標

- 說明 AI 與 LLM 的範圍差異。
- 用 Python 建立簡單的下一詞頻率表。

## 概念

AI（人工智慧）是讓電腦完成智能任務的廣泛領域；LLM（大型語言模型）是以語言為主要處理對象的一類模型。**推論**是把輸入交給已訓練模型並取得輸出的過程。Token（詞元）是模型處理文字時使用的基本單位，可能是一個字、字的一部分或標點；不一定等於一個中文字或單字。

Softmax（軟最大化）是把一組分數轉成總和為 1 的機率分布，讓模型可比較各候選 token 的相對可能性。

真實 LLM 會依上下文估計後續 token 的機率；下面用固定句子做玩具示範，並非 LLM。

## 動手做

```python
sentences = ["我 喜歡 蘋果", "我 喜歡 香蕉", "我 喜歡 蘋果"]
counts = {}
for sentence in sentences:
    words = sentence.split()
    pair = (words[0], words[1])
    counts.setdefault(pair, {})
    counts[pair][words[2]] = counts[pair].get(words[2], 0) + 1

next_word = max(counts[("我", "喜歡")], key=counts[("我", "喜歡")].get)
print("前文：我 喜歡")
print("最常見的下一詞：" + next_word)
```

**實際輸出（以 Python 執行上述程式）：**

```text
前文：我 喜歡
最常見的下一詞：蘋果
```

這個頻率表只從三句例句統計，不理解語意；模型的能力也不等於人類式理解。

## 練習

新增句子 `我 喜歡 葡萄`，觀察預測是否改變。若次數相同，程式依插入順序取第一個，這只是此示範的規則。

## 版本提醒

原文使用 GPT-2 Transformers 範例；模型版本、套件安裝方式與生成結果會變。此處用標準函式庫，避免假設當前模型或下載流程。後續 LLM API 範例採 OpenAI 相容格式，執行前查看供應商文件；不推測價格或現行可用模型。

## 原文

[Day 2 原文：什麼是 AI 和大型語言模型（LLM）？](https://ithelp.ithome.com.tw/articles/10351250?sc=rss.iron)
