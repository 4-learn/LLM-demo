# Day 12｜評估驅動開發：用固定案例檢查輸出

> **主題：** Prompt 或模型改動後，以固定測試集檢查是否更符合需求，而不是挑一個好看的回答。

Prompt（提示詞）是交給模型的任務指示文字；本章的輸出評估不呼叫模型，而是用固定案例檢查程式結果。

## 動手做：精確比對

```python
cases = [
    ("strawberry", "r", 3),
    ("banana", "a", 3),
    ("apple", "p", 2),
    ("kiwi", "z", 0),
]
predictions = [text.count(char) for text, char, _ in cases]
correct = sum(pred == expected for pred, (_, _, expected) in zip(predictions, cases))
for (text, char, expected), pred in zip(cases, predictions):
    print(f"{text!r} 中 {char!r}: expected={expected}, actual={pred}")
print(f"accuracy: {correct}/{len(cases)} = {correct / len(cases):.2f}")
```

預期輸出：

```text
'strawberry' 中 'r': expected=3, actual=3
'banana' 中 'a': expected=3, actual=3
'apple' 中 'p': expected=2, actual=2
'kiwi' 中 'z': expected=0, actual=0
accuracy: 4/4 = 1.00
```

此任務用 Python 可精確計數，因此不需要 LLM。對生成任務，還需人工評分或適當的評估準則；單一正確率未必能表達完整品質。測試集應涵蓋常見、邊界及失敗案例，並與用來調整 Prompt 的資料分開。

## 小練習

把 `"Strawberry"`、空字串及大小寫是否相同的案例加入測試集。先寫清楚產品規格，再決定程式是否要將字串轉小寫。

## 原文與版本

原文：[Day 12｜初探 LLM 評估驅動開發與 Prompt 生成器：用 AI 來調教 AI](https://ithelp.ithome.com.tw/articles/10358186?sc=rss.iron)

原文介紹評估驅動開發、Prompt 產生器與當時工具。生成式模型及評估服務的 API、模型名稱與價格會變動；本練習是純 Python、可重現的精確比對，不依賴供應商 API。
