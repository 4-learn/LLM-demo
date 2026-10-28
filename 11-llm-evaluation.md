# 11｜檢查模型回答：正確率

> 本課只學一件事：用幾個有標準答案的例子，算出模型答對幾題。

[回課程書目](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 小型測驗

我們問模型三題，標準答案是「對、對、錯」。模型回答「對、錯、對」。逐題比較，只有第一題答對，因此正確率是 1/3。這個數字只描述這三題，題目太少不能代表所有情況。

## Demo：逐題比較

<!-- demo: 11_llm_evaluation.py -->
```python
answers = ["對", "對", "錯"]
predictions = ["對", "錯", "對"]

correct = 0
for i in range(len(answers)):
    if answers[i] == predictions[i]:
        correct += 1

print("答對", correct, "題")
print("總題數", len(answers))
print("正確率", correct / len(answers))
```

輸出正確率為 `0.333...`。

## 動手試試

只改 `predictions`，讓兩題答對，再執行。想一想：題目變多、變難時，這個結果會不會改變？

**延伸名詞**：Precision、Recall、Judge 評分與回歸測試都能回答不同評估問題，本課先不實作。
