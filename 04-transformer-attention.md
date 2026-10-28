# 04｜Attention｜用權重混合資訊

> 本課只學一件事：不同權重會改變加權平均；權重總和為 1 才可直接視為比例。

## 從熟悉的例子開始

兩家店評分是 10 和 20。如果較喜歡第二家，就給第二家較大的權重，結果會更靠近 20。Attention 也會依目前內容分配權重，再混合資訊；這個玩具例子只說明「加權」，不是完整模型，也不代表模型真的理解或答案正確。

## Demo：加權平均

<!-- demo: 04_transformer_attention.py -->
```python
def average(weights, values):
    total_weight = 0
    weighted_sum = 0
    for i in range(len(weights)):
        total_weight += weights[i]
        weighted_sum += weights[i] * values[i]
    return weighted_sum / total_weight

weights = [0.25, 0.75]
values = [10, 20]
print(average(weights, values))
```

輸出 `17.5`。加權平均要用「加權總和 ÷ 權重總和」。此例權重相加為 1；若權重不是 1，除以總和才能得到比例平均。例如 `[1, 1]` 的加權平均是 15，不是 30。

## 這跟 LLM 的關係

常見 Transformer attention 先用 softmax 正規化權重，再混合 token 資訊。本程式另外除以權重總和，是一般加權平均的寫法，不是所有 attention 變體都必然如此；權重也不保證答案正確。

## 動手試試

把 `weights` 改成 `[0.75, 0.25]`；預期輸出 `12.5`，比原本更靠近 10。再試 `[1, 1]`；預期仍是 `15.0`，因為程式會除以權重總和 2。

## 自我檢查

1. `[0.25, 0.75]` 權重總和是多少？2. `[1, 1]` 為什麼除以 2？3. 權重較大代表答案正確嗎？

**要點**：1；化成比例；不代表正確。
