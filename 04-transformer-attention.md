# 04｜Transformer、Attention、MoE 與推理模型

> 本課只學一件事：**不同權重會讓加權結果改變**。其他名詞是延伸，不要求背誦。

[回課程書目](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 看一個熟悉的例子

你要選午餐，附近兩家店評分各是 10 分和 20 分。若比較喜歡第二家，就給它較大的權重；加權結果也會更靠近 20。模型中的 Attention（注意力）也會依目前內容，替不同資訊分配不同權重，再混合資訊。

本課先不推導公式。只記得：權重是計算用的比例，不代表模型真的理解，也不保證答案正確。

## Demo：加權平均

<!-- demo: 04_transformer_attention.py -->
```python
def average(weights, values):
    total = 0
    for i in range(len(weights)):
        total += weights[i] * values[i]
    return total

weights = [0.25, 0.75]
values = [10, 20]
print(average(weights, values))
```

輸出：`17.5`。第二筆權重較大，所以結果較接近 20。這是 Attention 的小玩具，不是完整模型。

## 動手試試

把 `weights` 改成 `[0.75, 0.25]`，再執行一次。結果應更靠近 10。想一想：權重改變了，哪些資料的影響變大？

**延伸名詞（不要求本課掌握）**：Query/Key/Value、softmax、多頭 Attention、MoE、推理模型。
