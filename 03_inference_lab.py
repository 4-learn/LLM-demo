"""第 03 節：文字接龍機（只用 Python 標準函式庫）。

從合成工安句子數出「前一個字後面接什麼字」，把次數當成 logits 轉成
next-token 機率，再一個一個接下去。它只看前一個字，是教學用的最小
語言模型，不是 LLM；但「算機率 → 選一個 → 接回去 → 再算」的生成
流程和 LLM 相同。
"""

import math
import random
from collections import Counter, defaultdict

BOS, EOS = "<s>", "<eos>"
CORPUS = [
    "請立即通知現場安全人員。",
    "請立即通知工安主任到場。",
    "請立即暫停施工並確認現場。",
    "請立即提醒人員配戴安全帽。",
    "請提醒人員配戴安全帽。",
    "請提醒人員離開施工區。",
    "請通知安全人員到場確認。",
    "請通知工安主任覆核影像。",
    "請確認人員是否配戴安全帽。",
    "請確認現場是否有人進入施工區。",
    "請安全人員到場確認。",
    "請安全人員覆核影像後回報。",
]


def learn(sentences):
    """訓練：數每個字後面接了哪些字、各幾次。"""
    table = defaultdict(Counter)
    for sentence in sentences:
        tokens = [BOS] + list(sentence) + [EOS]
        for prev, nxt in zip(tokens, tokens[1:]):
            table[prev][nxt] += 1
    return table


def next_probs(table, prev, temperature=1.0):
    """推論一步：次數 → logits → softmax → 下一個 token 的機率。"""
    if not (math.isfinite(temperature) and temperature > 0):
        raise ValueError("temperature must be finite and positive")
    counts = table.get(prev)
    if not counts:
        raise ValueError(f"語料裡沒看過「{prev}」後面接什麼")
    logits = {tok: math.log(n) for tok, n in counts.most_common()}
    largest = max(logits.values())
    weights = {tok: math.exp((z - largest) / temperature) for tok, z in logits.items()}
    total = sum(weights.values())
    return {tok: w / total for tok, w in weights.items()}


def generate(table, prompt="請", max_new_tokens=30, greedy=False,
             temperature=1.0, seed=None, trace=None):
    """接龍：算機率 → 選一個 → 接回去 → 再算，直到 <eos> 或長度上限。"""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    rng = random.Random(seed)
    out = list(prompt)
    for _ in range(max_new_tokens):
        prev = out[-1] if out else BOS
        probs = next_probs(table, prev, temperature)
        if greedy:
            tok = max(probs, key=probs.get)
        else:
            tok = rng.choices(list(probs), weights=list(probs.values()))[0]
        if trace is not None:
            trace.append((prev, probs, tok))
        if tok == EOS:
            return "".join(out), "eos"
        out.append(tok)
    return "".join(out), "length"


def show(probs, top=4):
    return "｜".join(f"{tok} {p:.2f}" for tok, p in list(probs.items())[:top])


if __name__ == "__main__":
    table = learn(CORPUS)

    print("== 1. 「請」後面接什麼？")
    print("  次數：", dict(table["請"].most_common()))
    print("  機率：", show(next_probs(table, "請"), top=5))

    print("== 2. 從「請」開始接龍（seed=1，逐步列出）")
    steps = []
    text, reason = generate(table, "請", seed=1, trace=steps)
    for i, (prev, probs, tok) in enumerate(steps, 1):
        print(f"  step={i:<2} 前一字「{prev}」→ {show(probs)} → 選「{tok}」")
    print(f"  結果：{text}  stop={reason}")

    print("== 3. 同一個開頭，抽三次籤（seed=2、3、4）")
    for seed in (2, 3, 4):
        print(f"  seed={seed}：", *generate(table, "請", seed=seed))

    print("== 4. 固定 seed=7，跑三次")
    for _ in range(3):
        print("  ", *generate(table, "請", seed=7))

    print("== 5. greedy：每步都選最高機率")
    print("  ", *generate(table, "請", greedy=True))
    print("  ", *generate(table, "請", greedy=True, max_new_tokens=5))

    print("== 6. temperature 改變「請」後面的分布")
    for t in (0.5, 1.0, 2.0):
        print(f"  T={t}：", show(next_probs(table, "請", t), top=5))

    print("== 7. 沒看過的開頭")
    try:
        generate(table, "頭盔")
    except ValueError as error:
        print("  ", error)