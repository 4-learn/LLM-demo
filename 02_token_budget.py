"""第 02 節：子詞切分與上下文預算（只用 Python 標準函式庫）。

玩具 BPE 用頻率合併字元對，示範「子詞」怎麼來；它不是你的 API 用的
tokenizer，計數不可互換。plan_context() 的預算公式與計數器無關，
換一個 count 就能接真實 tokenizer。
"""

from collections import Counter

END = "</w>"
CORPUS = [
    "安全帽", "安全帽", "未戴安全帽", "已戴安全帽",
    "施工區", "施工區", "進入施工區",
    "通報", "通報", "現場通報",
]
MERGE_COUNT = 6


def learn_merges(corpus, merge_count):
    words = [list(word) + [END] for word in corpus]
    rules = []
    for _ in range(merge_count):
        counts = Counter()
        for word in words:
            counts.update(zip(word, word[1:]))
        if not counts:
            break
        best = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
        rules.append(best)
        words = [_merge(word, best) for word in words]
    return rules


def _merge(symbols, pair):
    merged, index = [], 0
    while index < len(symbols):
        if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
            merged.append(symbols[index] + symbols[index + 1])
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return merged


def apply_rules(text, rules):
    tokens = []
    for word in text.split():
        symbols = list(word) + [END]
        for pair in rules:
            symbols = _merge(symbols, pair)
        for symbol in symbols:
            if symbol == END:
                continue
            tokens.append(symbol[: -len(END)] if symbol.endswith(END) else symbol)
    return tokens


def plan_context(window, system_tokens, output_reserve, safety_margin, evidences, count):
    fixed = system_tokens + output_reserve + safety_margin
    if fixed >= window:
        raise ValueError("fixed overhead does not fit in the context window")
    remaining = window - fixed
    kept, dropped, used = [], [], 0
    for name, text in evidences:
        cost = count(text)
        if used + cost <= remaining:
            kept.append((name, cost))
            used += cost
        else:
            dropped.append((name, cost))
    return {
        "window": window,
        "fixed": fixed,
        "remaining": remaining,
        "used": used,
        "spare": remaining - used,
        "kept": kept,
        "dropped": dropped,
    }


if __name__ == "__main__":
    rules = learn_merges(CORPUS, MERGE_COUNT)
    print("rules:", ["+".join(pair) for pair in rules])
    toy_count = lambda text: len(apply_rules(text, rules))
    for text in ("安全帽", "未戴安全帽", "施工區", "現場通報", "頭盔", "AB-123"):
        print(f"{text} chars={len(text)} toy_tokens={toy_count(text)}")
    result = plan_context(
        16,
        system_tokens=5,
        output_reserve=4,
        safety_margin=1,
        evidences=[
            ("E1", "安全帽"),
            ("E2", "未戴安全帽"),
            ("E3", "施工區"),
            ("E4", "現場通報"),
            ("E5", "頭盔"),
        ],
        count=toy_count,
    )
    print("remaining:", result["remaining"], "used:", result["used"], "spare:", result["spare"])
    print("kept:", result["kept"])
    print("dropped:", result["dropped"])
    try:
        plan_context(8, 5, 4, 1, [], toy_count)
    except ValueError as exc:
        print("tiny window:", exc)
