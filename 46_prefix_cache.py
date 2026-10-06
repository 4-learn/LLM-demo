"""第 46 節：KV Cache 與 Prefix Cache 的機制與局限（只用 Python 標準函式庫）。

不跑任何模型。我們只**數**：每個請求有幾個 token 需要做 prefill（算 K／V），
有幾個 token 的 K／V 可以從快取拿。

兩個常被混為一談的東西：
  - KV cache：**同一次生成之內**，前面 token 的 K／V 留著，生成下一個 token 時不必重算。
  - Prefix cache：**不同請求之間**，相同的 token 前綴，K／V 可以重用。
兩者都**不是答案快取**：命中只省 prefill，decode（生成答案）一個 token 都沒省。

token 化用「一個字一個 token」簡化，不代表任何真實 tokenizer（見第 02 節）。
"""

BLOCK = 4               # 快取以區塊為單位，部分區塊不能重用（多數實作如此）


def tokenize(text):
    return list(text)


def decode_cost(prompt_len, output_len, kv_cache):
    """生成 output_len 個 token，要對多少個 token 做 attention 的 K/V 計算。"""
    if kv_cache:
        return output_len                     # 每步只算新的那一個
    return sum(prompt_len + step for step in range(1, output_len + 1))   # 每步全部重算


class PrefixCache:
    def __init__(self, capacity_blocks):
        self.capacity = capacity_blocks
        self.blocks = []                       # LRU：越後面越新；每項是 (設定, 前綴 tuple)

    def lookup(self, config, tokens):
        """回傳可重用的 token 數（以完整區塊為單位，必須從頭連續相同、設定也相同）。

        上限是 len(tokens) - 1：最後一個輸入 token 一定要重算，才能產生第一個輸出 token 的
        logits。所以即使整段完全相同，也不可能 prefill 為 0。
        """
        reused = 0
        for end in range(BLOCK, len(tokens), BLOCK):
            key = (config, tuple(tokens[:end]))
            if key in self.blocks:
                self.blocks.remove(key)
                self.blocks.append(key)       # 用到就移到最新
                reused = end
            else:
                break
        return reused

    def store(self, config, tokens):
        for end in range(BLOCK, len(tokens) + 1, BLOCK):
            key = (config, tuple(tokens[:end]))
            if key not in self.blocks:
                self.blocks.append(key)
            while len(self.blocks) > self.capacity:
                self.blocks.pop(0)             # 驅逐最舊的


SYSTEM = "你是工地巡檢助理只依通報內容回答"            # 16 字 = 4 個完整區塊
REQUESTS = (
    ("r1 首次",                 "model-a", SYSTEM + "三號出口有紙箱嗎", 12),
    ("r2 同前綴、不同問題",     "model-a", SYSTEM + "有人沒戴安全帽嗎", 12),
    ("r3 前綴多一個空白",       "model-a", " " + SYSTEM + "有人沒戴安全帽嗎", 12),
    ("r4 同前綴、不同模型設定", "model-b", SYSTEM + "有人沒戴安全帽嗎", 12),
    ("r5 動態內容放在最前面",   "model-a", "2026-10-05" + SYSTEM + "有人沒戴安全帽嗎", 12),
    ("r6 完全相同的請求",       "model-a", SYSTEM + "有人沒戴安全帽嗎", 12),
)


def run(capacity):
    cache = PrefixCache(capacity)
    rows = []
    for label, config, prompt, output_len in REQUESTS:
        tokens = tokenize(prompt)
        reused = cache.lookup(config, tokens)
        cache.store(config, tokens)
        rows.append((label, len(tokens), reused, len(tokens) - reused, output_len))
    return rows


def main():
    print("=== 一、KV cache：同一次生成之內 ===")
    for prompt_len, output_len in ((20, 10), (20, 100), (1000, 100)):
        without = decode_cost(prompt_len, output_len, kv_cache=False)
        with_cache = decode_cost(prompt_len, output_len, kv_cache=True)
        print(f"  輸入 {prompt_len:>4}、輸出 {output_len:>3}：沒有 KV cache 要算 {without:>6} 個 token 的 K/V，"
              f"有的話 {with_cache:>3}（{without / with_cache:.1f} 倍）")
    print("  代價：K/V 要一直留在記憶體裡，上下文越長佔越多（第 05 節：Qwen 8k 上下文 96 MB）。")
    print()

    print(f"=== 二、Prefix cache：不同請求之間（區塊 {BLOCK} token，容量充足）===")
    print(f"  {'請求':<22} {'輸入':>4} {'重用':>4} {'需prefill':>9} {'decode':>6}")
    for label, total, reused, prefill, output_len in run(capacity=100):
        print(f"  {label:<22} {total:>4} {reused:>4} {prefill:>9} {output_len:>6}")
    print()

    print("=== 三、容量不足時：同樣的請求，驅逐之後 ===")
    small = run(capacity=6)
    for label, total, reused, prefill, output_len in small:
        print(f"  {label:<22} {total:>4} {reused:>4} {prefill:>9} {output_len:>6}")
    print()

    print("=== 四、命中 ≠ 答案快取 ===")
    full = [r for r in run(capacity=100) if r[0].startswith("r6")][0]
    print(f"  r6 與 r2 完全相同，prefill 只剩 {full[3]} 個 token（省了 {full[2]}/{full[1]}；"
          f"最後一個區塊必須重算才能產生第一個輸出），")
    print(f"  但 decode 仍是 {full[4]} 個 token——答案要重新生成，取樣設定不同時內容也可能不同（第 04 節）。")


if __name__ == "__main__":
    main()
