# KV Cache 與 Prefix Cache 的機制與局限

> **新版正文：20E07-E03-05｜對應新版 LLM 第 E08 節：KV Cache 與 Prefix Cache。**
> 兩個名字很像、常被混為一談的機制。核心主張：**兩者都只省 prefill（處理輸入），都不是答案快取；命中不代表答案免推論，更不代表品質。** 本節不跑模型，只用 Python 標準函式庫**數**每個請求有多少 token 需要計算。

[回 51 節必修＋9 節選修課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 流程圖：KV Cache 與 Prefix Cache

```mermaid
flowchart TD
    A[請求 prompt 與模型／執行設定] --> B[Tokenize 並計算 prefix cache key]
    B --> C[Key 包含租戶／隔離範圍、模型及相容設定]
    C --> D{相同 token prefix 的完整區塊命中？}
    D -- 命中 --> E[重用 prefix K/V，剩餘 prompt 繼續 prefill]
    D -- miss --> F[完整 prompt 做 prefill，建立可快取 prefix K/V]
    E --> G[對本次請求執行 decode 生成答案]
    F --> G
    G --> H[回傳本次輸出，不重用其他請求答案]
    H --> I[比較命中／未命中 token 計算量與輸出一致性]
    J[租戶或模型設定不相容] --> K[隔離快取，不共享 K/V]

## 學習目標與時間

先修第 02 節（token）、第 45 節（prefill 與 decode）、第 46 節（記憶體量測）；第 05 節的 KV cache 記憶體公式。

| 時間 | 內容 |
| --- | --- |
| 0–8 分鐘 | K／V 是什麼；為什麼要留著 |
| 8–18 分鐘 | KV cache：同一次生成之內 |
| 18–28 分鐘 | Prefix cache：不同請求之間；命中條件與 miss 原因 |
| 28–35 分鐘 | 跑 Demo：六個請求、兩種容量 |
| 35–50 分鐘 | Workshop：改提示順序、改容量、算省下多少 |

驗收能力：交付 **cache 命中條件與限制表**，不能把命中當成整段答案免推論或品質保證。

## 一、K／V 是什麼，為什麼要留著

Transformer 生成每一個新 token 時，要讓它「看」前面所有的 token。看的方式是用新 token 的 Q（query）去比對前面每個 token 的 K（key），再加權取 V（value）。

重點是：**前面 token 的 K、V 一旦算出來就不會變**。所以有兩個地方可以省：

| 機制 | 範圍 | 省什麼 |
| --- | --- | --- |
| **KV cache** | **同一次生成之內** | 生成第 n 個 token 時，不必重算前 n−1 個的 K／V |
| **Prefix cache** | **不同請求之間** | 新請求的開頭跟之前某個請求一樣時，不必重算那段開頭的 K／V |

## 二、KV cache：省很多，但要付記憶體

沒有 KV cache 時，每生成一個 token 都要把**整段**（輸入＋已生成）的 K／V 重算一次。有 KV cache 時，每步只算新的那一個。Demo 第一段算給你看：輸入越長、輸出越長，差距越大。

代價是**記憶體**：K／V 要一直留著。第 05 節算過，`Qwen2.5-0.5B-Instruct` 在 8192 token 上下文、單一序列下要 96 MB；同樣條件下沒有 GQA 的 `Phi-3.5-mini` 要 3072 MB。上下文變長、同時服務的請求變多，KV cache 往往比模型權重還先把記憶體吃光。

**KV cache 現在是所有自迴歸推論的標準做法**，不是可選的優化。本節講它，是為了讓你知道 prefix cache 是在它之上**再多一層**。

## 三、Prefix cache：命中條件很嚴格

Prefix cache 讓不同請求共用相同開頭的 K／V。它的命中條件是：

| 條件 | 為什麼 |
| --- | --- |
| **從第一個 token 開始**連續相同 | K／V 取決於它前面所有的 token；前面差一個字，後面全部不同 |
| **同一個模型與相容的執行設定** | 不同模型、不同精度的 K／V 不能混用 |
| **以完整區塊為單位** | 多數實作以固定大小的區塊管理；不滿一塊的尾巴不重用 |
| **還沒被驅逐** | 快取容量有限，舊的會被擠掉 |
| **最後一個 token 一定重算** | 要產生第一個輸出 token 的 logits，至少要算最後一個位置 |

「從第一個 token 開始」是最常被違反的一條：**把日期、使用者名稱、請求 ID 放在提示最前面**，等於保證每個請求都 miss。

## 四、可執行 Demo

六個請求，共用一段 16 字的系統提示（剛好 4 個區塊）。token 化用「一個字一個 token」簡化，**不代表任何真實 tokenizer**；區塊大小 4 也是教學值。

<!-- demo: E08_prefix_cache.py -->
```python
"""第 47 節：KV Cache 與 Prefix Cache 的機制與局限（只用 Python 標準函式庫）。

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
    ("r5 動態內容放在最前面",   "model-a", "20E07-E03-05" + SYSTEM + "有人沒戴安全帽嗎", 12),
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
```

執行指令（在 [`llm-demo`](https://github.com/4-learn/llm-demo) 目錄下，**不需要網路、不需要模型**）：

```bash
python3 E08_prefix_cache.py
```

關鍵輸出（Linux／Python 3.10.15，20E07-E03-05）：

```text
=== 一、KV cache：同一次生成之內 ===
  輸入   20、輸出  10：沒有 KV cache 要算    255 個 token 的 K/V，有的話  10（25.5 倍）
  輸入   20、輸出 100：沒有 KV cache 要算   7050 個 token 的 K/V，有的話 100（70.5 倍）
  輸入 1000、輸出 100：沒有 KV cache 要算 105050 個 token 的 K/V，有的話 100（1050.5 倍）
  代價：K/V 要一直留在記憶體裡，上下文越長佔越多（第 05 節：Qwen 8k 上下文 96 MB）。

=== 二、Prefix cache：不同請求之間（區塊 4 token，容量充足）===
  請求                       輸入   重用  需prefill decode
  r1 首次                    24    0        24     12
  r2 同前綴、不同問題              24   16         8     12
  r3 前綴多一個空白               25    0        25     12
  r4 同前綴、不同模型設定            24    0        24     12
  r5 動態內容放在最前面             34    0        34     12
  r6 完全相同的請求               24   20         4     12

=== 三、容量不足時：同樣的請求，驅逐之後 ===
  r1 首次                    24    0        24     12
  r2 同前綴、不同問題              24   16         8     12
  r3 前綴多一個空白               25    0        25     12
  r4 同前綴、不同模型設定            24    0        24     12
  r5 動態內容放在最前面             34    0        34     12
  r6 完全相同的請求               24    0        24     12

=== 四、命中 ≠ 答案快取 ===
  r6 與 r2 完全相同，prefill 只剩 4 個 token（省了 20/24；最後一個區塊必須重算才能產生第一個輸出），
  但 decode 仍是 12 個 token——答案要重新生成，取樣設定不同時內容也可能不同（第 04 節）。
```

## 五、結果怎麼讀

### 1. 六個請求，只有兩個命中

| 請求 | 重用 | 原因 |
| --- | --- | --- |
| r1 首次 | 0 | 快取是空的 |
| r2 同前綴、不同問題 | **16** | 系統提示 16 字＝4 個完整區塊，全部命中；問題部分不同，要重算 |
| r3 前綴多一個空白 | 0 | **第一個 token 就不同**，後面全部錯位 |
| r4 不同模型設定 | 0 | 文字完全相同，但 `model-b` 的 K／V 不能用 `model-a` 的 |
| r5 日期放在最前面 | 0 | 跟 r3 同一個問題，只是更常見 |
| r6 完全相同 | **20** | 24 個 token 裡，最後一個區塊要重算 |

r3 是最值得記住的一列：**一個空白就讓 16 個 token 的快取完全失效。** 真實系統裡，這種差異常常來自模板拼接時多一個換行、或不同程式碼路徑產生了略有不同的系統提示。

### 2. 驅逐讓 r6 從 20 變成 0

第三段用同樣的請求、但快取只能放 6 個區塊。r3、r4、r5 各自塞進新的區塊，把 r1／r2 留下的系統提示區塊**擠掉了**。r6 跟 r2 一字不差，**重用卻是 0**。

所以「這個前綴之前出現過」不保證命中。命中率取決於流量模式與容量，**要量測，不能假設**。

### 3. 命中省的只有 prefill

```text
r6 與 r2 完全相同，prefill 只剩 4 個 token（省了 20/24；…），
但 decode 仍是 12 個 token
```

即使是完全相同的請求，**答案仍然要一個 token 一個 token 生成**。prefix cache 省的是第 45 節的 TTFT（第一個 token 出來前的時間），不是生成答案的時間。

而且答案**不一定一樣**：如果取樣溫度大於 0（第 04 節），同一個前綴、同樣的 K／V，生成出來的內容仍可能不同。**prefix cache 不是答案快取，也不保證答案品質。**

### 4. 資料隔離

prefix cache 在多個使用者之間共用時，有一個安全問題：**命中與否會影響回應時間**。理論上，攻擊者可以透過量測延遲，推測別人是否送過某段特定開頭的提示。服務端通常以租戶或使用者隔離快取來處理；自建服務時要確認這一點。

## 六、常見誤判

| 誤判 | 為什麼錯 |
| --- | --- |
| 「prefix cache 命中，答案就是快取的」 | 只省 prefill；decode 照樣跑，答案要重新生成 |
| 「同樣的問題問兩次，第二次一定比較快」 | 可能已被驅逐（第三段 r6 = 0） |
| 「系統提示一樣就會命中」 | 必須從第一個 token 開始一樣；前面多一個空白或日期就全部 miss |
| 「KV cache 和 prefix cache 是同一個東西」 | 前者在一次生成內，後者跨請求 |
| 「命中代表品質一樣」 | 取樣溫度 > 0 時，同樣的 K／V 也會生成不同答案 |
| 「快取越大越好」 | 快取吃的是 KV cache 同一塊記憶體；給快取越多，能同時服務的請求越少 |
| 「快取在不同使用者之間共用沒問題」 | 命中時間差可能洩漏別人的提示內容；要隔離 |

## Workshop：命中條件與限制

### 任務

1. 執行 Demo，先預測 r2～r6 各自的重用數，再核對。
2. 把 r5 改成「日期放在**最後面**」（`SYSTEM + "有人沒戴安全帽嗎" + "20E07-E03-05"`），重跑。重用變成多少？**這告訴你提示該怎麼排？**
3. 把 `BLOCK` 改成 1，重跑第二段。r2 和 r6 的重用各變成多少？區塊越小越好嗎？（提示：想想管理成本。）
4. 第三段容量 6 時 r6 是 0。容量至少要多少，r6 才會重新命中 20？用程式找出來，不要用猜的。
5. 用 Demo 第一段的公式，算「輸入 500、輸出 50」沒有 KV cache 時的 K／V 計算量，再用程式驗證。

### 參考判讀

- 第 2 題：重用變成 **24**——系統提示與問題（共 24 字）都在前面且與 r2 一字不差，日期接在後面，所以 r2 留下的 6 個區塊全部命中；新請求長度 34，24 小於上限 33，不受「最後一個 token 要重算」影響。原本放最前面時是 **0**。**把固定內容放前面、變動內容放後面**，是使用 prefix cache 最重要的一條規則。
- 第 3 題：區塊 1 時 r2 仍是 **16**（問題第一個字「有」與 r1 的「三」就不同）。但 **r6 變成 0**——區塊 1 時每個請求都產生 24～34 個區塊，到 r5 時快取已經塞滿 100 個區塊，r2 的區塊被驅逐了。**區塊越小，同樣的容量能記住的請求越少。**這不是直覺上的「越精細越好」，而是記憶體換重用率的取捨。
- 第 4 題：用迴圈從 6 往上試，**最小容量是 26 個區塊**。r1～r5 依序新增 6、2、6、6、8 個區塊，共 28 個；但 r6 只需要 r2 的前 5 個區塊（重用 20），而 r1 尾端那 2 個區塊（「三號出口…」）在 LRU 裡最舊、可以先被驅逐，所以 28 − 2 = 26。這種推理很容易漏掉一步，這正是要你用程式找的原因；測試 `test_minimum_capacity_for_r6_hit` 釘住 26。
- 第 5 題：`sum(500 + step for step in 1..50)` = 500 × 50 + (1 + 2 + … + 50) = 25000 + 1275 = **26275**；有 KV cache 時是 50。

### 驗收

- 交出**命中條件與限制表**：至少列出前綴連續、模型設定相容、區塊對齊、驅逐、最後 token 重算五項，每項附一個你在 Demo 裡實際觀察到的例子。
- 分開報 prefill 與 decode：寫明命中省了哪一段、沒省哪一段。
- 不得宣稱命中等於答案快取，或命中代表品質相同。
- 若使用真實服務的快取統計，附服務名稱、版本與日期；本地 runtime 不支援時，寫明本節是機制模擬，不是量測。

## 教師紀錄與下一步

- 本節**完全不需要模型、網路或任何 SDK**，也**不是效能量測**：它只數「需要計算 K／V 的 token 數」，不量時間。證據等級：**本機實測（快取規則的行為模擬）**。
- 一字一 token、區塊 4、LRU 驅逐都是教學簡化；真實服務的區塊大小、驅逐策略與命中統計方式各不相同，**課綱要求的 hit／miss trace 需要真實 runtime**，屬第 07 節（本地推論）待決的範圍。
- 撰寫時的實際紀錄：① 第一版系統提示是 17 字而註解寫 16 字，r2 的「重用 16」其實只覆蓋了系統提示的前 16 字；改成真正的 16 字後區塊才對齊。② 第一版允許整段完全重用，r6 顯示 **prefill 為 0**——這在真實模型上不可能，因為產生第一個輸出 token 至少要算最後一個輸入位置；已將重用上限改為 `len(tokens) - 1`。
- Workshop 參考判讀初稿有兩處憑推論寫錯，實跑後改正：第 2 題寫成 20（實際 24）；第 3 題寫成 r6 重用 23（實際 0，因為區塊 1 時容量 100 被塞滿而驅逐）。後者反而成為該題更好的答案。全部由測試釘住。
- 待教師確認：第 07 節的本地 runtime 定案後，是否補一段真實的 prefill 時間對照（同前綴 vs. 不同前綴）。
