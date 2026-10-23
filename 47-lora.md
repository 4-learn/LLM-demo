# LoRA 低秩權重與訓練紀錄判讀

> **新版正文：20E07-E03-05｜對應新版 LLM 第 43 節：LoRA 低秩權重與教師訓練紀錄。**
> LoRA 不改 base 模型，只在指定的線性層旁邊加一個低秩更新 `ΔW = B A`。本節先用小矩陣看 rank 與參數量，再用 Qwen2.5-0.5B 的**真實 config** 算不同設定要訓練多少參數，最後判讀三份訓練紀錄。**本節不訓練任何模型**；三份紀錄是**為教學編的合成資料**，不是實訓結果。

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先修第 05 節（config 與參數量）、第 42 節（train／validation／test 與 manifest）。

| 時間 | 內容 |
| --- | --- |
| 0–10 分鐘 | 凍結 base、低秩更新、adapter 檔是什麼 |
| 10–20 分鐘 | rank 與 target modules 怎麼決定可訓練參數 |
| 20–28 分鐘 | 讀訓練紀錄：train／validation loss、checkpoint、記憶體 |
| 28–35 分鐘 | 跑 Demo |
| 35–50 分鐘 | Workshop：手算參數量、調判讀門檻 |

驗收能力：交付**參數量計算**與**訓練紀錄判讀**；能說明為什麼選 checkpoint 要看 validation。

> **本節不需要訓練任何模型**：學生與教師都只手算參數量、判讀一份事先備妥的訓練紀錄。課綱提到教師機器（M4 Pro）；示範機 4-learn 只有 8 GB、沒有 GPU，不適合現場訓練。教師若想額外示範訓練，要先確認 RAM、選定模型並做 smoke test；本節不保證任何框架、MLX 或 GGUF 之間可以互轉。

## 一、ΔW = B A

一個線性層的權重 `W` 是「輸入維度 × 輸出維度」的矩陣。完整微調要更新 `W` 的每一個數。LoRA 把 `W` **凍結**，另外學一個更新：

```text
W' = W + ΔW，  ΔW = B A
B：輸出 × r，  A：r × 輸入，  r 遠小於輸入與輸出
```

- **可訓練參數**從「輸入 × 輸出」變成「r × (輸入＋輸出)」。
- **ΔW 的 rank 不會超過 r**：這是「低秩」的意思，也是它的限制——能表達的更新變少了。
- **B 初始化為 0**：訓練開始時 ΔW 是 0，加上 adapter 的模型和 base 完全相同，從 base 的行為出發。
- **adapter 檔**只存 A、B（與設定），所以很小；使用時要搭配**同一個 base 與 revision**。

## 二、target modules

Transformer 每一層有好幾個線性層：注意力的 `q_proj`、`k_proj`、`v_proj`、`o_proj`，MLP 的 `gate_proj`、`up_proj`、`down_proj`。LoRA 要選加在哪幾個上（target modules）。選越多、r 越大，可訓練參數越多，能學的東西也越多；但也越容易過擬合，adapter 也越大。

**GQA 會影響計算**：Qwen2.5-0.5B 有 14 個 query head，但只有 2 個 KV head（第 05 節），所以 `k_proj`、`v_proj` 的輸出只有 128 維，不是 896。

## 三、訓練紀錄要看什麼

| 欄位 | 看什麼 |
| --- | --- |
| train loss | 有沒有在下降；突然暴增（尖峰）或變成 NaN 是不穩定 |
| validation loss | **選 checkpoint 只看它**；它開始持續上升而 train 還在降，就是過擬合 |
| 記憶體峰值 | 能不能在目標機器上重現；要記硬體與框架版本 |
| checkpoint | 存了哪些 step；最佳 validation 的那一個有沒有存到 |

test 不能拿來選 checkpoint——那等於拿考卷調答案（第 42 節）。

## 四、可執行 Demo

<!-- demo: E08_lora_math.py -->
```python
"""第 43 節：LoRA 低秩權重與訓練紀錄判讀（只用 Python 標準函式庫，不訓練任何模型）。

一、用小矩陣看 Delta W = B A：rank、參數量、B 為何初始化為 0。
二、用 Qwen2.5-0.5B 的真實 config（第 05 節的快照）算不同 target modules 與 rank 的可訓練參數。
三、判讀三份**合成**訓練紀錄：正常、過擬合、不穩定。合成紀錄不是實訓結果。
"""

import json
import math
from fractions import Fraction
from pathlib import Path

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def rank(matrix):
    """高斯消去法求 rank；用 Fraction 避免浮點誤差。"""
    rows = [[Fraction(x) for x in row] for row in matrix]
    r = 0
    for col in range(len(rows[0])):
        pivot = next((i for i in range(r, len(rows)) if rows[i][col] != 0), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i][col] != 0:
                factor = rows[i][col] / rows[r][col]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[r])]
        r += 1
    return r


def load_config():
    here = Path(__file__).resolve().parent
    for base in (here, Path.cwd()):
        path = base / "data" / "model-candidates.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            entry = next(c for c in data["candidates"] if c["id"] == MODEL)
            return entry["config"], entry["official_parameter_count"], entry["revision"], data["fetched_at"]
    raise FileNotFoundError("找不到 data/model-candidates.json（第 05 節的快照）")


def linear_shapes(config):
    """每一層的線性層 (輸入維度, 輸出維度)。GQA：k、v 的輸出只有 kv_heads × head_dim。"""
    h, inter = config["hidden_size"], config["intermediate_size"]
    head_dim = h // config["num_attention_heads"]
    kv = config["num_key_value_heads"] * head_dim
    return {"q_proj": (h, h), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (h, h),
            "gate_proj": (h, inter), "up_proj": (h, inter), "down_proj": (inter, h)}


TARGETS = {
    "q,v": ("q_proj", "v_proj"),
    "注意力 4 個": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "全部線性層 7 個": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
}


def lora_params(config, targets, r):
    shapes = linear_shapes(config)
    return config["num_hidden_layers"] * sum(r * (shapes[m][0] + shapes[m][1]) for m in targets)


# ---- 三份合成訓練紀錄（step, train_loss, val_loss）；數字是為教學編的 ----
LOGS = {
    "R1 正常": [(100, 2.10, 2.15), (200, 1.62, 1.70), (300, 1.41, 1.52), (400, 1.30, 1.46),
               (500, 1.24, 1.44), (600, 1.20, 1.43), (700, 1.17, 1.43), (800, 1.15, 1.44)],
    "R2 過擬合": [(100, 2.05, 2.12), (200, 1.48, 1.63), (300, 1.02, 1.49), (400, 0.71, 1.52),
                 (500, 0.45, 1.61), (600, 0.27, 1.74), (700, 0.15, 1.90), (800, 0.08, 2.05)],
    "R3 不穩定": [(100, 2.08, 2.14), (200, 1.55, 1.66), (300, 4.90, 4.71), (400, 1.71, 1.83),
                 (500, 1.49, 1.62), (600, 6.30, 6.02), (700, float("nan"), float("nan")), (800, float("nan"), float("nan"))],
}
RISE = 3           # val loss 在最佳點之後連續上升幾次評估，就視為過擬合
SPIKE = 2.0        # train loss 超過前一次的幾倍，視為尖峰


def diagnose(log):
    finite = [(s, t, v) for s, t, v in log if not (math.isnan(t) or math.isnan(v))]
    best = min(finite, key=lambda row: row[2])
    after = [v for s, _, v in finite if s > best[0]]
    rises = 0
    for prev, cur in zip([best[2]] + after, after):
        rises = rises + 1 if cur > prev else 0
        if rises >= RISE:
            break
    spikes = [s for (_, t0, _), (s, t1, _) in zip(log, log[1:]) if not math.isnan(t1) and t1 > SPIKE * t0]
    nan_at = next((s for s, t, v in log if math.isnan(t) or math.isnan(v)), None)
    last = log[-1]
    findings = []
    if rises >= RISE:
        findings.append(f"過擬合：val 在 step {best[0]} 後連續上升 {RISE} 次")
    if spikes:
        findings.append(f"loss 尖峰於 step {', '.join(map(str, spikes))}")
    if nan_at:
        findings.append(f"step {nan_at} 起出現 NaN")
    return {"best_step": best[0], "best_val": best[2], "last": last,
            "findings": findings or ["未見過擬合或不穩定徵兆"]}


def main():
    print("=== 一、小矩陣：Delta W = B A ===")
    B = [[1, 0], [2, 1], [0, 3], [1, 1], [3, 2], [0, 1]]           # 6 × 2
    A = [[1, 2, 0, 1, 0, 2, 1, 0], [0, 1, 1, 0, 2, 0, 1, 3]]     # 2 × 8
    delta = matmul(B, A)
    print(f"  W 是 6×8：完整更新要訓練 {6 * 8} 個數；r=2 的 B(6×2)、A(2×8) 只要 {6 * 2 + 2 * 8} 個")
    print(f"  B A 的 rank＝{rank(delta)}（不會超過 r）；第 1 列 {delta[0]}")
    zero_b = [[0, 0] for _ in range(6)]
    print(f"  訓練開始時 B 全 0 → Delta W 全 0：{all(x == 0 for row in matmul(zero_b, A) for x in row)}"
          "（加上 adapter 的模型一開始與 base 完全相同）")
    print()

    config, total, revision, fetched = load_config()
    print(f"=== 二、{MODEL}@{revision[:7]}（config 快照 {fetched[:10]}）===")
    shapes = linear_shapes(config)
    print(f"  每層線性層（輸入→輸出）：" + "、".join(f"{m} {i}→{o}" for m, (i, o) in shapes.items()))
    print(f"  官方參數量 {total:,}；層數 {config['num_hidden_layers']}")
    print(f"  {'r':>3} {'可訓練參數':>11} {'占比':>6} {'adapter bf16':>12}  target modules")
    for label, targets in TARGETS.items():
        for r in (8, 16, 64):
            n = lora_params(config, targets, r)
            print(f"  {r:>3} {n:>16,} {n / total:>7.2%} {n * 2 / 2**20:>9.1f} MiB  {label}")
    print("  回本 rank（r × (輸入＋輸出) 等於完整矩陣時）：" +
          "、".join(f"{m} {i * o // (i + o)}" for m, (i, o) in shapes.items() if m in ("q_proj", "k_proj", "gate_proj")))
    print("  可訓練參數少，不等於訓練記憶體少同樣比例：前向與反向仍要經過整個 base。")
    print()

    print("=== 三、合成訓練紀錄判讀（數字為教學編造，不是實訓）===")
    for name, log in LOGS.items():
        d = diagnose(log)
        last = d["last"]
        last_text = f"train {last[1]}／val {last[2]}"
        print(f"  {name}：最佳 val {d['best_val']} @ step {d['best_step']}｜最後一步 {last_text}")
        for finding in d["findings"]:
            print(f"    - {finding}")
    print("  選 checkpoint 看 validation，不看最後一步，也不看 test（第 42 節的 manifest）。")


if __name__ == "__main__":
    main()
```

執行指令（在 [`llm-demo`](https://github.com/4-learn/llm-demo) 目錄下；讀 `data/model-candidates.json`，**不需要網路、不需要模型**）：

```bash
python3 E08_lora_math.py
```

關鍵輸出（Linux／Python 3.10.15，20E07-E03-05）：

```text
=== 一、小矩陣：Delta W = B A ===
  W 是 6×8：完整更新要訓練 48 個數；r=2 的 B(6×2)、A(2×8) 只要 28 個
  B A 的 rank＝2（不會超過 r）；第 1 列 [1, 2, 0, 1, 0, 2, 1, 0]
  訓練開始時 B 全 0 → Delta W 全 0：True（加上 adapter 的模型一開始與 base 完全相同）

=== 二、Qwen/Qwen2.5-0.5B-Instruct@7ae5576（config 快照 20E07-E03-05）===
  每層線性層（輸入→輸出）：q_proj 896→896、k_proj 896→128、v_proj 896→128、o_proj 896→896、gate_proj 896→4864、up_proj 896→4864、down_proj 4864→896
  官方參數量 494,032,768；層數 24
    r       可訓練參數     占比 adapter bf16  target modules
    8          540,672   0.11%       1.0 MiB  q,v
   16        1,081,344   0.22%       2.1 MiB  q,v
   64        4,325,376   0.88%       8.2 MiB  q,v
    8        1,081,344   0.22%       2.1 MiB  注意力 4 個
   16        2,162,688   0.44%       4.1 MiB  注意力 4 個
   64        8,650,752   1.75%      16.5 MiB  注意力 4 個
    8        4,399,104   0.89%       8.4 MiB  全部線性層 7 個
   16        8,798,208   1.78%      16.8 MiB  全部線性層 7 個
   64       35,192,832   7.12%      67.1 MiB  全部線性層 7 個
  回本 rank（r × (輸入＋輸出) 等於完整矩陣時）：q_proj 448、k_proj 112、gate_proj 756
  可訓練參數少，不等於訓練記憶體少同樣比例：前向與反向仍要經過整個 base。

=== 三、合成訓練紀錄判讀（數字為教學編造，不是實訓）===
  R1 正常：最佳 val 1.43 @ step 600｜最後一步 train 1.15／val 1.44
    - 未見過擬合或不穩定徵兆
  R2 過擬合：最佳 val 1.49 @ step 300｜最後一步 train 0.08／val 2.05
    - 過擬合：val 在 step 300 後連續上升 3 次
  R3 不穩定：最佳 val 1.62 @ step 500｜最後一步 train nan／val nan
    - loss 尖峰於 step 300, 600
    - step 700 起出現 NaN
  選 checkpoint 看 validation，不看最後一步，也不看 test（第 42 節的 manifest）。
```

## 五、結果怎麼讀

### 1. 小矩陣

6×8 的完整更新是 48 個數，r=2 只要 28 個——在這麼小的矩陣上省得不多。B A 的 rank 是 2：不論訓練出什麼，這個更新都只有兩個「方向」。B 全 0 時 ΔW 全 0，模型一開始和 base 相同。

### 2. Qwen2.5-0.5B

- 只加在 q、v 上、r=8，可訓練參數是 **540,672**，約占全模型的 **0.11%**，adapter 用 bf16 存約 1 MiB。
- 加在全部 7 個線性層、r=64，就是 **35,192,832**（**7.12%**）、約 67 MiB。設定不同，數字差了 65 倍；**報告一定要寫 target modules 與 r**，只寫「用了 LoRA」沒有意義。
- 「回本 rank」是 r × (輸入＋輸出) 等於完整矩陣的那個 r：`k_proj` 是 112。r 到了這個值，低秩就不省參數了。
- **參數少不等於記憶體少同樣比例**：前向與反向都要經過整個 base，啟用值（activations）也要存。訓練記憶體要實測，不能從 0.11% 推算。

### 3. 合成訓練紀錄

| 紀錄 | 判讀 | 該選哪個 checkpoint |
| --- | --- | --- |
| R1 | val 在 step 600 最低（1.43），之後持平；正常 | step 600 |
| R2 | train 一路降到 0.08，val 從 step 300 後連續上升到 2.05；**過擬合** | step 300；之後的 checkpoint 都不能用 |
| R3 | step 300、600 的 train loss 暴增，step 700 起變成 NaN；**不穩定** | step 500 的 val 最低，但這次訓練本身要查原因（學習率、資料、精度）再重跑 |

R2 的最後一步 train loss 0.08 看起來「很好」，但它只是把訓練資料背下來了。**只看 train loss、或只看最後一步，會選到最差的 checkpoint**。

## 六、常見誤判

| 誤判 | 為什麼錯 |
| --- | --- |
| 「LoRA 只訓練 0.1% 參數，記憶體也只要 0.1%」 | 前向與反向仍經過整個 base；要實測 |
| 「r 越大越好」 | 參數變多、更容易過擬合；r 到回本值就不省了 |
| 「k_proj 和 q_proj 一樣大」 | GQA 讓 k、v 只有 128 維輸出 |
| 「train loss 最低的 checkpoint 最好」 | 可能是過擬合；看 validation |
| 「用 test 選 checkpoint 比較準」 | 那是拿考卷調答案；test 只在最後用一次 |
| 「adapter 可以搭配任何版本的 base」 | A、B 是對特定權重學的；base 與 revision 要一致 |
| 「合成紀錄的判讀等於實訓經驗」 | 本節的紀錄是編的；真實紀錄的雜訊與徵兆可能更模糊 |

## Workshop：手算參數量、調判讀門檻

### 任務

以下每題都**重新載入原始程式**後單獨修改，不要累積前一題的修改。

1. 手算：只加在 q、v 上、r=4，Qwen2.5-0.5B 的可訓練參數是多少？寫出算式，再用 `lora_params` 驗證。
2. 把 config 的 `num_key_value_heads` 改成 14（假裝沒有 GQA），q、v、r=8 的可訓練參數變成多少？為什麼會變？
3. 把 `RISE` 改成 5，R2 還會被判成過擬合嗎？改成 1，R1 會怎樣？門檻要怎麼定？
4. 把 `SPIKE` 改成 4.0，R3 還抓得到哪些尖峰？
5. 寫一份判讀報告：三份紀錄各選哪個 checkpoint、理由、你用的門檻；並註明「合成紀錄，非實訓」。

### 參考判讀

以下每一題都是**重新載入程式後**單獨修改、實際執行的結果。

- 第 1 題：**270,336**。每層 q 是 4 × (896＋896)＝7,168，v 是 4 × (896＋128)＝4,096，合計 11,264；24 層 × 11,264＝270,336。
- 第 2 題：**688,128**（原本 540,672）。沒有 GQA 時 `v_proj` 的輸出是 896，不是 128。**算參數量要看 config，不能假設每個投影都一樣大**。
- 第 3 題：`RISE=5` 時 R2 仍然是過擬合（step 300 之後正好連續上升 5 次）。`RISE=1` 時 R1 也被判成過擬合——step 800 的 val 只比 step 700 高 0.01。門檻太低會把雜訊當成過擬合，太高會太晚發現；要依評估的間隔和 val 的雜訊來定，並寫進報告。
- 第 4 題：只剩 **step 600**（6.30／1.49 約 4.2 倍）；step 300 的 4.90／1.55 約 3.2 倍，抓不到了。NaN 仍然抓得到。
- 第 5 題：沒有標準答案。

### 驗收

- 參數量計算寫出算式（含 GQA 下 k、v 的維度），並與 `lora_params` 一致。
- 三份紀錄各選一個 checkpoint，理由引用 validation，不引用 train 或 test。
- 判讀門檻寫在報告裡。
- 標明證據等級：合成紀錄、非實訓。

## 教師紀錄與下一步

- 本節**不需要模型、網路或任何套件**，不訓練。參數量用第 05 節的 config 快照計算，證據等級：**本機實測（由公開 config 計算）**；三份訓練紀錄是**合成資料**，不得寫成實訓結果。教師的真實訓練紀錄要等 RAM 確認、模型選定與 smoke test 之後才有，屬**待驗證**。
- 參數量只算 LoRA 的 A、B，不含 bias，也不含 PEFT 等函式庫可能額外加的參數；函式庫實際印出的數字要以實測為準。
- 撰寫時的實際紀錄：① 參數量表第一版把中文的 target modules 標籤放在第一欄，中文字寬度讓欄位對不齊；改把標籤移到最後一欄。② `diagnose` 第一版回傳一個沒有用到的 `gap` 欄位，已刪除。
- 第 44 節接上：用第 42 節的 test manifest 比較微調前後；adapter 交付要附 base revision、target modules、r 與訓練紀錄。
