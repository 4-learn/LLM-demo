"""第 42 節：LoRA 低秩權重與訓練紀錄判讀（只用 Python 標準函式庫，不訓練任何模型）。

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
    print("  選 checkpoint 看 validation，不看最後一步，也不看 test（第 41 節的 manifest）。")


if __name__ == "__main__":
    main()
