"""第 47 節：模型路由、級聯與品質門檻（只用 Python 標準函式庫）。

兩層「模型」都是第 07 節的確定性規則，**不是語言模型**：
  - 便宜層 = v1 天真關鍵字（每筆成本 1 單位）
  - 昂貴層 = v2 否定與指令優先（每筆成本 20 單位）
成本單位是教學假設，不是任何供應商的價格。

問題：能不能大部分交給便宜層，只把「可能出錯」的升級？
關鍵在**用什麼判斷「可能出錯」**——便宜層自己的信心，還是外部的門檻訊號。
"""

import json
from pathlib import Path

CHEAP_COST, EXPENSIVE_COST = 1, 20
SAFETY_CRITICAL = "missing"

# 與第 07 節相同的關鍵字（測試會檢查兩邊一致，避免各自漂移）
MISSING_MARKERS = ("未戴", "沒戴", "沒有戴")
PRESENT_MARKERS = ("已戴", "配戴", "戴好", "合規")
UNCERTAIN_MARKERS = ("模糊", "不確定", "無法確認", "看不清楚", "無法逐一", "可能", "輪廓", "重疊")
INSTRUCTION_MARKERS = ("忽略", "系統提示", "管理員", "請輸出", "不要標成", "你現在是")
NEGATED_MISSING = ("並非未戴", "不是未戴", "無未戴", "沒有未戴")


def load_eval_set():
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        path = base / "data" / "eval-set.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("找不到 data/eval-set.json")


def cheap(text):
    """便宜層：回傳 (標籤, 自述信心)。只要有比到關鍵字，它就說自己很有把握。"""
    if any(m in text for m in MISSING_MARKERS):
        return "missing", 0.95
    if any(m in text for m in PRESENT_MARKERS):
        return "present", 0.95
    return "unknown", 0.50


def expensive(text):
    if any(m in text for m in INSTRUCTION_MARKERS + UNCERTAIN_MARKERS):
        return "unknown"
    for pattern in NEGATED_MISSING:
        text = text.replace(pattern, "")
    if any(m in text for m in MISSING_MARKERS):
        return "missing"
    if any(m in text for m in PRESENT_MARKERS):
        return "present"
    return "unknown"


# 三種「要不要升級」的門檻
def gate_self_confidence(text, label, confidence, threshold=0.9):
    """信便宜層自己：信心低於門檻才升級。"""
    return confidence < threshold


def gate_risk_signals(text, label, confidence):
    """外部訊號：便宜層答 unknown，或句子裡有否定／不確定／指令字樣，就升級。"""
    risky = INSTRUCTION_MARKERS + UNCERTAIN_MARKERS + NEGATED_MISSING
    return label == "unknown" or any(m in text for m in risky)


def gate_safety_first(text, label, confidence):
    """外部訊號＋安全關鍵類別一律覆核：便宜層說 missing 也要升級。"""
    return gate_risk_signals(text, label, confidence) or label == SAFETY_CRITICAL


POLICIES = (
    ("全部便宜層", None),
    ("全部昂貴層", "all"),
    ("級聯：信自述信心", gate_self_confidence),
    ("級聯：外部風險訊號", gate_risk_signals),
    ("級聯：風險＋missing 覆核", gate_safety_first),
)


def route(items, gate):
    rows, cost, escalated = [], 0, 0
    for item in items:
        if gate == "all":
            rows.append((item, expensive(item["text"])))
            cost += EXPENSIVE_COST
            escalated += 1
            continue
        label, confidence = cheap(item["text"])
        cost += CHEAP_COST
        if gate is not None and gate(item["text"], label, confidence):
            label = expensive(item["text"])
            cost += EXPENSIVE_COST           # 級聯的代價：升級的那幾筆付了兩次
            escalated += 1
        rows.append((item, label))
    return rows, cost, escalated


def score(rows):
    correct = sum(1 for item, guess in rows if guess == item["gold"])
    gold_missing = [guess for item, guess in rows if item["gold"] == SAFETY_CRITICAL]
    recall = sum(1 for g in gold_missing if g == SAFETY_CRITICAL) / len(gold_missing)
    wrong = [item["id"] for item, guess in rows if guess != item["gold"]]
    return correct / len(rows), recall, wrong


def table(title, items):
    print(f"=== {title}（{len(items)} 筆）===")
    print(f"  {'政策':<22} {'準確率':>6} {'missing召回':>10} {'升級':>5} {'成本':>5} {'相對全昂貴':>9}  錯題")
    full = EXPENSIVE_COST * len(items)
    results = {}
    for name, gate in POLICIES:
        rows, cost, escalated = route(items, gate)
        acc, recall, wrong = score(rows)
        results[name] = (acc, recall, escalated, cost, wrong)
        print(f"  {name:<22} {acc:>6.0%} {recall:>10.2f} {escalated:>3}/{len(items):<2} {cost:>5}"
              f" {cost / full:>9.0%}  {' '.join(wrong) or '—'}")
    print()
    return results


def main():
    data = load_eval_set()
    dev = [i for i in data["items"] if i["split"] == "dev"]
    holdout = [i for i in data["items"] if i["split"] == "holdout"]
    print(f"評測集 {data['dataset_version']}：開發 {len(dev)}、保留 {len(holdout)}"
          f"｜成本假設 便宜 {CHEAP_COST}、昂貴 {EXPENSIVE_COST}（教學單位）")
    print()
    table("開發集：用來選政策", dev)
    table("保留集：選定之後只看一次", holdout)

    print("=== 便宜層「很有把握」的錯誤 ===")
    for item in data["items"]:
        label, confidence = cheap(item["text"])
        if confidence >= 0.9 and label != item["gold"]:
            print(f"  {item['id']} 信心 {confidence:.2f} 答 {label:<7} 正解 {item['gold']:<7} {item['text']}")
    print()
    worst = len(holdout) * (CHEAP_COST + EXPENSIVE_COST)
    print(f"最壞情況預算（保留集每筆都升級）：{worst}，比全部昂貴層的 {len(holdout) * EXPENSIVE_COST} 還多")


if __name__ == "__main__":
    main()
