"""第 07 節：評測集、指標與可重現基準（只用 Python 標準函式庫）。

讀 `data/eval-set.json`：25 筆已標註的合成巡查通報，切分固定在檔案裡。
比較三個規則版本，並分開報開發集與保留集的成績——因為只有保留集能回答
「你是學會了，還是背下來了」。
"""

import json
from pathlib import Path

LABELS = ("missing", "present", "unknown")
SAFETY_CRITICAL = "missing"      # 漏掉一次未戴，比多標一次嚴重

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


def strip_negated_missing(text):
    """把「並非未戴」這類雙重否定拿掉，剩下的才是它真正在講的事。"""
    for pattern in NEGATED_MISSING:
        text = text.replace(pattern, "")
    return text


def classify_naive(text):
    """v1：看到「未戴」就說 missing，先比對再說。"""
    if any(marker in text for marker in MISSING_MARKERS):
        return "missing"
    if any(marker in text for marker in PRESENT_MARKERS):
        return "present"
    return "unknown"


def classify_ordered(text):
    """v2：先排除指令與不確定，再處理否定，最後才比對關鍵字。"""
    if any(marker in text for marker in INSTRUCTION_MARKERS):
        return "unknown"
    if any(marker in text for marker in UNCERTAIN_MARKERS):
        return "unknown"
    stripped = strip_negated_missing(text)
    if any(marker in stripped for marker in MISSING_MARKERS):
        return "missing"
    if any(marker in stripped for marker in PRESENT_MARKERS):
        return "present"
    return "unknown"


def make_memoriser(items):
    """v3：把開發集整句背下來。沒看過的一律 unknown。"""
    table = {item["text"]: item["gold"] for item in items if item["split"] == "dev"}

    def classify(text):
        return table.get(text, "unknown")
    return classify


def evaluate(items, classify):
    return [(item, classify(item["text"])) for item in items]


def metrics(rows):
    total = len(rows)
    correct = sum(1 for item, guess in rows if guess == item["gold"])
    answered = sum(1 for _, guess in rows if guess != "unknown")
    per_label = {}
    for label in LABELS:
        tp = sum(1 for item, guess in rows if item["gold"] == label and guess == label)
        fp = sum(1 for item, guess in rows if item["gold"] != label and guess == label)
        fn = sum(1 for item, guess in rows if item["gold"] == label and guess != label)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        per_label[label] = (tp, fp, fn, precision, recall)
    return {
        "accuracy": correct / total,
        "coverage": answered / total,
        "per_label": per_label,
        "rows": rows,
    }


def fmt(value):
    return "  n/a" if value is None else f"{value:5.2f}"


def main():
    data = load_eval_set()
    items = data["items"]
    dev = [item for item in items if item["split"] == "dev"]
    holdout = [item for item in items if item["split"] == "holdout"]
    print(f"評測集 {data['dataset_version']}／標註 {data['annotation_version']}："
          f"{len(items)} 筆（開發 {len(dev)}、保留 {len(holdout)}）")
    print(f"安全關鍵類別：{SAFETY_CRITICAL}（漏標比多標嚴重）")
    print()

    variants = [
        ("v1 天真關鍵字", classify_naive),
        ("v2 否定與指令優先", classify_ordered),
        ("v3 死記開發集", make_memoriser(items)),
    ]

    header = (f"{'版本':<20} {'正確率':>7} {'覆蓋率':>7} "
              f"{SAFETY_CRITICAL + ' P':>11} {SAFETY_CRITICAL + ' R':>11}")
    print("=== 開發集（可以用來調規則）===")
    print(header)
    dev_scores = {}
    for name, function in variants:
        result = metrics(evaluate(dev, function))
        dev_scores[name] = result["accuracy"]
        _, _, _, precision, recall = result["per_label"][SAFETY_CRITICAL]
        print(f"{name:<20} {result['accuracy']:6.1%} {result['coverage']:6.1%} "
              f"{fmt(precision):>11} {fmt(recall):>11}")

    best = max(dev_scores.values())
    tied = [name for name, score in dev_scores.items() if score == best]
    print(f"\n→ 只看開發集，第一名有 {len(tied)} 個並列：{'、'.join(tied)}")
    print("   開發集分不出「學會了」和「背下來了」。")
    print()

    print("=== 保留集（只能看一次）===")
    print(header)
    for name, function in variants:
        result = metrics(evaluate(holdout, function))
        _, _, _, precision, recall = result["per_label"][SAFETY_CRITICAL]
        print(f"{name:<20} {result['accuracy']:6.1%} {result['coverage']:6.1%} "
              f"{fmt(precision):>11} {fmt(recall):>11}")
    print()

    print("=== v2 在保留集的分組表現 ===")
    rows = evaluate(holdout, classify_ordered)
    for group in data["groups"]:
        group_rows = [row for row in rows if row[0]["group"] == group]
        if not group_rows:
            continue
        correct = sum(1 for item, guess in group_rows if guess == item["gold"])
        print(f"  {group:<8} {correct}/{len(group_rows)}")
    print()

    print("=== v1 在保留集錯在哪 ===")
    for item, guess in evaluate(holdout, classify_naive):
        if guess != item["gold"]:
            print(f"  {item['id']} [{item['group']}] gold={item['gold']:<7} v1={guess:<7} {item['text']}")


if __name__ == "__main__":
    main()
