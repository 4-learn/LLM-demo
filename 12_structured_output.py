"""第 12 節：Structured Output——語法、Schema 與語義（只用 Python 標準函式庫）。

三層驗證缺一不可：
  第 1 層 語法：這段字串是不是合法 JSON？
  第 2 層 Schema：它的形狀對不對？
  第 3 層 語義：裡面的內容有沒有來源支持？

**第 2 層通過不代表第 3 層會過。** 這是本節最重要的一句話。

候選輸出是教學用的固定字串，不是任何模型的即時回應。
"""

import json

SOURCE = (
    "巡檢紀錄 R-2026-1005-01：三號出口堆放紙箱，已於 14:30 通知清除。"
    "現場未發現人員受傷。安全帽配戴正常。"
)

SCHEMA = {
    "type": "object",
    "required": ["report_id", "hazard", "severity", "resolved", "workers_injured"],
    "properties": {
        "report_id": {"type": "string"},
        "hazard": {"type": "string", "enum": ["blocked_exit", "no_helmet", "none"]},
        "severity": {"type": "string", "enum": ["low", "medium", "high", "unknown"]},
        "resolved": {"type": "boolean"},
        "workers_injured": {"type": "integer", "minimum": 0},
        "note": {"type": ["string", "null"]},
    },
}

CANDIDATES = (
    ("A 合法且正確",
     '{"report_id":"R-2026-1005-01","hazard":"blocked_exit","severity":"unknown",'
     '"resolved":true,"workers_injured":0,"note":null}'),
    ("B Schema 通過、內容錯誤",
     '{"report_id":"R-2026-1005-01","hazard":"no_helmet","severity":"high",'
     '"resolved":true,"workers_injured":0}'),
    ("C 被截斷",
     '{"report_id":"R-2026-1005-01","hazard":"blocked_ex'),
    ("D 形狀不對（缺必填）",
     '{"hazard":"blocked_exit"}'),
    ("E 拒絕回答",
     '無法從這份紀錄判斷，請提供更完整的巡檢表。'),
)

# 語義層用的關鍵字。這裡刻意寫成「主張 → 來源必須出現的字串」，
# 讓人一眼看得出這層是規則，不是模型。
SUPPORT = {
    ("hazard", "blocked_exit"): ("堆放", "阻擋", "佔用"),
    ("hazard", "no_helmet"): ("未戴", "沒戴", "沒有戴"),
    ("hazard", "none"): ("正常", "無異常", "未發現異常"),
    ("resolved", True): ("通知清除", "已清除", "已改善"),
    ("workers_injured", 0): ("未發現人員受傷", "無人員受傷", "無人受傷"),
    # 來源沒有提到嚴重程度，所以任何非 unknown 的判斷都沒有依據。
    ("severity", "low"): ("輕微",),
    ("severity", "medium"): ("中度",),
    ("severity", "high"): ("嚴重", "重大"),
}


def type_ok(value, expected):
    mapping = {"string": str, "boolean": bool, "integer": int, "number": (int, float),
               "object": dict, "array": list, "null": type(None)}
    if expected == "integer" and isinstance(value, bool):
        return False                      # Python 的 bool 是 int 的子類，要排掉
    if expected == "number" and isinstance(value, bool):
        return False
    return isinstance(value, mapping[expected])


def validate_schema(instance, schema):
    """回傳錯誤清單。空清單＝形狀合規。故意寫得比 JSON Schema 小很多。"""
    errors = []
    if schema.get("type") == "object":
        if not isinstance(instance, dict):
            return ["根節點必須是物件"]
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"缺少必填欄位 {key}")
        for key, value in instance.items():
            rule = schema.get("properties", {}).get(key)
            if rule is None:
                errors.append(f"出現未定義欄位 {key}")
                continue
            expected = rule["type"]
            expected = expected if isinstance(expected, list) else [expected]
            if not any(type_ok(value, item) for item in expected):
                errors.append(f"{key} 型別應為 {'/'.join(expected)}，實際 {type(value).__name__}")
                continue
            if "enum" in rule and value not in rule["enum"]:
                errors.append(f"{key} 的值 {value!r} 不在允許清單 {rule['enum']}")
            if "minimum" in rule and isinstance(value, (int, float)) and value < rule["minimum"]:
                errors.append(f"{key}={value} 小於下限 {rule['minimum']}")
    return errors


def check_semantics(record):
    """回傳沒有來源支持的主張。"""
    unsupported = []
    if not isinstance(record, dict):
        return ["不是物件，無法檢查語義"]
    for (field, expected), keywords in SUPPORT.items():
        if record.get(field) != expected:
            continue
        if not any(keyword in SOURCE for keyword in keywords):
            unsupported.append(f"{field}={expected!r} 在來源中找不到支持（找過 {keywords}）")
    report_id = record.get("report_id")
    if isinstance(report_id, str) and report_id not in SOURCE:
        unsupported.append(f"report_id={report_id!r} 沒有出現在來源中（疑似編造）")
    return unsupported


def run(text):
    try:
        record = json.loads(text)
    except json.JSONDecodeError as error:
        return {"parsed": False, "parse_error": f"{error.msg}（位置 {error.pos}）",
                "schema_errors": None, "unsupported": None, "record": None}
    return {"parsed": True, "parse_error": None, "record": record,
            "schema_errors": validate_schema(record, SCHEMA),
            "unsupported": check_semantics(record)}


def main():
    print(f"來源（{len(SOURCE)} 字）：{SOURCE}")
    print()
    print("=== 三層驗證 ===")
    print(f"{'候選':<26} {'1 語法':<8} {'2 Schema':<10} {'3 語義':<10}")
    results = {}
    for label, text in CANDIDATES:
        outcome = run(text)
        results[label] = outcome
        syntax = "通過" if outcome["parsed"] else "失敗"
        schema = "—" if not outcome["parsed"] else ("通過" if not outcome["schema_errors"] else "失敗")
        semantic = "—" if not outcome["parsed"] or outcome["schema_errors"] else (
            "通過" if not outcome["unsupported"] else "失敗")
        print(f"{label:<26} {syntax:<8} {schema:<10} {semantic:<10}")
    print()

    print("=== 失敗原因 ===")
    for label, _ in CANDIDATES:
        outcome = results[label]
        if not outcome["parsed"]:
            print(f"{label}：第 1 層 {outcome['parse_error']}")
        elif outcome["schema_errors"]:
            for error in outcome["schema_errors"]:
                print(f"{label}：第 2 層 {error}")
        elif outcome["unsupported"]:
            for problem in outcome["unsupported"]:
                print(f"{label}：第 3 層 {problem}")
        else:
            print(f"{label}：三層全部通過")
    print()

    print("=== 把 C 補成合法 JSON 會怎樣？ ===")
    truncated = CANDIDATES[2][1]
    model_gave = ["report_id", "hazard"]          # 截斷前只有這兩個欄位出現過
    repaired = ('{"report_id":"R-2026-1005-01","hazard":"blocked_exit","severity":"unknown",'
                '"resolved":true,"workers_injured":0}')
    record = json.loads(repaired)
    invented = [key for key in record if key not in model_gave]
    outcome = run(repaired)
    print(f"C 原文（截斷，{len(truncated)} 字）：{truncated}")
    print(f"補完後的欄位：{sorted(record)}")
    print(f"其中模型沒說過的：{sorted(invented)}（{len(invented)}/{len(record)}）")
    print(f"補完後三層驗證：語法 {'通過' if outcome['parsed'] else '失敗'}／"
          f"Schema {'通過' if not outcome['schema_errors'] else '失敗'}／"
          f"語義 {'通過' if not outcome['unsupported'] else '失敗'}")
    print("三層全過，但這份結果已經回答不了「模型到底說了什麼」——")
    print("補進去的是**你的判斷**，不是模型的輸出，而驗證器看不出差別。")


if __name__ == "__main__":
    main()
