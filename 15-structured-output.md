# Structured Output：語法、Schema 與語義

> **新版正文：20E07-E03-05｜對應新版 LLM 第 13 節：Structured Output——語法、Schema 與語義。**
> 本節要讓程式的輸入可以被信任。核心主張：**「是合法 JSON」「符合 schema」「內容有依據」是三件不同的事，而且前兩件通過完全不代表第三件。** 本節只用 Python 標準函式庫。

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先修第 11、12 節的圖文證據與來源頁碼。

| 時間 | 內容 |
| --- | --- |
| 0–8 分鐘 | 為什麼「模型說它回傳 JSON」不是保證 |
| 8–20 分鐘 | 第 1 層語法與第 2 層 Schema；nullable 與 unknown |
| 20–28 分鐘 | 第 3 層語義：schema 不保證事實正確 |
| 28–35 分鐘 | 跑 Demo，看一份 schema 全過但內容錯誤的輸出 |
| 35–50 分鐘 | Workshop：三份輸出各跑三層，保留反例 |

驗收能力：能交付**三層驗證結果與錯誤處理**，並能指出哪些欄位來自模型、哪些是後處理補上的。

## 一、三層驗證，缺一不可

| 層 | 問題 | 通過是什麼意思 | 誰能檢查 |
| --- | --- | --- | --- |
| **1 語法** | 這段字串是合法 JSON 嗎？ | 可以 `json.loads` | 解析器 |
| **2 Schema** | 形狀對不對？必填、型別、允許值 | 下游程式不會炸 | 驗證器 |
| **3 語義** | 內容有沒有依據？ | 這份資料值得相信 | **只有你** |

三層是**獨立的**。一份輸出可以 1、2 全過而 3 全錯——本節 Demo 的候選 `B` 就是。反過來，一份 1 失敗的輸出（例如被截斷）不代表模型講錯，它只是沒講完。

> 這是本節唯一要記住的一句話：**Schema 驗證的是形狀，不是事實。**

## 二、第 1 層：語法

三種常見的語法失敗，原因完全不同：

| 現象 | 原因 | 正確處理 |
| --- | --- | --- |
| 被截斷（`Unterminated string`） | 達到 `max_new_tokens` 或連線中斷 | **重試或縮小輸出**；不要自己補完 |
| 多了一段前言（`以下是 JSON：{...}`） | 模型把說明跟資料混在一起 | 用原生結構化輸出，或明確要求只輸出 JSON |
| 拒絕回答（整段是自然語言） | 模型判斷無法完成 | **保留為「拒絕」**，不要當成解析失敗重試 |

拒絕（refusal）是最容易被寫錯的一種：它是**合法結果**，不是錯誤。如果程式把拒絕當成解析失敗然後重試，你會把一個誠實的「我不知道」逼成一個編出來的答案。

## 三、第 2 層：Schema

Schema 檢查實際在檢查這幾件事：

| 檢查 | 例子 | 沒檢查會怎樣 |
| --- | --- | --- |
| 必填欄位存在 | `report_id` 不可缺 | 下游 `KeyError` |
| 型別正確 | `workers_injured` 要是整數 | 字串 `"0"` 讓加總變成串接 |
| 值在允許清單 | `hazard` 只能是三個值之一 | 出現 `"nohelmet"` 這種拼錯的類別，統計時靜默漏掉 |
| 數值範圍 | 受傷人數不可為負 | 出現 `-1` |
| 額外欄位 | 未定義的欄位要不要放行 | 模型自己發明欄位，你以為有這個欄位 |

有兩個實作細節值得記住：

- **`null` 和 `unknown` 不是同一件事。** `null` 是「這個欄位沒有值」，`unknown` 可能是「有這個欄位，但無法判定」。本節 Demo 的 schema 允許 `note` 為 `null`，而 `severity` 有一個明確的 `unknown` 選項——因為「嚴重程度未知」是這個任務的真實狀態，不是缺值。
- **Python 的 `bool` 是 `int` 的子類。** `isinstance(True, int)` 是 `True`，所以 `workers_injured: true` 會意外通過整數檢查。本節 Demo 明確排掉這種情況。

還有一點：**每家供應商的 schema 子集不一樣。** 你寫的 schema 在你測的那個模型上通過，不代表換一家還在。第 15 節會用能力矩陣處理這件事。

## 四、第 3 層：語義

語義層問的是：**每一個主張，在來源裡找不找得到支持？** 這層沒有通用工具，因為「支持」的定義取決於你的領域。

本節 Demo 把語義層寫成明確的規則（「主張 → 來源必須出現的字串」），讓人一眼看得出**這是規則，不是模型**。這樣寫有兩個好處：可稽核、可測試。但有個必須誠實面對的缺點：

> **規則會誤判，而且是雙向的。** 我在寫這一節時，第一版把 `resolved=true` 的支持字串寫成「已通知清除」，但來源寫的是「已於 14:30 **通知清除**」——**正確的輸出被判成錯的**。修正之後，規則又反過來可能把「未通知清除」誤判成支持。

所以語義層的規則本身也要有測試與覆核。**它不是真理，它是一組你願意為它負責的判斷。**

## 五、三種取得結構化輸出的方式

| 方式 | 怎麼做 | 保證到哪一層 | 什麼時候用 |
| --- | --- | --- | --- |
| **提示 JSON** | 在提示裡寫「請輸出 JSON」 | 只到第 1 層（而且不保證） | 教學、快速原型 |
| **原生結構化輸出** | 用供應商的 schema 參數（如 JSON mode／response schema） | 到第 2 層（依子集而定） | 正式用途 |
| **本地驗證** | 自己 parse ＋ 驗 schema ＋ 查來源 | 到第 3 層 | **一定要有** |

**前兩種都不能取代第三種。** 第 09 節的原則在這裡再次出現：**供應商的保證是「通常會」，不是「不能」。** 你的程式必須假設它會失敗。

## 六、可執行 Demo

情境：把一份 61 字的巡檢紀錄抽成結構化紀錄。五個候選輸出（教學用的固定字串，不是模型即時回應）各跑三層。

<!-- demo: E07_structured_output.py -->
```python
"""第 13 節：Structured Output——語法、Schema 與語義（只用 Python 標準函式庫）。

三層驗證缺一不可：
  第 1 層 語法：這段字串是不是合法 JSON？
  第 2 層 Schema：它的形狀對不對？
  第 3 層 語義：裡面的內容有沒有來源支持？

**第 2 層通過不代表第 3 層會過。** 這是本節最重要的一句話。

候選輸出是教學用的固定字串，不是任何模型的即時回應。
"""

import json

SOURCE = (
    "巡檢紀錄 R-20E07-10E02-01：三號出口堆放紙箱，已於 14:30 通知清除。"
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
     '{"report_id":"R-20E07-10E02-01","hazard":"blocked_exit","severity":"unknown",'
     '"resolved":true,"workers_injured":0,"note":null}'),
    ("B Schema 通過、內容錯誤",
     '{"report_id":"R-20E07-10E02-01","hazard":"no_helmet","severity":"high",'
     '"resolved":true,"workers_injured":0}'),
    ("C 被截斷",
     '{"report_id":"R-20E07-10E02-01","hazard":"blocked_ex'),
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
    repaired = ('{"report_id":"R-20E07-10E02-01","hazard":"blocked_exit","severity":"unknown",'
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
```

執行指令（在 [`llm-demo`](https://github.com/4-learn/llm-demo) 目錄下，**不需要網路、不需要模型**）：

```bash
python3 E07_structured_output.py
```

關鍵輸出（Linux／Python 3.10.15，20E07-E03-05）：

```text
來源（63 字）：巡檢紀錄 R-20E07-10E02-01：三號出口堆放紙箱，已於 14:30 通知清除。現場未發現人員受傷。安全帽配戴正常。

=== 三層驗證 ===
候選                         1 語法     2 Schema   3 語義      
A 合法且正確                    通過       通過         通過        
B Schema 通過、內容錯誤           通過       通過         失敗        
C 被截斷                      失敗       —          —         
D 形狀不對（缺必填）                通過       失敗         —         
E 拒絕回答                     失敗       —          —         

=== 失敗原因 ===
A 合法且正確：三層全部通過
B Schema 通過、內容錯誤：第 3 層 hazard='no_helmet' 在來源中找不到支持（找過 ('未戴', '沒戴', '沒有戴')）
B Schema 通過、內容錯誤：第 3 層 severity='high' 在來源中找不到支持（找過 ('嚴重', '重大')）
C 被截斷：第 1 層 Unterminated string starting at（位置 41）
D 形狀不對（缺必填）：第 2 層 缺少必填欄位 report_id
D 形狀不對（缺必填）：第 2 層 缺少必填欄位 severity
D 形狀不對（缺必填）：第 2 層 缺少必填欄位 resolved
D 形狀不對（缺必填）：第 2 層 缺少必填欄位 workers_injured
E 拒絕回答：第 1 層 Expecting value（位置 0）

=== 把 C 補成合法 JSON 會怎樣？ ===
C 原文（截斷，52 字）：{"report_id":"R-20E07-10E02-01","hazard":"blocked_ex
補完後的欄位：['hazard', 'report_id', 'resolved', 'severity', 'workers_injured']
其中模型沒說過的：['resolved', 'severity', 'workers_injured']（3/5）
補完後三層驗證：語法 通過／Schema 通過／語義 通過
三層全過，但這份結果已經回答不了「模型到底說了什麼」——
補進去的是**你的判斷**，不是模型的輸出，而驗證器看不出差別。
```

## 七、結果怎麼讀

### 1. B 是本節要保留的反例

```text
B Schema 通過、內容錯誤           通過       通過         失敗
```

`B` 是一份**完美合規**的紀錄：每個必填欄位都在、型別都對、值都在允許清單內。如果你的管線只做到第 2 層，它會被寫進資料庫，然後出現在週報上：「三號出口：**未戴安全帽，嚴重程度高**」。

來源裡沒有任何一個字支持這兩件事。來源寫的是「安全帽配戴正常」。

> **Workshop 驗收要求保留一份這樣的反例**，就是因為它最容易被當成成功案例。

### 2. A 的 `severity` 是 `unknown`，這是正確答案

來源沒有提到嚴重程度。所以 `A` 填 `unknown` 才是對的——任何 `low`／`medium`／`high` 都是模型自己的判斷，而不是來源裡的事實。

這也是為什麼 schema 要**明確提供 `unknown` 選項**：如果允許清單只有 `low`／`medium`／`high`，你等於**強迫**模型編一個。

### 3. C 和 E 都是第 1 層失敗，但處理方式相反

| 候選 | 失敗原因 | 正確處理 |
| --- | --- | --- |
| C 被截斷 | `Unterminated string`——沒講完 | 重試、縮短輸出或提高上限 |
| E 拒絕回答 | `Expecting value`——根本不是 JSON | **記為拒絕**，交人工；**不要重試到它肯答為止** |

兩者在解析器眼裡錯誤訊息不同，但只看「解析失敗」這個布林值，你分不出來。**程式要分得出來。**

### 4. 補成合法 JSON 會讓驗證器失明

```text
C 原文（截斷，52 字）：{"report_id":"R-20E07-10E02-01","hazard":"blocked_ex
其中模型沒說過的：['resolved', 'severity', 'workers_injured']（3/5）
補完後三層驗證：語法 通過／Schema 通過／語義 通過
```

這是本節最危險的一段輸出。補完之後**三層全過**——但五個欄位裡有三個是**你**填的，不是模型說的。驗證器看不出差別，資料庫也看不出差別。

課綱的驗收條件寫的是「**禁止用補成合法 JSON 代替事實檢查**」，原因就在這裡：修補不只是沒有用，它**銷毀了「哪些內容來自模型」這個資訊**。之後有人問「這筆資料是怎麼來的」，你已經答不出來了。

## 八、常見誤判

| 誤判 | 為什麼錯 |
| --- | --- |
| 「用了 JSON mode，輸出就可信」 | JSON mode 保證到第 1 層（頂多第 2 層），不碰第 3 層 |
| 「schema 驗證通過了」 | 本節 `B` 通過了，內容是錯的 |
| 「解析失敗就重試」 | 截斷可以重試，**拒絕不行**——重試到它肯答，就是在逼它編 |
| 「截斷的話，自己補完就好」 | 補的是你的判斷，不是模型的輸出，而且之後無法區分 |
| 「enum 只放 low／medium／high 比較乾淨」 | 沒有 `unknown` 選項，就是在強迫模型猜 |
| 「語義檢查交給另一個 LLM」 | 那只是把第 3 層換成另一個需要第 3 層的東西；至少要有人工抽查 |
| 「我寫的語義規則是對的」 | 本節第一版規則就把正確答案判錯了；規則本身也要測試 |

## Workshop：三份輸出，各跑三層

### 任務

1. 執行 Demo，先預測五個候選各會在哪一層失敗，再核對。
2. 自己寫一份**新的**候選 `F`：讓它通過第 1、2 層，但在第 3 層失敗，而且失敗原因跟 `B` **不一樣**（例如 `workers_injured` 寫錯，或 `report_id` 編造）。
3. 把 `SCHEMA` 裡 `severity` 的 `unknown` 拿掉，重跑。`A` 會在哪一層失敗？**這代表 schema 設計在逼模型做什麼？**
4. 在 `SOURCE` 裡把「通知清除」改成「尚未通知清除」，重跑。`A` 還是三層全過嗎？**這說明了語義規則的什麼缺陷？**你會怎麼修？
5. 把 `workers_injured` 的值改成 `true`（布林值），重跑。它會在哪一層被擋下？把 `type_ok` 裡排除 `bool` 的那兩行刪掉，再跑一次。

### 參考判讀

- 第 3 題：`A` 會在**第 2 層**失敗（`severity` 的值 `'unknown'` 不在允許清單）。這代表 schema 在**懲罰誠實的答案**——模型若想通過，唯一的辦法是猜一個嚴重程度。
- 第 4 題：`A` **仍然三層全過**，因為「尚未通知清除」也包含「通知清除」這四個字。這就是**子字串比對無法處理否定**的缺陷——跟第 08 節 `v1` 被「無未戴紀錄」騙到是同一類問題。修法是先檢查否定詞（如第 08 節 `v2` 的做法），而且要**為這個修法加測試**。
- 第 5 題：有排除時，第 2 層擋下（`workers_injured 型別應為 integer，實際 bool`）。刪掉排除之後，**`true` 會通過整數檢查**，因為 Python 的 `bool` 是 `int` 的子類。而且它**連第 3 層也通過**——語義規則只檢查 `workers_injured == 0` 的主張，`true` 不等於 `0`，所以根本沒有規則被觸發。**三層全過，但「受傷人數：true」是一筆無意義的資料。**這說明語義規則只能檢查它有寫到的主張，沒寫到的就是盲區。

### 驗收

- 交出三份輸出（至少一份是你自己寫的）的**三層驗證表**，每一份標明在哪一層失敗、原因是什麼。
- 至少保留**一份 schema 通過但內容錯誤的反例**，並寫明是哪個欄位、來源裡實際寫的是什麼。
- 錯誤處理要**分開截斷與拒絕**：寫出各自的處理方式。
- 不得用補成合法 JSON 代替事實檢查；如果真的需要修補，必須**標記哪些欄位是補的**。
- 第 4 題的語義規則缺陷要寫進報告，不得假裝規則是完美的。

## 教師紀錄與下一步

- 本節**完全不需要模型與網路**：五個候選是教學用的固定字串，不是任何模型的即時回應。**證據等級：本機實測（驗證器在固定字串上的行為）**；本節沒有任何模型輸出。
- `validate_schema` 刻意寫得比 JSON Schema 規格小很多（只有 `type`／`required`／`enum`／`minimum`／未定義欄位），讓學生讀得完。正式專案應該用成熟的驗證函式庫——但**第 3 層無論用什麼函式庫都要自己寫**。
- 撰寫時的實際紀錄：第一版語義規則把 `resolved=true` 的支持字串寫成「已通知清除」，而來源寫的是「已於 14:30 通知清除」，結果**正確的候選 `A` 被判為第 3 層失敗**。改成「通知清除」之後，又產生了 Workshop 第 4 題的反向缺陷。兩件事都保留在講義裡，作為「規則本身也要測試」的實例。
- 第 14 節會接上：工具呼叫的**參數**就是一份結構化輸出，同樣需要三層驗證，而且第 3 層多了一個問題——「這個角色有沒有權限做這件事」（第 09 節）。
- 待教師確認：① 是否要加一組真實供應商的原生結構化輸出對照（需要 API 金鑰與價格時效紀錄，屬第 15 節範圍）；② 候選字串是否要改成與第 08 節評測集同一批合成通報，讓兩節共用資料。
