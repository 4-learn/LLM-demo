# 條件分支與 Unknown 測試

> **新版正文：20E07-E03-06｜對應新版 LLM 第 28 節：條件分支與 Unknown 測試。**
> 工安判斷的流程至少有三種結局：自動通過、需人工覆核、無法判斷。最常見的錯誤是寫成「不是違規就通過」——於是模型輸出壞掉、標籤拼錯、信心不足，全部被放行。核心主張：**先把模型輸出解析成有型別的決策，再由程式路由；通過只有一條路，其他任何沒被明確辨識的結果都要有自己的去處，而且每條邊都要有測試走過。**

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先修第 13 節（輸出契約與驗證）、第 27 節（state 與 reducer）。第 30 節的 HITL 會接在本節的「需覆核」分支後面。

| 時間 | 內容 |
| --- | --- |
| 0–8 分鐘 | 模型判斷 vs 程式路由；typed decision |
| 8–20 分鐘 | conditional edges、path map、終止條件 |
| 20–28 分鐘 | unknown 的六種來源 |
| 28–35 分鐘 | 跑 Demo：兩種路由規則對照 |
| 35–50 分鐘 | Workshop：補 fixture、讓每條邊都被走過 |

驗收能力：能用固定 fixtures 證明流程的**每一條邊**都被走過，且任何未被明確辨識的輸出都**不會進入自動通過**。

## 一、模型判斷與程式路由要分開

| 層 | 誰負責 | 本節的函式 | 可以出錯嗎 |
| --- | --- | --- | --- |
| 判斷 | 模型 | `judge` 節點（本節用 fixtures 代替模型） | 會：格式壞、標籤拼錯、亂給信心 |
| 解析 | 程式 | `parse()` → `Decision` 或錯誤原因 | 不能：只認得完全符合契約的輸出 |
| 路由 | 程式 | `decide()` → 四個字串之一 | 不能：每種情況都要有明確去處 |

模型只負責「說出看法」；**要不要放行，是程式依規則決定**。所以路由函式裡不應該有任何模糊比對（例如「label 裡有 compliant 就算」），那等於把決定權交回給模型的拼字。

## 二、conditional edges 的三個要點

1. **path map**：`add_conditional_edges("judge", decide, {"retry": "judge", "auto_pass": ..., ...})`。寫出所有可能的回傳值與目的地，回傳了不在表裡的值會直接 `KeyError`。
2. **回傳型別寫成 `Literal`**：讓讀程式的人（和畫圖工具）知道只有這幾種。注意它**不會在執行時檢查**。
3. **終止條件**：只要有邊能回到自己（`retry → judge`），就要有次數上限。LangGraph 的 `recursion_limit`（預設 25）是最後一道保險，不是設計。

## 三、unknown 的六種來源

| 來源 | fixture | 為什麼不能放行 |
| --- | --- | --- |
| 模型自己說不知道 | U1 | 這是誠實的回答，應該被尊重 |
| 缺欄位 | M1 | 沒有 label，就沒有判斷 |
| 未預期標籤 | X1 `helmet_ok` | 不在契約裡的值，意思不明 |
| 大小寫不同 | C1 `Compliant` | 看起來像，但契約沒說可以；要放寬就改契約，不是改路由 |
| 信心格式錯 | N1 `"high"`、B1 `1.7` | 無法和門檻比較 |
| 解析失敗（重試後仍失敗） | J2 | 模型沒有照格式回答 |

另外有兩種「標籤是 compliant 但仍要覆核」：信心低於門檻（P2）、沒有附證據（P3）。這兩個不是 unknown，是**證據不足**，所以走 review。

## 四、可執行 Demo

需要第 27 節裝好的 `langgraph==1.2.12`；不需要模型。模型的回應用 `FIXTURES` 固定，每個 fixture 寫好**預期路由**。

<!-- demo: 51_branch_unknown.py -->
```python
"""第 28 節：條件分支與 Unknown 測試。

把「模型說了什麼」先解析成有型別的決策，再由程式路由到 通過／需覆核／unknown。
任何沒有被明確辨識的結果，都不得落入通過分支。
需要 langgraph（第 27 節已安裝）；不需要模型。模型輸出以固定 fixtures 代替。
"""
import json
import logging
from typing import Literal, Optional, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

LABELS = ("compliant", "violation", "unknown")
MIN_CONFIDENCE = 0.8
MAX_ATTEMPTS = 2
ROUTES = ("auto_pass", "review", "unknown")

# 每個 fixture：(代號, 預期路由, 模型依序回的原始文字)。第二個回應只在需要重試時用到。
FIXTURES = [
    ("P1", "auto_pass", ['{"label": "compliant", "confidence": 0.93, "evidence": ["E1"]}']),
    ("P2", "review",    ['{"label": "compliant", "confidence": 0.55, "evidence": ["E1"]}']),
    ("P3", "review",    ['{"label": "compliant", "confidence": 0.95, "evidence": []}']),
    ("V1", "review",    ['{"label": "violation", "confidence": 0.90, "evidence": ["E2"]}']),
    ("U1", "unknown",   ['{"label": "unknown", "confidence": 0.20, "evidence": []}']),
    ("M1", "unknown",   ['{"confidence": 0.90, "evidence": ["E1"]}']),
    ("X1", "unknown",   ['{"label": "helmet_ok", "confidence": 0.99, "evidence": ["E1"]}']),
    ("C1", "unknown",   ['{"label": "Compliant", "confidence": 0.93, "evidence": ["E1"]}']),
    ("N1", "unknown",   ['{"label": "compliant", "confidence": "high", "evidence": ["E1"]}']),
    ("B1", "unknown",   ['{"label": "compliant", "confidence": 1.7, "evidence": ["E1"]}']),
    ("J1", "auto_pass", ['{"label": "compliant", "confidence": 0.9',
                         '{"label": "compliant", "confidence": 0.91, "evidence": ["E1"]}']),
    ("J2", "unknown",   ['好的，我判斷這張照片是合規的。', '合規。']),
]


class Decision(TypedDict):
    label: str
    confidence: float
    evidence: list


class State(TypedDict, total=False):
    case_id: str
    responses: list            # 測試用：模型依序會回的文字
    attempts: int
    decision: Optional[Decision]
    error: Optional[str]
    route: str


def parse(raw):
    """原始文字 → (Decision, None) 或 (None, 錯誤原因)。只認得完全符合契約的輸出。"""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, "parse_error"
    if not isinstance(data, dict) or "label" not in data:
        return None, "missing_label"
    if data["label"] not in LABELS:
        return None, f"invalid_label:{data['label']}"
    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        return None, "invalid_confidence"
    evidence = data.get("evidence")
    if not isinstance(evidence, list):
        return None, "invalid_evidence"
    return {"label": data["label"], "confidence": float(confidence), "evidence": evidence}, None


# --- 節點：判斷（模型的事）和路由（程式的事）分開 -----------------------------------

def judge(state):
    attempts = state.get("attempts", 0)
    decision, error = parse(state["responses"][attempts])
    return {"attempts": attempts + 1, "decision": decision, "error": error}


def decide(state) -> Literal["retry", "auto_pass", "review", "unknown"]:
    """唯一的路由規則。通過只有一條路，其他全部有明確去處。"""
    if state.get("error") == "parse_error" and state["attempts"] < MAX_ATTEMPTS:
        return "retry"
    decision = state.get("decision")
    if decision is None or decision["label"] == "unknown":
        return "unknown"
    if decision["label"] == "violation":
        return "review"
    if decision["confidence"] >= MIN_CONFIDENCE and decision["evidence"]:
        return "auto_pass"
    return "review"


def naive(state):
    """常見寫法：「不是違規就通過」。parse 失敗、未知標籤都會被放行。"""
    decision = state.get("decision") or {}
    return "review" if decision.get("label") == "violation" else "auto_pass"


def build(router, path_map=True):
    graph = StateGraph(State)
    graph.add_node("judge", judge)
    for route in ROUTES:
        graph.add_node(route, lambda state, route=route: {"route": route})
        graph.add_edge(route, END)
    graph.add_edge(START, "judge")
    if path_map:
        graph.add_conditional_edges("judge", router, {"retry": "judge", **{r: r for r in ROUTES}})
    else:
        graph.add_conditional_edges("judge", router)
    return graph.compile()


def run(app, case_id, responses):
    return app.invoke({"case_id": case_id, "responses": responses, "attempts": 0})


def main():
    import importlib.metadata as metadata
    logging.getLogger("langgraph").setLevel(logging.ERROR)   # 第三段會讓 langgraph 印警告；不讓它混進輸出
    print("langgraph:", metadata.version("langgraph"), "｜ 通過門檻: confidence >=", MIN_CONFIDENCE, "且有 evidence")

    print("\n== 一、fixtures × 兩種路由 ==")
    typed, loose = build(decide), build(naive)
    print(f"{'代號':<4} {'預期':<9} {'decide':<9} {'naive':<9} 嘗試 error")
    leaked, taken = [], set()
    for case_id, expected, responses in FIXTURES:
        good, bad = run(typed, case_id, responses), run(loose, case_id, responses)
        taken.add(good["route"])
        if good["attempts"] > 1:
            taken.add("retry")
        if bad["route"] == "auto_pass" and expected != "auto_pass":
            leaked.append(case_id)
        mark = "" if good["route"] == expected else "  ← 不符"
        print(f"{case_id:<4} {expected:<9} {good['route']:<9} {bad['route']:<9} {good['attempts']:>2}   {good['error'] or '-'}{mark}")
    print("naive 誤放行:", leaked, f"（{len(leaked)}／{len(FIXTURES)}）")

    print("\n== 二、每條邊都要有 fixture 走過 ==")
    edges = ["retry", *ROUTES]
    print("走過:", [e for e in edges if e in taken], "｜ 沒走過:", [e for e in edges if e not in taken] or "無")

    print("\n== 三、路由回傳打錯字的節點名 ==")
    typo = lambda state: "reveiw"
    final = run(build(typo, path_map=False), "T1", FIXTURES[3][2])
    print("沒有 path_map → 正常結束，route =", final.get("route"), "（只在 log 留一行警告）")
    try:
        run(build(typo), "T1", FIXTURES[3][2])
    except KeyError as error:
        print("有 path_map   → KeyError:", error)

    print("\n== 四、終止條件 ==")
    endless = [FIXTURES[10][2][0]] * 50
    final = run(typed, "L1", endless)
    print(f"一直解析失敗 → 第 {final['attempts']} 次停止，route = {final['route']}")
    always_retry = lambda state: "retry"
    try:
        build(always_retry).invoke({"case_id": "L2", "responses": endless, "attempts": 0}, {"recursion_limit": 8})
    except GraphRecursionError:
        print("忘了寫上限 → GraphRecursionError（recursion_limit=8 才擋下）")


if __name__ == "__main__":
    main()
```

執行（程式也放在 [llm-demo](https://github.com/4-learn/llm-demo)）：

```bash
python 51_branch_unknown.py
```

關鍵輸出（Linux x86_64／Python 3.10.15，langgraph 1.2.12，20E07-E03-06）：

```text
langgraph: 1.2.12 ｜ 通過門檻: confidence >= 0.8 且有 evidence

== 一、fixtures × 兩種路由 ==
代號   預期        decide    naive     嘗試 error
P1   auto_pass auto_pass auto_pass  1   -
P2   review    review    auto_pass  1   -
P3   review    review    auto_pass  1   -
V1   review    review    review     1   -
U1   unknown   unknown   auto_pass  1   -
M1   unknown   unknown   auto_pass  1   missing_label
X1   unknown   unknown   auto_pass  1   invalid_label:helmet_ok
C1   unknown   unknown   auto_pass  1   invalid_label:Compliant
N1   unknown   unknown   auto_pass  1   invalid_confidence
B1   unknown   unknown   auto_pass  1   invalid_confidence
J1   auto_pass auto_pass auto_pass  2   -
J2   unknown   unknown   auto_pass  2   parse_error
naive 誤放行: ['P2', 'P3', 'U1', 'M1', 'X1', 'C1', 'N1', 'B1', 'J2'] （9／12）

== 二、每條邊都要有 fixture 走過 ==
走過: ['retry', 'auto_pass', 'review', 'unknown'] ｜ 沒走過: 無

== 三、路由回傳打錯字的節點名 ==
沒有 path_map → 正常結束，route = None （只在 log 留一行警告）
有 path_map   → KeyError: 'reveiw'

== 四、終止條件 ==
一直解析失敗 → 第 2 次停止，route = unknown
忘了寫上限 → GraphRecursionError（recursion_limit=8 才擋下）
```

## 五、結果怎麼讀

### 1. 「不是違規就通過」放行了 9／12

`naive` 只有一個判斷：label 是 `violation` 就覆核，其他通通通過。12 個 fixtures 中只有 2 個預期該通過，它卻放行了 11 個——9 個是誤放行，包括模型明說「不知道」的 U1、根本沒有 JSON 的 J2。這些在正常 demo 裡都看不到，因為正常 demo 只會餵 P1 和 V1。

### 2. 重試只針對「格式壞了」

J1 第一次回傳被截斷的 JSON，`decide` 回 `retry`，第二次拿到完整輸出就通過。J2 兩次都不是 JSON，用完 `MAX_ATTEMPTS` 後走 unknown。其他錯誤（標籤錯、信心錯）**不重試**：模型很可能下次還是一樣，重試只是多花錢（第 15 節）。

### 3. 每條邊都被走過

`decide` 有 4 種回傳值，12 個 fixtures 合起來每一種都走過。如果某條邊沒有任何 fixture 走過，就代表那條路從沒被測試——這是本節的驗收標準。

### 4. 沒寫 path map，打錯字不會報錯

路由回傳 `"reveiw"`（打錯字）。**沒有 path map** 時，LangGraph 只在 log 留一行 `wrote to unknown channel branch:to:reveiw, ignoring it`，流程正常結束，`route` 是 `None`——這份通報**沒有任何結局**，也沒有人會被通知。有 path map 時直接 `KeyError`。這和第 27 節「欄位打錯字靜默忽略」是同一類問題。

Demo 把 langgraph 的 log 等級調到 ERROR，是為了讓輸出可以逐字凍結；你自己跑的時候拿掉那行，就會看到那行警告。

### 5. 有上限 vs 靠框架保險

`decide` 有 `MAX_ATTEMPTS = 2`，連續失敗 50 次的輸入也在第 2 次停下並走 unknown。永遠回 `retry` 的路由則要等 `recursion_limit` 擋下，丟出 `GraphRecursionError`——流程沒有結局，只有例外。

## 六、常見誤判

| 說法 | 為什麼不對 |
| --- | --- |
| 「不是違規就通過」 | 解析失敗、未知標籤、模型說不知道都會被放行 |
| 「大小寫差一點，寬鬆比對就好」 | 寬鬆比對讓模型的拼字決定是否放行；要放寬就改契約並測試 |
| 「信心高就通過」 | 沒附證據的高信心（P3）仍是證據不足 |
| 「所有錯誤都重試」 | 標籤錯、信心錯重試通常一樣錯，只增加成本 |
| 「有 recursion_limit，不用自己寫上限」 | 那是例外，不是結局；通報會沒有任何去處 |
| 「Literal 型別會擋錯誤的回傳值」 | 不會；擋的是 path map |

## Workshop：補 fixture、讓每條邊都被走過

### 任務

1. 新增 fixture `E1`：`{"label": "compliant", "confidence": 0.9, "evidence": "E1"}`（evidence 是字串而不是清單）。預測它走哪條路，再跑跑看。
2. 新增 fixture `T1`：信心剛好等於門檻 `0.8`、有證據的 compliant。它該通過嗎？跑出結果後，寫一句話說明門檻「含等於」的理由或反對理由。
3. 把 `FIXTURES` 刪到只剩 P1、V1、U1 三筆，重跑第二段，哪條邊沒被走過？
4. 把 `naive` 改成「只有 compliant 才通過，其他都覆核」。它還會誤放行嗎？和 `decide` 比還差什麼？
5. 新增一條規則：`violation` 且信心 ≥ 0.95 時走新的 `urgent` 分支。需要改哪幾個地方？忘了改 path map 會怎樣？

### 參考判讀

以下為備課機實跑（langgraph 1.2.12，20E07-E03-06）：

- 第 1 題：`parse` 回 `invalid_evidence`，走 unknown。
- 第 2 題：`0.8 >= 0.8` 成立，走 auto_pass。門檻含不含等於沒有標準答案，但**必須寫在契約裡並有 fixture 測試**。
- 第 3 題：只剩 P1、V1、U1 時，`retry` 沒被走過。
- 第 4 題：改成只有 compliant 才通過後，誤放行剩 **P2、P3 兩筆**（信心低、沒證據）。它擋下了格式問題，但沒有門檻與證據檢查；M1 這類缺欄位的結果會被送去覆核，不是 unknown，人會收到無法判讀的工單。
- 第 5 題：要改 `decide`、`ROUTES`（新增節點與邊）與 path map。只改 `decide` 而沒加進 path map 時，回傳 `urgent` 會 `KeyError`。

### 驗收

- 每個 fixture 都有預期路由，且 `decide` 全部符合。
- 第二段顯示「沒走過：無」。
- 自動通過的 fixture 全部滿足：label 為 compliant、信心 ≥ 門檻、證據非空。

## 教師紀錄與下一步

- 證據等級：**本機實測**（Linux x86_64、Python 3.10.15、langgraph 1.2.12）；模型輸出是 **fixtures**，本節不呼叫模型。兩次執行輸出逐字相同；Workshop 參考判讀逐題實跑並由測試覆蓋。
- 撰寫時的實際紀錄：
  1. 原本以為沒有 path map 時，路由回傳不存在的節點名會報錯。實跑是**只在 log 警告、流程照常結束**，所以加成第五段第 4 點，並讓 Demo 一律使用 path map。
  2. 回傳型別寫 `Literal` 但回傳不在其中的值：沒有 path map 時同樣只有警告；有 path map 時是 `KeyError`。
- 門檻 0.8 與「需有證據」是教學設定，不是工安標準；實際門檻需由教師與業界顧問依第 08 節的評測結果決定。
- 下一步：第 30 節把 review 分支接上 `interrupt`，讓人工的核准或拒絕真正決定下一步；第 32 節的整合測試會沿用這組 fixtures。
