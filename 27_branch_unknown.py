"""第 27 節：條件分支與 Unknown 測試。

把「模型說了什麼」先解析成有型別的決策，再由程式路由到 通過／需覆核／unknown。
任何沒有被明確辨識的結果，都不得落入通過分支。
需要 langgraph（第 26 節已安裝）；不需要模型。模型輸出以固定 fixtures 代替。
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
