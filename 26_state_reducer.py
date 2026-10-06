"""第 26 節：LangGraph State 與 Reducer。

同一個 graph、同一份輸入，只差一行 reducer，結果就從「丟資料」「當掉」變成「正確合併」。
需要 langgraph（見講義安裝指令）；不需要模型、不需要 API key。
"""
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.errors import InvalidUpdateError

REPORT = {"report_id": "R-001", "text": "三樓外牆鷹架，一名工人未扣安全帶。"}


# --- 三種 state schema：findings 欄位的合併規則不同 ---------------------------------

class Overwrite(TypedDict):
    report_id: str
    findings: list                                  # 沒有 reducer：後寫的蓋掉先寫的


class Append(TypedDict):
    report_id: str
    findings: Annotated[list, operator.add]         # 串接：不會丟，但會重複


def merge_by_id(current, update):
    """依 finding id 合併：同 id 以新的為準；結果依 id 排序，與節點執行順序無關。"""
    merged = {item["id"]: item for item in current or []}
    for item in update or []:
        merged[item["id"]] = item
    return [merged[key] for key in sorted(merged)]


class Keyed(TypedDict):
    report_id: str
    findings: Annotated[list, merge_by_id]


# --- 節點：每個節點只回傳「這次要更新的欄位」，不是整份 state ------------------------

def vision(state):
    return {"findings": [{"id": "F-vision", "by": "vision", "label": "未見安全帶"}]}


def text(state):
    return {"findings": [{"id": "F-text", "by": "text", "label": "高處作業"}]}


def build(schema, parallel, checkpointer=None):
    graph = StateGraph(schema)
    graph.add_node("vision", vision)
    graph.add_node("text", text)
    if parallel:                                     # 同一步同時執行兩個節點
        graph.add_edge(START, "vision")
        graph.add_edge(START, "text")
        graph.add_edge("vision", END)
        graph.add_edge("text", END)
    else:                                            # 先 vision 再 text
        graph.add_edge(START, "vision")
        graph.add_edge("vision", "text")
        graph.add_edge("text", END)
    return graph.compile(checkpointer=checkpointer)


def labels(state):
    return [item["by"] for item in state["findings"]]


def run(schema, parallel, initial=None):
    start = {"report_id": REPORT["report_id"], "findings": initial or []}
    try:
        return labels(build(schema, parallel).invoke(start))
    except InvalidUpdateError as error:
        return f"InvalidUpdateError：{str(error).splitlines()[0]}"


def main():
    import importlib.metadata as metadata
    print("langgraph:", metadata.version("langgraph"))
    print("通報:", REPORT["report_id"], REPORT["text"])

    print("\n== 一、三種 reducer × 循序／並行 ==")
    for schema in (Overwrite, Append, Keyed):
        for parallel in (False, True):
            mode = "並行" if parallel else "循序"
            print(f"{schema.__name__:<9} {mode}: {run(schema, parallel)}")

    print("\n== 二、初始值也會被 reducer 處理 ==")
    seed = [{"id": "F-manual", "by": "manual", "label": "人工先標"}]
    for schema in (Overwrite, Append, Keyed):
        print(f"{schema.__name__:<9} 循序: {run(schema, False, seed)}")

    print("\n== 三、同一 thread 再跑一次（checkpoint 保留舊 state）==")
    for schema in (Append, Keyed):
        app = build(schema, True, InMemorySaver())
        config = {"configurable": {"thread_id": "R-001"}}
        first = labels(app.invoke({"report_id": "R-001", "findings": []}, config))
        second = labels(app.invoke({"report_id": "R-001", "findings": []}, config))
        print(f"{schema.__name__:<9} 第 1 次 {first}  第 2 次 {second}")

    print("\n== 四、同 id 更新：修正而不是新增 ==")
    corrected = [{"id": "F-vision", "by": "reviewer", "label": "安全帶已扣（覆核修正）"}]
    for schema in (Append, Keyed):
        state = build(schema, False).invoke({"report_id": "R-001", "findings": []})
        final = schema.__annotations__["findings"].__metadata__[0](state["findings"], corrected)
        print(f"{schema.__name__:<9} {[(i['id'], i['by']) for i in final]}")

    print("\n== 五、框架不會替你擋的錯 ==")
    graph = StateGraph(Keyed)
    graph.add_node("typo", lambda state: {"finding": [{"id": "F-x", "by": "typo", "label": "?"}]})
    graph.add_node("forgot", lambda state: None)
    graph.add_edge(START, "typo")
    graph.add_edge("typo", "forgot")
    graph.add_edge("forgot", END)
    out = graph.compile().invoke({"report_id": "R-001", "findings": []})
    print("欄位打錯字（finding）與回傳 None → 結果:", out, "（沒有錯誤訊息）")


if __name__ == "__main__":
    main()
