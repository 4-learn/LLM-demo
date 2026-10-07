"""第 37 節：Multi-Agent、固定 Workflow 與 Reducer。

舊教材的 supervisor 照抄（只把呼叫 LLM 換成固定字串），先跑出它的 bug，再修；
接著把同一件事改成固定 workflow 比較步數，最後加上停止條件與重複輸入。
需要 langgraph==1.2.12；不需要模型、不需要 API key。
"""
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError, InvalidUpdateError
from langgraph.graph import END, START, StateGraph

EVENT = {"id": "EVT-2026-0042", "category": "高處墜落", "severity": "高"}
AGENTS = ("regulation", "report", "notification")


# ---------------------------------------------------------------- 三種 completed_agents 合併規則

def union(current, update):
    """去重合併：同一個代理完成兩次只記一次；結果排序，與執行順序無關。"""
    return sorted(set(current or []) | set(update or []))


class Plain(TypedDict, total=False):              # 舊教材：沒有 reducer
    event: dict
    next_agent: str
    completed_agents: list
    regulation_result: str
    report_result: str
    notification_result: str
    log: Annotated[list, operator.add]


class Added(Plain, total=False):
    completed_agents: Annotated[list, operator.add]


class Union(Plain, total=False):
    completed_agents: Annotated[list, union]


# ---------------------------------------------------------------- 代理：舊教材的三個 agent，LLM 換成固定字串

def regulation(state):
    return {"regulation_result": f"{state['event']['category']}：營造安全衛生設施標準（示意）",
            "completed_agents": ["regulation"], "log": ["regulation"]}


def report(state):
    basis = state.get("regulation_result", "尚未查詢")
    return {"report_result": f"報告 {state['event']['id']}；法規依據：{basis}",
            "completed_agents": ["report"], "log": ["report"]}


def notification(state):
    targets = {"高": 3, "中": 2}.get(state["event"]["severity"], 1)
    return {"notification_result": f"已通知 {targets} 人",
            "completed_agents": ["notification"], "log": ["notification"]}


NODES = {"regulation": regulation, "report": report, "notification": notification}


# ---------------------------------------------------------------- 模式一：supervisor（舊教材的決策邏輯）

def supervisor_decide(state):
    done = state.get("completed_agents") or []
    nxt = next((a for a in AGENTS if a not in done), "done")
    return {"next_agent": nxt, "log": [f"S→{nxt}"]}


def build_supervisor(schema, decide=supervisor_decide, checkpointer=None):
    graph = StateGraph(schema)
    graph.add_node("supervisor", decide)
    for name, fn in NODES.items():
        graph.add_node(name, fn)
        graph.add_edge(name, "supervisor")
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", lambda s: s["next_agent"],
                                {**{a: a for a in AGENTS}, "done": END, "stop": END})
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------- 模式二：固定 workflow

def build_fixed(schema, checkpointer=None):
    """法規與通知互不依賴 → 並行；報告需要法規 → 接在法規後面。"""
    graph = StateGraph(schema)
    for name, fn in NODES.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "regulation")
    graph.add_edge(START, "notification")
    graph.add_edge("regulation", "report")
    graph.add_edge("report", END)
    graph.add_edge("notification", END)
    return graph.compile(checkpointer=checkpointer)


def start():
    return {"event": dict(EVENT), "completed_agents": [], "log": []}


def run(app, limit=25):
    """回傳 (最終 state 或錯誤字串, 超步數, 節點執行次數)。同一超步內的節點是並行的。"""
    supersteps = nodes = 0
    final = None
    try:
        for mode, chunk in app.stream(start(), {"recursion_limit": limit}, stream_mode=["updates", "values"]):
            if mode == "updates":                    # 每個節點的更新各一筆
                nodes += len(chunk)
            else:                                    # 每個超步結束時一筆（另有一筆初始 state）
                supersteps += 1
                final = chunk
        return final, supersteps - 1, nodes
    except GraphRecursionError:
        return f"GraphRecursionError（{limit} 步內沒結束）", supersteps, nodes
    except InvalidUpdateError as error:
        return "InvalidUpdateError：" + str(error).splitlines()[0][:52], supersteps, nodes


def missing(final):
    """收尾檢查：不看路由器說了什麼，直接看 state 裡有沒有每個代理的結果。"""
    return [a for a in AGENTS if f"{a}_result" not in final]


def describe(final):
    if isinstance(final, str):
        return final
    return f"completed={final['completed_agents']} 缺結果={missing(final) or '無'}"


# ---------------------------------------------------------------- 模式三：會亂派工的路由（模擬 LLM supervisor）

def make_router(script, budget=None):
    """script：路由器依序給出的決定（教師構造，代表 LLM 可能的輸出）；budget：代理呼叫上限。"""
    picks = iter(script)

    def decide(state):
        calls = sum(1 for entry in state.get("log", []) if not entry.startswith("S→"))
        if budget is not None and calls >= budget:
            return {"next_agent": "stop", "log": [f"S→stop(預算 {budget} 用完)"]}
        nxt = next(picks, "done")
        if nxt not in AGENTS and nxt != "done":
            return {"next_agent": "stop", "log": [f"S→stop(未知代理 {nxt})"]}
        return {"next_agent": nxt, "log": [f"S→{nxt}"]}

    return decide


def main():
    import importlib.metadata as metadata
    print("langgraph:", metadata.version("langgraph"))
    print("事件:", EVENT)

    print("\n== 一、舊教材的 supervisor：completed_agents 沒有 reducer ==")
    final, _, _ = run(build_supervisor(Plain), limit=12)
    print("  結果:", describe(final))
    app = build_supervisor(Plain)
    trail = []
    try:
        for update in app.stream(start(), {"recursion_limit": 12}, stream_mode="updates"):
            (node, data), = update.items()
            trail.append(f"{node}:{data.get('completed_agents', data.get('next_agent'))}")
    except GraphRecursionError:
        pass
    print("  前 6 步:", " → ".join(trail[:6]))

    print("\n== 二、加上 reducer ==")
    for schema in (Added, Union):
        final, supersteps, nodes = run(build_supervisor(schema))
        print(f"  {schema.__name__:<6} {describe(final)}")
        print(f"         超步 {supersteps}、節點執行 {nodes} 次  log={final['log']}")

    print("\n== 三、改成固定 workflow（法規 ∥ 通知 → 報告） ==")
    for schema in (Plain, Added, Union):
        final, supersteps, nodes = run(build_fixed(schema))
        print(f"  {schema.__name__:<6} {describe(final)}")
        if not isinstance(final, str):
            print(f"         超步 {supersteps}、節點執行 {nodes} 次  log={final['log']}")

    print("\n== 四、重複輸入：同一事件在同一 thread 送兩次 ==")
    for schema in (Added, Union):
        app = build_fixed(schema, InMemorySaver())
        config = {"configurable": {"thread_id": EVENT["id"]}}
        app.invoke(start(), config)
        final = app.invoke({"event": dict(EVENT), "log": []}, config)
        print(f"  {schema.__name__:<6} 第 2 次後 completed={final['completed_agents']}  log 長度 {len(final['log'])}")

    print("\n== 五、停止條件：路由器亂派工（構造的決定序列） ==")
    loops = ["regulation"] * 30
    cases = (
        ("一直派同一個，無預算", make_router(loops), 25),
        ("一直派同一個，預算 5", make_router(loops, budget=5), 25),
        ("派到不存在的代理", make_router(["regulation", "lawyer"]), 25),
        ("漏派法規就說 done", make_router(["notification", "report", "done"]), 25),
    )
    for label, decide, limit in cases:
        final, _, nodes = run(build_supervisor(Union, decide), limit=limit)
        print(f"  {label}（節點執行 {nodes} 次）")
        if isinstance(final, str):
            print(f"    {final}")
            continue
        verdict = "通過" if not missing(final) else "不通過，不能當成完成"
        print(f"    最後決定 {final['log'][-1]}；收尾檢查：缺 {missing(final) or '無'} → {verdict}")
        if "report_result" in final:
            print(f"    報告內容：{final['report_result']}")


if __name__ == "__main__":
    main()
