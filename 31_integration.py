"""第 31 節：流程里程碑——沿用骨架整合。

不重寫任何東西：直接載入第 23（RAG 引用）、25（副作用政策）、28（重試與冪等）、
30（trace 與遮罩）節的程式，用第 26–29 節的 LangGraph 寫法接成一條流程，
跑「正常、unknown、拒絕、工具失敗」四類整合案例。
需要 langgraph==1.2.12；不需要模型（生成沿用第 23 節的劇本）。
"""
import re
import types
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

HERE = Path(__file__).resolve().parent


def lesson(nn):
    """先找 llm-demo 裡的 NN_*.py；找不到就從講義 NN-*.md 抽出內嵌程式。

    兩種都是讀原始碼後 exec，不用 import：import 會在資料夾裡留下 __pycache__。
    """
    found = [(p, p.read_text(encoding="utf-8")) for p in sorted(HERE.glob(f"{nn}_*.py"))]
    for md in sorted(HERE.glob(f"{nn}-*.md")):
        found += [(md, src) for _, src in re.findall(r"<!-- demo: ([\w.]+) -->\n```python\n(.*?)\n```",
                                                     md.read_text(encoding="utf-8"), re.S)]
    if not found:
        raise FileNotFoundError(f"找不到第 {nn} 節的程式（{nn}_*.py 或 {nn}-*.md）")
    path, source = found[0]
    module = types.ModuleType(f"lesson{nn}")
    module.__file__ = str(path)
    exec(compile(source, path.name, "exec"), module.__dict__)
    return module


RAG, POLICY, RETRY, TRACE = lesson("23"), lesson("25"), lesson("28"), lesson("30")


class State(TypedDict, total=False):
    case_id: str
    role: str
    action: str                 # 通過 RAG 後要做的事：create_ticket 或 notify_site
    qid: str
    disposition: str
    route: str
    answer: Optional[str]
    result: str


# --- 共用環境：一次整合測試一份 -------------------------------------------------------

class World:
    def __init__(self, ticket_failures=(), unmasked=False):
        corpus = RAG.load("sop_corpus.json")
        self.corpus = dict(corpus, documents=corpus["documents"] + RAG.EXTRA_DOCS)
        self.index = RAG.LexicalIndex(corpus["chunks"] + RAG.EXTRA_CHUNKS)
        self.cases = {c["case_id"]: c for c in RAG.load("search_cases.json")["cases"]}
        self.generator = RAG.ScriptedGenerator(RAG.SCRIPT)
        self.sandbox = POLICY.Sandbox()                         # 通知寫在這裡
        self.tickets = RETRY.TicketService(list(ticket_failures))   # 開單走這裡（含故障劇本）
        self.clock = TRACE.FakeClock()
        self.tracer = TRACE.Tracer(self.clock, (lambda a: dict(a)) if unmasked else TRACE.mask_by_value)

    def side_effects(self):
        return len(self.tickets.tickets) + len(self.sandbox.notifications)


def build(world, script_case=None):
    tracer, clock = world.tracer, world.clock

    def rag(state):
        span = tracer.start(state["case_id"], "rag", qid=state["qid"])
        case = dict(world.cases[state["qid"]], case_id=script_case or state["case_id"])
        out = RAG.handle(case, world.corpus, world.index, world.generator)
        clock.advance(900)
        tracer.end(span, disposition=out["處置"], context=out["上下文"])
        return {"disposition": out["處置"]}

    def route(state):
        if state["disposition"].startswith("回覆"):
            return "act" if state["action"] == "create_ticket" else "ask"
        return "review"                                      # 查無、衝突、驗證失敗、檢索故障：一律人工

    def act(state):
        span = tracer.start(state["case_id"], "tool:create_ticket", role=state["role"])
        verdict, reason = POLICY.decide("create_ticket", {"site": "A", "priority": "low"}, state["role"])
        if verdict != "執行":
            tracer.end(span, status="denied", reason=reason)
            return {"result": f"未開單（{reason}）"}
        key = f"{state['case_id']}:create_ticket"           # 冪等鍵：同一案件只開一張
        ticket, log = RETRY.with_retry(lambda: world.tickets.create("A", key), 3, RETRY.FakeClock())
        clock.advance(120 * len(log))
        tracer.end(span, status="ok" if ticket else "error", attempts=len(log), last=log[-1])
        return {"result": f"已開單 {ticket['id']}" if ticket else f"開單失敗（{log[-1]}）→ 轉人工"}

    def ask(state):
        verdict, reason = POLICY.decide("notify_site", {"site": "A", "message": "停工"}, state["role"])
        if verdict == "拒絕":
            tracer.start(state["case_id"], "policy", decision="deny")
            tracer.end(tracer.stack[-1], status="denied", reason=reason)
            return {"answer": "denied_by_policy"}
        answer = interrupt({"case_id": state["case_id"], "action": "notify_site", "reason": reason})
        return {"answer": answer}

    def after_ask(state):
        return "notify" if state["answer"] == "approve" else "stop"

    def notify(state):
        span = tracer.start(state["case_id"], "tool:notify_site", approved=True)
        verdict, reason, _ = POLICY.call(world.sandbox, "notify_site",
                                         {"site": "A", "message": f"{state['case_id']} 停工"}, state["role"], approved=True)
        tracer.end(span, decision=verdict)
        return {"result": "已通知工地"}

    def stop(state):
        why = "政策拒絕" if state["answer"] == "denied_by_policy" else "人工拒絕"
        return {"result": f"未通知（{why}）"}

    def review(state):
        span = tracer.start(state["case_id"], "review_queue")
        tracer.end(span, status="needs_review")
        return {"result": "轉人工覆核"}

    graph = StateGraph(State)
    for name, fn in (("rag", rag), ("act", act), ("ask", ask), ("notify", notify), ("stop", stop), ("review", review)):
        graph.add_node(name, fn)
    graph.add_edge(START, "rag")
    graph.add_conditional_edges("rag", route, {"act": "act", "ask": "ask", "review": "review"})
    graph.add_conditional_edges("ask", after_ask, {"notify": "notify", "stop": "stop"})
    for name in ("act", "notify", "stop", "review"):
        graph.add_edge(name, END)
    return graph.compile(checkpointer=InMemorySaver())


# 整合案例：(代號, 類別, 角色, 動作, 題目, 借用第 23 節哪一個劇本, 人工回答, 開單故障, 預期結果, 預期副作用數)
CASES = (
    ("I-OK",     "正常",     "staff",   "create_ticket", "Q02", "A1", None,      (),                             "已開單 T1",       1),
    ("I-RETRY",  "正常",     "staff",   "create_ticket", "Q02", "A1", None,      ("timeout_after_write",),       "已開單 T1",       1),
    ("I-UNK",    "unknown",  "staff",   "create_ticket", "Q05", "N1", None,      (),                             "轉人工覆核",      0),
    ("I-BADCIT", "unknown",  "staff",   "create_ticket", "Q02", "A2", None,      (),                             "轉人工覆核",      0),
    ("I-REJ",    "拒絕",     "manager", "notify_site",   "Q02", "A1", "reject",  (),                             "未通知（人工拒絕）", 0),
    ("I-DENY",   "拒絕",     "staff",   "notify_site",   "Q02", "A1", None,      (),                             "未通知（政策拒絕）", 0),
    ("I-APPR",   "正常",     "manager", "notify_site",   "Q02", "A1", "approve", (),                             "已通知工地",      1),
    ("I-FAIL",   "工具失敗", "staff",   "create_ticket", "Q02", "A1", None,      ("rate_limited",) * 3,          "開單失敗",        0),
    ("I-AUTH",   "工具失敗", "staff",   "create_ticket", "Q02", "A1", None,      ("unauthorized",),              "開單失敗",        0),
)


def run_case(case, unmasked=False):
    case_id, _, role, action, qid, script, answer, faults, _, _ = case
    world = World(faults, unmasked)
    app = build(world, script)
    config = {"configurable": {"thread_id": case_id}}
    out = app.invoke({"case_id": case_id, "role": role, "action": action, "qid": qid}, config)
    if "__interrupt__" in out:
        out = app.invoke(Command(resume=answer), config)
    return out, world


def main():
    import importlib.metadata as metadata
    print("langgraph:", metadata.version("langgraph"), "｜ 沿用：第 23、25、28、30 節程式；生成為第 23 節劇本")

    print("\n== 一、整合案例 ==")
    print(f"{'代號':<9} {'類別':<5} {'結果':<38} 副作用 trace 狀態")
    passed = 0
    for case in CASES:
        out, world = run_case(case)
        expected, effects = case[8], case[9]
        ok = out["result"].startswith(expected) and world.side_effects() == effects
        passed += ok
        statuses = ",".join(f"{s['name']}={s['status']}" for s in world.tracer.spans)
        print(f"{case[0]:<9} {case[1]:<5} {out['result']:<38} {world.side_effects()}   {statuses}{'' if ok else '  ← 不符'}")
    print(f"通過 {passed}／{len(CASES)}")

    print("\n== 二、拒絕零副作用 ==")
    for case in CASES:
        if case[1] == "拒絕":
            out, world = run_case(case)
            print(f"{case[0]}: 通知 {len(world.sandbox.notifications)}、開單 {len(world.tickets.tickets)}、"
                  f"沙盒寫入紀錄 {world.sandbox.writes}")

    print("\n== 三、失敗也看得到 ==")
    for case_id in ("I-RETRY", "I-FAIL", "I-AUTH"):
        out, world = run_case(next(c for c in CASES if c[0] == case_id))
        span = next(s for s in world.tracer.spans if s["name"] == "tool:create_ticket")
        print(f"{case_id}: status={span['status']} attempts={span['attrs']['attempts']} last={span['attrs']['last']}")

    print("\n== 四、介面斷點：沒有遮罩的 trace ==")
    case = ("I-PII", "正常", "staff", "create_ticket", "Q02", "A1", None, (), "已開單", 1)
    for unmasked in (False, True):
        world = World((), unmasked)
        app = build(world, "A1")
        app.invoke({"case_id": "I-PII", "role": "staff", "action": "create_ticket", "qid": "Q02"},
                   {"configurable": {"thread_id": "I-PII"}})
        span = world.tracer.start("I-PII", "note", text="回報人 0912-345-678")
        world.tracer.end(span)
        print(f"{'不遮罩' if unmasked else '遮罩'}: trace 中的個資 {TRACE.leaks(world.tracer.jsonl()) or '無'}")

    print("\n== 五、尚未涵蓋（不是工安認證）==")
    for gap in ("生成是劇本，未接真實模型；第 23 節 N2 的否定句誤引仍會通過驗證",
                "checkpoint 用 InMemorySaver；跨程序恢復見第 29 節，尚未接進本流程",
                "開單與通知是沙盒，未接真實系統；權限只有兩種角色",
                "SOP 為虛構教學資料，未對照正式法規來源與版本"):
        print("-", gap)


if __name__ == "__main__":
    main()
