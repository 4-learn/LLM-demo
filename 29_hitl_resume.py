"""第 29 節：HITL——真正拒絕與恢復。

人工核准才寄出停工通知；拒絕就真的不寄；流程中斷後，換一個新程序也能從存檔繼續。
需要 langgraph==1.2.12 與 langgraph-checkpoint-sqlite==3.1.1；不需要模型。
「寄出通知」寫進本機沙盒 outbox（SQLite 表），不會真的寄信。

整段示範：      python 29_hitl_resume.py
跨程序恢復：    python 29_hitl_resume.py --db hitl.db start R-101
               python 29_hitl_resume.py --db hitl.db resume R-101 approve
"""
import argparse
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

ANSWERS = ("approve", "reject")
MAX_ASKS = 3
CALLS = []                        # 每個節點被執行的紀錄（看出 resume 時誰重跑）


class State(TypedDict, total=False):
    report_id: str
    draft: str
    answer: Optional[str]
    asks: int
    result: Optional[str]


# --- 沙盒 outbox：唯一的副作用；以 report_id 當冪等鍵 ---------------------------------

def open_outbox(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS outbox (key TEXT PRIMARY KEY, body TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS attempts (n INTEGER)")   # 每次「嘗試寄出」都記一筆
    return conn


def send(store, key, body):
    conn = store.outbox
    conn.execute("INSERT INTO attempts VALUES (1)")
    inserted = conn.execute("INSERT OR IGNORE INTO outbox VALUES (?, ?)", (key, body)).rowcount
    conn.commit()
    return inserted == 1


def outbox_count(store):
    conn = store.outbox
    sent = conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
    tried = conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    return sent, tried


# --- graph ---------------------------------------------------------------------------

def build(conn, unsafe=False):
    """unsafe=True：把寄出放在 interrupt() 之前（錯誤示範）。"""

    def draft(state):
        CALLS.append("draft")
        return {"draft": f"{state['report_id']}：三樓外牆鷹架暫停作業，待安全帶改善後復工。", "asks": 0}

    def ask(state):
        CALLS.append("ask")
        if unsafe:
            send(conn, state["report_id"], state["draft"])      # ← 在人回答前就寄了
        answer = interrupt({"report_id": state["report_id"], "draft": state["draft"],
                            "choices": list(ANSWERS), "asks": state["asks"] + 1})
        CALLS.append(f"ask→{answer}")
        return {"answer": answer, "asks": state["asks"] + 1}

    def after_ask(state):
        if state["answer"] == "approve":
            return "execute"
        if state["answer"] == "reject":
            return "rejected"
        return "ask" if state["asks"] < MAX_ASKS else "rejected"   # 看不懂的回答：再問，問太多次就當拒絕

    def execute(state):
        CALLS.append("execute")
        first = send(conn, state["report_id"], state["draft"])
        return {"result": "已寄出" if first else "已寄出（重複，冪等鍵擋下）"}

    def rejected(state):
        CALLS.append("rejected")
        reason = "人工拒絕" if state["answer"] == "reject" else f"回答無法辨識 {state['asks']} 次"
        return {"result": f"未寄出（{reason}）"}

    graph = StateGraph(State)
    for name, fn in (("draft", draft), ("ask", ask), ("execute", execute), ("rejected", rejected)):
        graph.add_node(name, fn)
    graph.add_edge(START, "draft")
    graph.add_edge("draft", "ask")
    graph.add_conditional_edges("ask", after_ask, {"execute": "execute", "rejected": "rejected", "ask": "ask"})
    graph.add_edge("execute", END)
    graph.add_edge("rejected", END)
    return graph.compile(checkpointer=SqliteSaver(conn.saver))


def config(report_id):
    return {"configurable": {"thread_id": report_id}}


def start(conn, report_id, unsafe=False):
    out = build(conn, unsafe).invoke({"report_id": report_id}, config(report_id))
    return out["__interrupt__"][0].value if "__interrupt__" in out else out


def resume(conn, report_id, answer, unsafe=False):
    out = build(conn, unsafe).invoke(Command(resume=answer), config(report_id))   # 每次都重建 graph
    return out["__interrupt__"][0].value if "__interrupt__" in out else out


class Store:
    """outbox 與 checkpointer 各用一條連線。

    共用同一條連線時，checkpointer 的交易與 outbox 的 commit 會互相干擾，
    備課機實測偶發 `cannot start a transaction within a transaction`。
    """

    def __init__(self, path):
        self.outbox = open_outbox(sqlite3.connect(path, check_same_thread=False))
        self.saver = sqlite3.connect(path, check_same_thread=False)

    def close(self):
        self.outbox.close()
        self.saver.close()


def connect(path):
    return Store(path)


def demo(db):
    import importlib.metadata as metadata
    print("langgraph:", metadata.version("langgraph"), "｜ checkpoint-sqlite:", metadata.version("langgraph-checkpoint-sqlite"))

    print("\n== 一、核准：中斷 → 關掉連線 → 新連線、新 graph 恢復 ==")
    conn = connect(db)
    CALLS.clear()
    waiting = start(conn, "R-001")
    print("等待人工:", waiting["report_id"], "選項", waiting["choices"])
    print("寄出前 outbox（寄出, 嘗試）:", outbox_count(conn))
    conn.close()                                            # 模擬程序結束
    conn = connect(db)                                      # 新程序：只靠 SQLite 檔
    final = resume(conn, "R-001", "approve")
    print("結果:", final["result"], "｜ outbox:", outbox_count(conn))
    print("節點執行順序:", CALLS)

    print("\n== 二、拒絕：真的不寄 ==")
    CALLS.clear()
    start(conn, "R-002")
    final = resume(conn, "R-002", "reject")
    print("結果:", final["result"], "｜ outbox:", outbox_count(conn), "｜ execute 執行次數:", CALLS.count("execute"))

    print("\n== 三、看不懂的回答：再問，不是默默當成拒絕或核准 ==")
    CALLS.clear()
    start(conn, "R-003")
    again = resume(conn, "R-003", "aprove")
    print("回 'aprove' → 再問第", again["asks"], "次")
    final = resume(conn, "R-003", "approve")
    print("回 'approve' →", final["result"], "｜ outbox:", outbox_count(conn))
    r003 = list(CALLS)
    CALLS.clear()
    start(conn, "R-004")
    for reply in ("ok", "好", "yes"):
        out = resume(conn, "R-004", reply)
    print("連 3 次看不懂 →", out["result"])

    print("\n== 四、resume 會重跑 interrupt 所在的節點 ==")
    print("R-003 節點紀錄:", r003)
    print("draft 執行", r003.count("draft"), "次；ask 執行", r003.count("ask"), "次（每次 resume 都從 ask 開頭重跑）")
    unsafe = connect(":memory:")
    CALLS.clear()
    start(unsafe, "R-009", unsafe=True)
    resume(unsafe, "R-009", "reject", unsafe=True)
    print("錯誤示範（寄出寫在 interrupt 前）: 人工拒絕，但 outbox（寄出, 嘗試）=", outbox_count(unsafe))
    print("ask 節點被執行:", CALLS.count("ask"), "次")

    print("\n== 五、結束後再 resume、重複核准 ==")
    final = resume(conn, "R-001", "approve")
    print("R-001 已結束再 resume → 結果:", final["result"], "｜ outbox:", outbox_count(conn))
    state = build(conn).get_state(config("R-001"))
    print("R-001 next:", state.next, "（空 = 已結束，不會再執行任何節點）")
    print("R-001 歷史 checkpoint 數:", len(list(build(conn).get_state_history(config("R-001")))))
    waiting = start(conn, "R-001")
    print("對已結束的 R-001 再 start → 又在等人工，第", waiting["asks"], "次問（整條流程重來）")
    final = resume(conn, "R-001", "approve")
    print("再核准一次 →", final["result"], "｜ outbox（寄出, 嘗試）:", outbox_count(conn))
    conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", help="SQLite 檔；不給就用暫存資料夾")
    parser.add_argument("command", nargs="?", choices=("start", "resume", "status"))
    parser.add_argument("report_id", nargs="?")
    parser.add_argument("answer", nargs="?")
    args = parser.parse_args(argv)
    if args.command is None:
        if args.db:
            return demo(args.db)
        with tempfile.TemporaryDirectory() as tmp:
            return demo(str(Path(tmp) / "hitl.db"))
    if not args.db or not args.report_id:
        parser.error("start／resume／status 需要 --db 與 report_id")
    conn = connect(args.db)
    if args.command == "start":
        print("等待人工:", start(conn, args.report_id))
    elif args.command == "resume":
        print("結果:", resume(conn, args.report_id, args.answer))
    print("next:", build(conn).get_state(config(args.report_id)).next, "｜ outbox:", outbox_count(conn))
    conn.close()


if __name__ == "__main__":
    main()
