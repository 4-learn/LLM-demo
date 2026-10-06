"""第 30 節：Trace 觀測與本地免費備援（只用 Python 標準函式庫）。

一條小流程：檢索 → 生成（劇本）→ 政策檢查 → 工具（含重試）。
每一步記一個 span，寫成 JSONL，再載入 SQLite 查詢「錯在哪一步」。
不需要 LangSmith 帳號，也不上傳任何資料；時間由 FakeClock 決定，所以輸出可重現。
"""

import json
import re
import sqlite3
import tempfile
from pathlib import Path

MODEL = "Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775"   # 第 05 節核准的 revision
PHONE = re.compile(r"09\d{2}-?\d{3}-?\d{3}")
EMAIL = re.compile(r"[\w.]+@[\w.]+\.\w+")
SENSITIVE_KEYS = {"phone", "email", "password", "token"}


class FakeClock:
    def __init__(self):
        self.now = 0

    def advance(self, ms):
        self.now += ms


class Tracer:
    def __init__(self, clock, mask, link_parent=True):
        self.clock, self.mask, self.link_parent = clock, mask, link_parent
        self.spans, self.stack, self.counter = [], [], 0

    def start(self, trace_id, name, **attrs):
        self.counter += 1
        parent = self.stack[-1]["span_id"] if self.stack and self.link_parent else None
        span = {"trace_id": trace_id, "span_id": f"s{self.counter:02d}", "parent_id": parent, "name": name,
                "start_ms": self.clock.now, "end_ms": None, "status": "ok", "attrs": self.mask(attrs)}
        self.stack.append(span)
        self.spans.append(span)
        return span

    def end(self, span, status="ok", **attrs):
        span["end_ms"], span["status"] = self.clock.now, status
        span["attrs"].update(self.mask(attrs))
        self.stack.remove(span)

    def jsonl(self):
        return "\n".join(json.dumps(s, ensure_ascii=False, sort_keys=True) for s in self.spans)


def mask_by_key(attrs):
    """第一版：只看欄位名稱。"""
    return {k: ("***" if k in SENSITIVE_KEYS else v) for k, v in attrs.items()}


def mask_by_value(attrs):
    """第二版：欄位名稱＋內容樣式。自由文字裡的電話、email 也要遮。"""
    out = {}
    for k, v in mask_by_key(attrs).items():
        if isinstance(v, str):
            v = EMAIL.sub("<email>", PHONE.sub("<phone>", v))
        out[k] = v
    return out


# ---- 流程 ----
POLICY = {"notify_site": {"roles": {"manager"}}, "create_ticket": {"roles": {"staff", "manager"}}}
SCRIPT = {   # 劇本模型：每個請求要呼叫哪個工具
    "T1": {"tool": "create_ticket", "prompt_tokens": 412, "completion_tokens": 38},
    "T2": {"tool": "notify_site", "prompt_tokens": 398, "completion_tokens": 41},
    "T3": {"tool": "create_ticket", "prompt_tokens": 455, "completion_tokens": 36},
}
TOOL_FAULTS = {"T3": ["timeout", "timeout", "timeout"]}
REQUESTS = (
    ("T1", "staff",   "A 區感測器離線，請開單。聯絡人 0912-345-678"),
    ("T2", "staff",   "請通知 B 工地停工"),
    ("T3", "manager", "C 區閘道器連不上，請開單，回覆寄 lin@example.com"),
)


def handle(tracer, clock, trace_id, role, text, phone="0912-345-678"):
    root = tracer.start(trace_id, "request", role=role, user_text=text, phone=phone)
    span = tracer.start(trace_id, "retrieve", top_k=3)
    clock.advance(35)
    tracer.end(span, hits=["C011", "C012"] if trace_id != "T2" else [])
    plan = SCRIPT[trace_id]
    span = tracer.start(trace_id, "generate", model=MODEL)
    clock.advance(900)
    tracer.end(span, prompt_tokens=plan["prompt_tokens"], completion_tokens=plan["completion_tokens"],
               tool_call=plan["tool"])
    span = tracer.start(trace_id, "policy", tool=plan["tool"], role=role)
    allowed = role in POLICY[plan["tool"]]["roles"]
    tracer.end(span, status="ok" if allowed else "denied", decision="allow" if allowed else "deny",
               reason="" if allowed else f"角色 {role} 不在 {sorted(POLICY[plan['tool']]['roles'])}")
    if not allowed:
        tracer.end(root, status="denied")
        return
    tool = tracer.start(trace_id, f"tool:{plan['tool']}", idempotency_key=trace_id)
    faults = list(TOOL_FAULTS.get(trace_id, []))
    for attempt in range(1, 4):
        span = tracer.start(trace_id, "attempt", n=attempt)
        clock.advance(2000 if faults else 120)
        if faults:
            faults.pop(0)
            tracer.end(span, status="error", error="timeout")
            if attempt < 3:                                # 最後一次失敗後不再退避
                clock.advance(500 * 2 ** (attempt - 1))   # 退避（第 28 節）
            continue
        tracer.end(span)
        tracer.end(tool, attempts=attempt)
        tracer.end(root)
        return
    tracer.end(tool, status="error", attempts=3)
    tracer.end(root, status="error")


def run(mask, link_parent=True):
    clock = FakeClock()
    tracer = Tracer(clock, mask, link_parent)
    for trace_id, role, text in REQUESTS:
        handle(tracer, clock, trace_id, role, text)
    return tracer


def leaks(jsonl):
    return sorted(set(PHONE.findall(jsonl) + EMAIL.findall(jsonl)))


def orphans(spans):
    roots = [s for s in spans if s["parent_id"] is None]
    return sum(1 for s in roots if s["name"] != "request")


def load_sqlite(jsonl):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE span (trace_id, span_id, parent_id, name, start_ms, end_ms, status, attrs)")
    for line in jsonl.splitlines():
        s = json.loads(line)
        db.execute("INSERT INTO span VALUES (?,?,?,?,?,?,?,?)",
                   (s["trace_id"], s["span_id"], s["parent_id"], s["name"], s["start_ms"], s["end_ms"],
                    s["status"], json.dumps(s["attrs"], ensure_ascii=False)))
    return db


def main():
    print(f"模型 {MODEL.split('@')[0]}（劇本，非真實推論）｜時間由 FakeClock 決定")
    print()

    print("=== 一、遮罩：只看欄位名稱 vs 也看內容 ===")
    for label, mask in (("只看欄位名稱", mask_by_key), ("名稱＋內容樣式", mask_by_value)):
        tracer = run(mask)
        print(f"  {label:<8} span {len(tracer.spans)} 個｜JSONL 中仍可見的個資：{leaks(tracer.jsonl()) or '無'}")
    print("  phone 欄位兩版都遮了；漏的是使用者在自由文字裡寫的電話與 email。")
    print()

    print("=== 二、父子關係：忘了傳 parent ===")
    for label, link in (("沒傳 parent", False), ("有傳 parent", True)):
        spans = run(mask_by_value, link_parent=link).spans
        print(f"  {label}：根 span {sum(1 for s in spans if s['parent_id'] is None)} 個，其中不是 request 的孤兒 {orphans(spans)} 個")
    print("  孤兒 span 不知道屬於哪一步，失敗的 attempt 看不出是哪個工具、哪次請求。")
    print()

    tracer = run(mask_by_value)
    with tempfile.TemporaryDirectory() as tmp:            # 教學示範：寫到暫存目錄，結束即刪
        out = Path(tmp) / "trace.jsonl"
        out.write_text(tracer.jsonl() + "\n", encoding="utf-8")
        db = load_sqlite(out.read_text(encoding="utf-8"))
    print(f"=== 三、寫成 {out.name}（{len(tracer.spans)} 行），載入 SQLite（{sqlite3.sqlite_version}）查詢 ===")
    print("  每個 trace 的結果，與最早出問題的 span：")
    for trace_id, status in db.execute("SELECT trace_id, status FROM span WHERE name='request' ORDER BY trace_id"):
        # 最早出問題、且底下沒有出問題子 span 的那一個：失敗的源頭，不是一路往上傳的外層
        first = db.execute("""SELECT s.span_id, s.name, s.status, s.attrs, p.name FROM span s
                              LEFT JOIN span p ON p.span_id = s.parent_id
                              WHERE s.trace_id=? AND s.status!='ok' AND NOT EXISTS
                                (SELECT 1 FROM span c WHERE c.parent_id=s.span_id AND c.status!='ok')
                              ORDER BY s.start_ms, s.span_id LIMIT 1""", (trace_id,)).fetchone()
        if first is None:
            where = "—"
        else:
            attrs = json.loads(first[3])
            where = f"{first[0]} {first[4]} › {first[1]}（{first[2]}）{attrs.get('reason') or attrs.get('error', '')}"
        total = db.execute("SELECT end_ms-start_ms FROM span WHERE trace_id=? AND name='request'", (trace_id,)).fetchone()[0]
        print(f"    {trace_id} {status:<7} {total:>5} ms｜{where}")
    print()
    print("  T3 的時間花在哪裡（self time＝自己的時間扣掉子 span）：")
    rows = db.execute("""SELECT p.name, p.span_id, (p.end_ms-p.start_ms) -
                                COALESCE((SELECT SUM(c.end_ms-c.start_ms) FROM span c WHERE c.parent_id=p.span_id), 0)
                         FROM span p WHERE p.trace_id='T3' ORDER BY 3 DESC, p.span_id""").fetchall()
    for name, span_id, self_ms in rows:
        print(f"    {span_id} {name:<20} {self_ms:>5} ms")
    tokens = db.execute("""SELECT SUM(json_extract(attrs,'$.prompt_tokens')), SUM(json_extract(attrs,'$.completion_tokens'))
                           FROM span WHERE name='generate'""").fetchone()
    print(f"  三個請求的 token：prompt {tokens[0]}、completion {tokens[1]}（劇本數字，不是實際計量）")


if __name__ == "__main__":
    main()
