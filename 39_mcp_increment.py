"""第 39 節：MCP 增量整合里程碑。

不另建系統：直接載入第 31 節的流程（graph、政策、HITL、重試、trace 全部不動），
只把「開單服務」換成一個 stdio MCP server。換之前、換之後跑同一組案例，對照輸出契約與副作用。
需要 langgraph==1.2.12、mcp==2.3.0；不需要模型。
"""
import asyncio
import json
import logging
import re
import sys
import tempfile
import types
from pathlib import Path

from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters

HERE = Path(__file__).resolve().parent

SERVER_SOURCE = r'''
"""開單 MCP server。用法：python ticket_server.py <工單檔> <故障劇本檔> <是否遵守冪等鍵 1/0>"""
import json, logging, sys, time
from pathlib import Path
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

logging.getLogger("mcp").setLevel(logging.CRITICAL)
STORE, FAULTS, HONOUR = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1"
server = MCPServer("tickets", version="1.0.0")


def next_fault():
    faults = json.loads(FAULTS.read_text(encoding="utf-8"))
    event = faults.pop(0) if faults else "ok"
    FAULTS.write_text(json.dumps(faults), encoding="utf-8")
    return event


@server.tool()
def create_ticket(site: str, idempotency_key: str) -> dict:
    """建立工單。同一把 idempotency_key 只會建立一張，重送時回傳原本那張。"""
    event = next_fault()
    if event == "rate_limited":
        raise ToolError("429 限流")
    if event == "unauthorized":
        raise ToolError("401 未授權")
    data = json.loads(STORE.read_text(encoding="utf-8"))
    if HONOUR and idempotency_key in data["keys"]:
        return data["keys"][idempotency_key]
    ticket = {"id": f"T{len(data['tickets']) + 1}", "site": site}
    data["tickets"].append(ticket)
    data["keys"][idempotency_key] = ticket
    STORE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")   # 先寫入
    if event == "timeout_after_write":
        time.sleep(30)                                                          # 回應遲遲不來：client 會逾時
    return ticket


server.run("stdio")
'''


def lesson(nn):
    """同第 31 節：先找 NN_*.py，再找講義 NN-*.md；讀原始碼後 exec。"""
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


L31 = lesson("31")
RETRY = L31.RETRY
READ_TIMEOUT = 3          # 秒；每次 tools/call 等回應的上限


def root_cause(error):
    while getattr(error, "exceptions", None):
        error = error.exceptions[0]
    return error


class McpTickets:
    """與第 28 節 TicketService 相同的介面（create、tickets），底下改走 MCP。

    介面一樣，流程就不用改；錯誤要翻譯回第 28 節的兩類，重試規則才會照舊。
    """

    def __init__(self, base, failures=(), honour_keys=True, command=sys.executable):
        self.store = base / "tickets.json"
        self.store.write_text(json.dumps({"tickets": [], "keys": {}}), encoding="utf-8")
        faults = base / "faults.json"
        faults.write_text(json.dumps(list(failures)), encoding="utf-8")
        server_file = base / "ticket_server.py"
        server_file.write_text(SERVER_SOURCE, encoding="utf-8")
        self.params = StdioServerParameters(command=command,
                                            args=[str(server_file), str(self.store), str(faults), "1" if honour_keys else "0"])
        self.calls = []                                 # 每次呼叫的結果：審計用

    @property
    def tickets(self):
        """副作用以磁碟上的工單檔為準，不以 client 收到的回應為準。"""
        return json.loads(self.store.read_text(encoding="utf-8"))["tickets"]

    def create(self, site, key):
        async def go():
            async with Client(self.params, read_timeout_seconds=READ_TIMEOUT) as client:
                return await client.call_tool("create_ticket", {"site": site, "idempotency_key": key})

        try:
            result = asyncio.run(go())
        except Exception as error:
            cause = root_cause(error)
            text = str(cause)
            if "Timed out" in text or "timed out" in text.lower():
                self.calls.append("逾時")
                raise RETRY.TransientError("MCP 逾時（寫入可能已完成）")
            self.calls.append(f"連線失敗 {type(cause).__name__}")
            raise RETRY.TransientError(f"MCP 連線失敗（{type(cause).__name__}）")
        text = result.content[0].text
        if result.is_error:
            self.calls.append(text.split(": ", 1)[-1])
            if "401" in text:
                raise RETRY.PermanentError("401 未授權")
            raise RETRY.TransientError("429 限流")
        self.calls.append("成功")
        return json.loads(text)


def run_case(case, base=None, **mcp):
    """base 為 None：第 31 節原樣（本地服務）；否則把 world.tickets 換成 McpTickets。"""
    case_id, _, role, action, qid, script, answer, faults, _, _ = case
    world = L31.World(faults)
    if base is not None:
        world.tickets = McpTickets(base, faults, **mcp)
    app = L31.build(world, script)
    config = {"configurable": {"thread_id": case_id}}
    out = app.invoke({"case_id": case_id, "role": role, "action": action, "qid": qid}, config)
    if "__interrupt__" in out:
        out = app.invoke(L31.Command(resume=answer), config)
    return out, world


def summary(out, world):
    statuses = ",".join(f"{s['name']}={s['status']}" for s in world.tracer.spans)
    return out["result"], world.side_effects(), statuses


def main():
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    import importlib.metadata as metadata
    print("langgraph:", metadata.version("langgraph"), "｜mcp:", metadata.version("mcp"),
          "｜沿用第 31 節流程，只換開單服務")

    print("\n== 一、替換前後：同一組案例，比結果、副作用、trace ==")
    same = 0
    rows = {}
    for case in L31.CASES:
        with tempfile.TemporaryDirectory() as tmp:
            before = summary(*run_case(case))
            out, world = run_case(case, Path(tmp))
            after = summary(out, world)
            rows[case[0]] = world.tickets.calls
        same += before == after
        print(f"  {case[0]:<9}{'相同' if before == after else '不同！'}  {after[0]}｜副作用 {after[1]}")
        if before != after:
            print(f"           替換前：{before}")
    print(f"  {same}／{len(L31.CASES)} 案例輸出契約相同")

    print("\n== 二、MCP 呼叫紀錄（client 端每次呼叫的結果） ==")
    for case_id, calls in rows.items():
        print(f"  {case_id:<9}{'、'.join(calls) if calls else '未呼叫 MCP'}")

    print("\n== 三、Workshop 三類：成功、連線失敗、人工拒絕 ==")
    cases = {c[0]: c for c in L31.CASES}
    with tempfile.TemporaryDirectory() as tmp:
        out, world = run_case(cases["I-OK"], Path(tmp))
        print(f"  成功      {out['result']}｜工單檔 {world.tickets.tickets}")
    with tempfile.TemporaryDirectory() as tmp:
        out, world = run_case(cases["I-OK"], Path(tmp), command=str(Path(tmp) / "沒有這個程式"))
        span = next(s for s in world.tracer.spans if s["name"] == "tool:create_ticket")
        print(f"  連線失敗  {out['result']}")
        print(f"            副作用 {world.side_effects()}｜trace {span['status']}，嘗試 {span['attrs']['attempts']} 次")
    with tempfile.TemporaryDirectory() as tmp:
        out, world = run_case(cases["I-REJ"], Path(tmp))
        print(f"  人工拒絕  {out['result']}｜MCP 呼叫 {len(world.tickets.calls)} 次｜工單檔 {world.tickets.tickets}")

    print("\n== 四、換了 transport，契約有沒有跟著搬過去？ ==")
    with tempfile.TemporaryDirectory() as tmp:
        out, world = run_case(cases["I-RETRY"], Path(tmp), honour_keys=False)
        print(f"  server 不遵守冪等鍵、I-RETRY：{out['result']}｜工單檔實際 {len(world.tickets.tickets)} 張"
              f"（{[t['id'] for t in world.tickets.tickets]}）")
    with tempfile.TemporaryDirectory() as tmp:
        staff_ticket = ("I-STAFF", "正常", "visitor", "create_ticket", "Q02", "A1", None, (), "", 0)
        out, world = run_case(staff_ticket, Path(tmp))
        print(f"  角色 visitor 開單：{out['result']}｜MCP 呼叫 {len(world.tickets.calls)} 次")

    print("\n== 五、尚未完成（遠端驗證） ==")
    for gap in ("本節只用 stdio；改走第 36 節的 Streamable HTTP 時，token 與 Origin 檢查要重測一次",
                "server 的權限是「能啟動它的人」；沒有每位使用者的身分，角色檢查仍只在 host 端",
                "工單檔是本機 JSON，不是真實工單系統；冪等鍵沒有到期時間",
                "逾時後 server 程序被結束，但寫入已經發生：要靠冪等鍵，不能靠「逾時就當沒做」"):
        print("-", gap)


if __name__ == "__main__":
    main()
