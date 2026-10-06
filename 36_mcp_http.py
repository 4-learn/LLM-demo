"""第 36 節：Streamable HTTP MCP 服務。

把第 35 節讀檔的 MCP server 掛進 FastAPI，用 Uvicorn 在 127.0.0.1 啟動成獨立程序；
這支程式是另一個程序，經 HTTP 連進去。示範：啟動紀錄、discovery 與工具呼叫、
Bearer token、Origin／Host 檢查、逾時、關閉。只綁 127.0.0.1，不對外公開。
需要 mcp==2.3.0、fastapi==0.142.2、uvicorn（mcp 已帶入）；不需要模型。
"""
import asyncio
import json
import re
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import httpx2
from mcp.client.client import Client
from mcp.client.streamable_http import streamable_http_client

HERE = Path(__file__).resolve().parent

APP_SOURCE = r'''
"""FastAPI＋MCP。由 uvicorn 載入；設定全部走環境變數：MCP_TOKEN、MCP_SERVER35、MCP_ROOT。"""
import asyncio, contextlib, hmac, logging, os, runpy, sys
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mcp.server.transport_security import TransportSecuritySettings

SERVER35, ROOT = os.environ["MCP_SERVER35"], os.environ["MCP_ROOT"]
TOKEN = os.environ["MCP_TOKEN"]                      # 不寫在程式裡、不寫在命令列（ps 看得到命令列）
logging.getLogger("mcp").setLevel(logging.CRITICAL)

sys.argv = [SERVER35, ROOT, "safe"]                  # 沿用第 35 節的 server，路徑防護用 safe
server = runpy.run_path(SERVER35, run_name="server35")["server"]


@server.tool()
async def slow_report(seconds: float) -> str:
    """模擬很慢的報表（唯讀）。"""
    await asyncio.sleep(seconds)
    return f"報表完成，耗時 {seconds} 秒"


mcp_app = server.streamable_http_app(
    streamable_http_path="/",                        # 掛在 /mcp 底下，所以這裡是 /
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*"],
        allowed_origins=["http://127.0.0.1:*", "http://localhost:*"],
    ),
)


@contextlib.asynccontextmanager
async def lifespan(app):
    # 掛進別的 ASGI app 時，子 app 的 lifespan 不會自動跑；session manager 要在這裡啟動
    async with server.session_manager.run():
        print("lifespan：MCP session manager 已啟動", flush=True)
        yield
    print("lifespan：MCP session manager 已關閉", flush=True)


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def bearer_auth(request, call_next):
    if request.url.path.startswith("/mcp"):
        given = request.headers.get("authorization", "")
        if not hmac.compare_digest(given.encode(), f"Bearer {TOKEN}".encode()):   # 固定時間比對
            return JSONResponse({"error": "unauthorized"}, status_code=401,
                                headers={"WWW-Authenticate": "Bearer"})
    return await call_next(request)


@app.get("/healthz")
def healthz():
    return {"ok": True}                              # 不需要 token，也不透露任何資料


app.mount("/mcp", mcp_app)
'''


def lesson(nn):
    """同第 32 節：先找 NN_*.py，再找講義 NN-*.md。"""
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


L35 = lesson("35")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_service(base, token):
    root = L35.build_workspace(base)
    (base / "server35.py").write_text(L35.SERVER_SOURCE, encoding="utf-8")
    (base / "app.py").write_text(APP_SOURCE, encoding="utf-8")
    port = free_port()
    log = open(base / "uvicorn.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port),
         "--log-level", "info", "--no-access-log"],
        cwd=base, stdout=log, stderr=subprocess.STDOUT,
        env={"MCP_TOKEN": token, "MCP_SERVER35": str(base / "server35.py"), "MCP_ROOT": str(root),
             "PATH": "", "PYTHONPATH": str(base), "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
    )
    return proc, port, log, root


def wait_ready(port, proc, deadline=20):
    start = time.monotonic()
    while time.monotonic() - start < deadline:
        if proc.poll() is not None:
            raise RuntimeError("服務啟動失敗，見 uvicorn.log")
        try:
            if httpx2.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                return
        except httpx2.TransportError:
            time.sleep(0.2)
    raise TimeoutError("服務沒有在時限內就緒")


def log_lines(base):
    """啟動紀錄：去掉每次不同的 PID 與埠號。"""
    text = (base / "uvicorn.log").read_text(encoding="utf-8")
    text = re.sub(r"\[\d+\]", "[PID]", text)
    return re.sub(r"127\.0\.0\.1:\d+", "127.0.0.1:PORT", text).splitlines()


def mcp_client(url, token, timeout=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    http = httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(30, read=timeout or 300))
    return Client(streamable_http_client(url, http_client=http)), http


def root_cause(error):
    while getattr(error, "exceptions", None):
        error = error.exceptions[0]
    return error


async def scenarios(port, token):
    url = f"http://127.0.0.1:{port}/mcp/"

    print("\n== 二、另一個程序經 HTTP：discovery 與工具呼叫 ==")
    client, http = mcp_client(url, token)
    async with http, client:
        names = sorted(t.name for t in (await client.list_tools()).tools)
        print("  工具：", names)
        result = await client.call_tool("count_events", {"site": "A"})
        print("  count_events(A)：", json.dumps(json.loads(result.content[0].text), ensure_ascii=False))
        result = await client.call_tool("read_file", {"path": "../secret.txt"})
        print("  read_file(../secret.txt)：", "錯誤" if result.is_error else "讀到！",
              result.content[0].text.replace("Error executing tool ", ""))

    print("\n== 三、在 MCP 之前就擋下的請求（直接送 HTTP） ==")
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    base_headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    cases = (
        ("沒有 token", {}),
        ("錯的 token", {"Authorization": "Bearer wrong"}),
        ("token 對，Origin 是外站", {"Authorization": f"Bearer {token}", "Origin": "https://evil.example"}),
        ("token 對，Host 不在清單", {"Authorization": f"Bearer {token}", "Host": "attacker.example"}),
    )
    async with httpx2.AsyncClient(timeout=5) as raw:
        for label, extra in cases:
            r = await raw.post(url, json=body, headers={**base_headers, **extra})
            print(f"  HTTP {r.status_code}  {r.text[:24]:<24}  ← {label}")

    print("\n== 四、MCP client 用錯 token：連線階段就失敗 ==")
    client, http = mcp_client(url, "wrong")
    try:
        async with http, client:
            await client.list_tools()
        print("  竟然連上了！")
    except Exception as error:
        error = root_cause(error)
        print(f"  {type(error).__name__}：{str(error)[:60]}")

    print("\n== 五、逾時：client 讀取上限 1 秒，工具要 3 秒 ==")
    client, http = mcp_client(url, token, timeout=1)
    started = time.monotonic()
    try:
        async with http, client:
            await client.call_tool("slow_report", {"seconds": 3})
        print("  沒有逾時")
    except Exception as error:
        waited = time.monotonic() - started
        print(f"  {type(root_cause(error)).__name__}，{'不到 3 秒就放棄（沒等工具做完）' if waited < 3 else f'等了 {waited:.0f} 秒'}")
    client, http = mcp_client(url, token)
    async with http, client:
        print("  服務仍可用：", (await client.call_tool("slow_report", {"seconds": 0.1})).content[0].text)


def main():
    import logging
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    logging.getLogger("httpx2").setLevel(logging.CRITICAL)
    token = secrets.token_urlsafe(24)                 # 每次啟動產生；正式環境由祕密管理系統給
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        proc, port, log, root = start_service(base, token)
        try:
            wait_ready(port, proc)
            print("== 一、啟動紀錄（PID、埠號已遮蔽） ==")
            for line in log_lines(base):
                print("  " + line)
            asyncio.run(scenarios(port, token))
        finally:
            proc.send_signal(signal.SIGINT)           # 等同在終端機按 Ctrl-C
            code = proc.wait(timeout=20)
            log.close()
        print("\n== 六、關閉 ==")
        tail = log_lines(base)
        for line in tail[tail.index(next(l for l in tail if "Shutting down" in l)):]:
            print("  " + line)
        print(f"  程序結束碼 {code}；埠 {'仍在聽' if probe(port) else '已關閉'}")


def probe(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


if __name__ == "__main__":
    main()
