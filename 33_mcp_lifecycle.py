"""第 33 節：MCP Client 生命週期與 async。

用 stdio 把 MCP server 開成獨立子程序，示範：初始化 → 列工具 → 呼叫 → 關閉；
忘了 await、逾時、取消、沒關閉、在事件迴圈裡呼叫 asyncio.run 的實際行為。
需要 mcp==2.3.0；不需要模型。server 程式寫進暫存資料夾後由子程序執行。
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters

SERVER_SOURCE = '''
import os, time
import anyio
from mcp.server.mcpserver import MCPServer

server = MCPServer("lifecycle-lab", version="0.1.0")


@server.tool()
def whoami() -> int:
    """回傳 server 子程序的 PID（用來確認關閉後子程序是否還在）。"""
    return os.getpid()


@server.tool()
async def async_write(path: str, seconds: float) -> str:
    """等 seconds 秒後寫一行到 path（async 工具）。"""
    await anyio.sleep(seconds)
    with open(path, "a", encoding="utf-8") as f:
        f.write("written\\\\n")
    return "done"


@server.tool()
def sync_write(path: str, seconds: float) -> str:
    """等 seconds 秒後寫一行到 path（同步工具：time.sleep 會卡住執行緒）。"""
    time.sleep(seconds)
    with open(path, "a", encoding="utf-8") as f:
        f.write("written\\\\n")
    return "done"


if __name__ == "__main__":
    server.run("stdio")
'''

TIMEOUT = 0.5        # client 等待單次回應的上限（秒）
SLOW = 1.5           # 慢工具要跑的時間；遠大於 TIMEOUT，避免機器忙時結果翻轉
SETTLE = 2.5         # 逾時後再等多久，看 server 端是否仍完成寫入


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def wait_gone(pid, limit=3.0):
    """關閉後子程序要一點時間結束；最多等 limit 秒。"""
    for _ in range(int(limit / 0.05)):
        if not alive(pid):
            return True
        await asyncio.sleep(0.05)
    return False


async def demo(workdir):
    import importlib.metadata as metadata
    server_file = workdir / "lifecycle_server.py"
    server_file.write_text(SERVER_SOURCE, encoding="utf-8")
    params = StdioServerParameters(command=sys.executable, args=[str(server_file)])
    print("mcp:", metadata.version("mcp"), "｜ transport: stdio（server 是獨立子程序）")

    print("\n== 一、完整生命週期 ==")
    async with Client(params, read_timeout_seconds=TIMEOUT) as client:
        print("1 initialize：協定", client.protocol_version, "｜ server", client.server_info.name)
        names = [t.name for t in (await client.list_tools()).tools]
        print("2 list_tools：", names)
        result = await client.call_tool("whoami", {})
        pid = int(result.content[0].text)
        print("3 call_tool：取得子程序 PID，存活", alive(pid))
    print("4 離開 async with 後，子程序結束", await wait_gone(pid))

    print("\n== 二、忘了 await ==")
    async with Client(params) as client:
        pending = client.call_tool("whoami", {})
        print("沒有 await 拿到的是:", type(pending).__name__, "｜ 有 content 嗎:", hasattr(pending, "content"))
        pending.close()                                   # 不 await 也不關，Python 會警告 never awaited
        result = await client.call_tool("whoami", {})
        print("await 之後拿到的是:", type(result).__name__, "｜ isError", result.is_error)

    print("\n== 三、逾時：client 不等了，server 呢？ ==")
    async with Client(params, read_timeout_seconds=TIMEOUT) as client:
        for tool in ("async_write", "sync_write"):
            target = workdir / f"{tool}.txt"
            try:
                await client.call_tool(tool, {"path": str(target), "seconds": SLOW})
                print(f"{tool:<11} 沒有逾時（不符預期）")
            except Exception as error:
                print(f"{tool:<11} client 收到 {type(error).__name__}: {error}")
            await asyncio.sleep(SETTLE)
            print(f"{'':<11} {SETTLE} 秒後，server 端寫入了嗎：{target.exists()}")
        result = await client.call_tool("whoami", {})
        print("逾時後同一條連線還能用：", not result.is_error)

    print("\n== 四、取消 ==")
    async with Client(params) as client:
        target = workdir / "cancelled.txt"
        task = asyncio.create_task(client.call_tool("async_write", {"path": str(target), "seconds": 0.6}))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            print("呼叫端：CancelledError")
        await asyncio.sleep(1.0)
        print("server 端寫入了嗎：", target.exists())

    print("\n== 五、沒有關閉 ==")
    client = Client(params)
    await client.__aenter__()                             # 錯誤示範：手動進入、不用 async with
    pid = int((await client.call_tool("whoami", {})).content[0].text)
    print("用完沒關：子程序存活", alive(pid))
    await client.__aexit__(None, None, None)              # 補上關閉；沒有這行，程式結束時會卡住
    print("補上關閉：子程序結束", await wait_gone(pid))
    try:
        await client.call_tool("whoami", {})
    except RuntimeError as error:
        print("關閉後再呼叫：", error)

    print("\n== 六、sync／async 邊界 ==")
    probe = asyncio.sleep(0)
    try:
        asyncio.run(probe)                                # 常見於「把 async 工具包成同步函式」
    except RuntimeError as error:
        print("在事件迴圈裡呼叫 asyncio.run：", error)
    finally:
        probe.close()


def main():
    import logging
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(demo(Path(tmp)))


if __name__ == "__main__":
    main()
