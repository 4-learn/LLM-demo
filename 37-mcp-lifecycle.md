# MCP Client 生命週期與 Async

> **新版正文：20E07-E03-06｜對應新版 LLM 第 34 節：MCP Client Lifecycle 與 Async。**
> 第 33 節的 server 和 client 在同一個程序裡。實際使用時 server 常是另一個程序，透過 stdio 或網路連線。這時最常見的錯誤不是協定，而是 async：忘了 await、沒關連線、把逾時當成「沒做」。核心主張：**用 `async with` 管理連線，離開時子程序一定要結束；每個呼叫都要 await；逾時只代表 client 不等了——server 端可能還是做完了，所以寫入類工具必須搭配冪等（第 29 節）。**

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先修第 29 節（逾時與冪等）、第 33 節（MCP 角色與契約）。課前請先讀懂一段 `async def`／`await`／`asyncio.run` 的短例。

| 時間 | 內容 |
| --- | --- |
| 0–10 分鐘 | stdio transport；`async with` 的進入與離開 |
| 10–20 分鐘 | await、coroutine、事件迴圈 |
| 20–30 分鐘 | 逾時與取消：client 與 server 各看到什麼 |
| 30–35 分鐘 | 跑 Demo |
| 35–50 分鐘 | Workshop：呼叫一次、觸發逾時、確認無殘留子程序 |

驗收能力：交出已 await 的工具結果與關閉紀錄（子程序已結束），且**不用同步包裝去呼叫非同步工具**。

## 一、stdio：server 是子程序

`StdioServerParameters(command=..., args=[...])` 告訴 client 用哪個指令啟動 server。client 開出子程序，透過子程序的標準輸入／輸出交換 JSON-RPC 訊息。

```text
async with Client(params) as client:   ← 啟動子程序 → initialize → 能力協商
    await client.list_tools()
    await client.call_tool(...)
                                        ← 離開：關閉 session → 關閉 stdin → 子程序結束
```

`async with` 保證「離開」一定會執行，即使中間丟出例外。**自己手動 `__aenter__` 卻忘了 `__aexit__`，子程序就會留著**，而且備課機實測：程式結束時會卡住不退出（下面第三點）。

## 二、await 與事件迴圈

- `client.call_tool(...)` 回傳的是 **coroutine**（還沒執行的工作），不是結果。要 `await` 才會真的送出請求並拿到 `CallToolResult`。
- 所有 `await` 都在同一個**事件迴圈**裡執行。`asyncio.run()` 會建立一個新的迴圈，所以**不能在已經在跑的迴圈裡再呼叫它**。
- 常見的壞寫法：把 async 工具包成同步函式 `def call(): return asyncio.run(...)`，在一般腳本可以用，一放進 LangGraph 的 async 節點、Jupyter 或 Web 框架就失敗。

## 三、逾時與取消

| 情況 | client 看到 | server 端 |
| --- | --- | --- |
| 逾時（`read_timeout_seconds`），async 工具 | `MCPError: Request 'tools/call' timed out` | SDK 送出取消通知，工具被取消，**沒寫入** |
| 逾時，同步工具（`time.sleep`） | 同上 | 同步程式無法被中途取消，**照樣寫入** |
| 呼叫端 `task.cancel()` | `CancelledError` | 取消通知送達，async 工具被取消 |

這張表是備課機實測（下面的 Demo）。重點是第二列：**client 收到逾時，不代表沒做**。這和第 29 節的「逾時但寫入其實完成」是同一件事，只是這次發生在真實的子程序裡。所以寫入類工具要有冪等鍵，重試時才不會重複。

另一個實測：沒有關閉 client 就讓 `asyncio.run` 結束，程式會**卡住不退出**（子程序也還在）。備課機用 `timeout 20` 包著執行才結束。

## 四、可執行 Demo

需要 `mcp==2.3.0`，不需要模型。server 原始碼寫進暫存資料夾後，由子程序執行。整段約 10 秒（包含刻意等待）。

<!-- demo: 51_mcp_lifecycle.py -->
```python
"""第 34 節：MCP Client 生命週期與 async。

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
```

執行（程式也放在 [llm-demo](https://github.com/4-learn/llm-demo)）：

```bash
python 51_mcp_lifecycle.py
```

關鍵輸出（Linux x86_64／Python 3.10.15，mcp 2.3.0，20E07-E03-06）：

```text
mcp: 2.3.0 ｜ transport: stdio（server 是獨立子程序）

== 一、完整生命週期 ==
1 initialize：協定 2026-07-28 ｜ server lifecycle-lab
2 list_tools： ['whoami', 'async_write', 'sync_write']
3 call_tool：取得子程序 PID，存活 True
4 離開 async with 後，子程序結束 True

== 二、忘了 await ==
沒有 await 拿到的是: coroutine ｜ 有 content 嗎: False
await 之後拿到的是: CallToolResult ｜ isError False

== 三、逾時：client 不等了，server 呢？ ==
async_write client 收到 MCPError: Request 'tools/call' timed out
            2.5 秒後，server 端寫入了嗎：False
sync_write  client 收到 MCPError: Request 'tools/call' timed out
            2.5 秒後，server 端寫入了嗎：True
逾時後同一條連線還能用： True

== 四、取消 ==
呼叫端：CancelledError
server 端寫入了嗎： False

== 五、沒有關閉 ==
用完沒關：子程序存活 True
補上關閉：子程序結束 True
關閉後再呼叫： Client must be used within an async context manager

== 六、sync／async 邊界 ==
在事件迴圈裡呼叫 asyncio.run： asyncio.run() cannot be called from a running event loop
```

## 五、結果怎麼讀

### 1. 生命週期

初始化協商出協定 `2026-07-28`，列出 3 個工具，呼叫 `whoami` 拿到子程序 PID。離開 `async with` 後，用 PID 確認子程序已經結束。這就是驗收要交的「關閉紀錄」。

### 2. 忘了 await

拿到的是 `coroutine`，沒有 `content`。如果接著寫 `result.content[0].text`，錯誤是 `AttributeError`，看起來像工具壞了，其實根本沒送出請求。

### 3. 逾時：同一個逾時，兩種結局

`async_write` 和 `sync_write` 都在 client 端逾時，錯誤訊息一模一樣。但 2.5 秒後，async 版沒寫入、同步版寫入了。**從 client 端分不出這兩種情況**，所以只能假設「可能已經做了」。

逾時後同一條連線還能繼續用。

### 4. 取消

`task.cancel()` 讓呼叫端收到 `CancelledError`，async 工具在 server 端也被取消，沒有寫入。

### 5. 沒有關閉

手動進入、用完沒關時，子程序還活著。補上 `__aexit__` 後子程序才結束；關閉後再呼叫會得到 `Client must be used within an async context manager`。

### 6. sync／async 邊界

在已經執行中的事件迴圈裡呼叫 `asyncio.run`，直接 `RuntimeError`。這是舊教材「async 節點裡用同步 invoke」問題的根源。

## 六、常見誤判

| 說法 | 為什麼不對 |
| --- | --- |
| 「`call_tool` 回傳結果」 | 回傳 coroutine，要 await |
| 「逾時了就是沒做」 | 同步工具照樣寫入；只能假設可能做了 |
| 「程式結束子程序就會自己收掉」 | 沒關閉時，備課機實測程式卡住不退出 |
| 「包成同步函式比較好用」 | 在 LangGraph、Jupyter、Web 框架的迴圈裡會 `RuntimeError` |
| 「server 用 `time.sleep` 模擬慢沒差」 | 同步工具無法被取消，逾時後仍會完成副作用 |

## Workshop：呼叫一次、觸發逾時、確認無殘留子程序

### 任務

1. 用 `async with` 連上 Demo 的 server，列工具並呼叫一次 `whoami`。離開後用 PID 證明子程序已結束。
2. 把 `read_timeout_seconds` 改成 `3.0`，再呼叫 `async_write(seconds=1.5)`。結果是什麼？逾時要設多長，由誰決定？
3. 不用 `read_timeout_seconds`，改用 `asyncio.wait_for(client.call_tool(...), 0.5)` 在外面包逾時。server 端寫入了嗎？兩種寫法差在哪？
4. 寫一個同步包裝 `def whoami_sync(): return asyncio.run(...)`。在一般腳本呼叫一次，再從 `async def` 裡呼叫一次，記錄結果。
5. 把 `sync_write` 改成 async（`await anyio.sleep`），重跑第三段，結果怎麼變？

### 參考判讀

以下為備課機實跑（mcp 2.3.0，20E07-E03-06）：

- 第 1 題：同第一段，子程序結束為 `True`。
- 第 2 題：回傳 `done`，檔案寫入。逾時要比工具正常耗時長，由 host 依工具性質決定（第 29 節的重試預算），不是越長越好：太長會讓使用者乾等。
- 第 3 題：`asyncio.TimeoutError`，server 端**沒有寫入**。`wait_for` 取消內部的呼叫，SDK 同樣送出取消通知。兩者差在錯誤型別（`TimeoutError` vs `MCPError`）和逾時是 client 的設定或單次呼叫的設定。對同步工具兩者都取消不了。
- 第 4 題：一般腳本可用；在 `async def` 裡呼叫得到 `RuntimeError: asyncio.run() cannot be called from a running event loop`。正確做法是讓呼叫者也是 async 並直接 await。
- 第 5 題：兩個工具都變成「逾時後沒寫入」。但這只是讓取消生效；工具若真的呼叫外部系統（寄信、資料庫），外部那一端仍可能已經完成，冪等鍵不能省。

### 驗收

- 第 1 題的輸出：工具結果（已 await）與子程序已結束的證據。
- 第 3 題的觀察：逾時的錯誤型別與 server 端是否寫入。
- 一段說明：你的工具若是寫入類，逾時後要怎麼處理才不會重複。

## 教師紀錄與下一步

- 證據等級：**本機實測**（Linux x86_64、Python 3.10.15、mcp 2.3.0，真實 stdio 子程序）。連跑 3 次輸出逐字相同、stderr 為空、執行後無殘留 server 子程序；Workshop 參考判讀逐題實跑並由測試覆蓋。
- 撰寫時的實際紀錄：
  1. 「沒關閉就結束」原本預期會留下孤兒子程序；實跑是**程式卡住不退出**，要外部 `timeout` 才結束，結束後子程序也被收掉。所以 Demo 只示範「沒關時子程序活著、補上關閉後結束」，不示範卡住。
  2. 同步工具逾時後仍寫入，是用 `time.sleep` 實測確認的；async 工具則被 SDK 的取消通知中止。這個差異來自 SDK 2.3.0 的實作，換版本要重測。
- 時間常數 `TIMEOUT=0.5`、`SLOW=1.5`、`SETTLE=2.5` 留了 3 倍以上的餘裕。學生機器若很慢，結果仍可能翻轉，請先跑一次確認。
- 舊教材 [MCP 真資料](https://hackmd.io/@yillkid/OC56rJxWRYOjcJ78K4kSmg) 的 async 節點與同步 invoke 混用，就是第六段的問題；改寫時以本節為準。
- 下一步：第 35 節把模型接進來，完成「模型提出呼叫 → MCP 執行 → 結果回模型 → 最終回答」的完整回合。
