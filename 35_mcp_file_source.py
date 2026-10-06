"""第 35 節：讀真實檔案的 MCP Server。

server 是獨立子程序（stdio，同第 33 節），每次呼叫都從磁碟讀資料，不寫死答案。
示範：來源雜湊、改檔後結果跟著變、路徑防護（..、絕對路徑、同字首目錄、symlink）、
不存在的 ID、回傳量上限。資料全是程式當場寫出的合成檔，不是任何真實工地。
需要 mcp==2.3.0；不需要模型。
"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters

SERVER_SOURCE = r'''
import csv, hashlib, json, logging, os, sys
from pathlib import Path
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

ROOT = Path(sys.argv[1]).resolve()        # 允許讀取的資料目錄；由啟動 server 的人決定
GUARD = sys.argv[2]                       # 路徑檢查的寫法；只有 "safe" 是正確的
MAX_CHARS = 300                           # 單次回傳上限
logging.getLogger("mcp").setLevel(logging.CRITICAL)   # 拒絕是預期結果，不要每次都印到 stderr

server = MCPServer("site-files", version="0.1.0")


def sha(data):
    return hashlib.sha256(data).hexdigest()[:12]


def locate(relative):
    """把使用者（模型）給的相對路徑換成磁碟路徑，並確認仍在 ROOT 內。"""
    # ---- 三種常見的錯誤寫法（教學用）----
    if GUARD == "concat":                 # 只把字串接起來，沒有檢查
        return Path(str(ROOT) + "/" + relative)
    if GUARD == "prefix":                 # 有 resolve，但用字串比字首：/data-private 也以 /data 開頭
        path = (ROOT / relative).resolve()
        if not str(path).startswith(str(ROOT)):
            raise ToolError("拒絕：路徑在資料目錄外")
        return path
    if GUARD == "normpath":               # 只在字面上消去 ..，不解 symlink
        path = Path(os.path.normpath(ROOT / relative))
        if not str(path).startswith(str(ROOT) + os.sep):
            raise ToolError("拒絕：路徑在資料目錄外")
        return path
    # ---- 正確寫法 ----
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ToolError("拒絕：只接受資料目錄內的相對路徑")
    path = (ROOT / candidate).resolve()   # 先解開 .. 與 symlink，再比對
    if not path.is_relative_to(ROOT):
        raise ToolError("拒絕：路徑在資料目錄外")
    return path


def read(relative):
    path = locate(relative)
    if not path.is_file():
        raise ToolError(f"查無檔案：{relative}")
    data = path.read_bytes()
    return path, data


@server.tool()
def list_sources() -> dict:
    """列出資料目錄內的檔案與內容雜湊（唯讀）。"""
    return {"files": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
            for p in sorted(ROOT.rglob("*")) if p.is_file() and not p.is_symlink()]}


@server.tool()
def count_events(site: str) -> dict:
    """從 events.csv 計算某工地的事件數（每次呼叫都重新讀檔）。"""
    _, data = read("events.csv")
    rows = list(csv.DictReader(data.decode("utf-8").splitlines()))
    return {"site": site, "count": sum(r["site"] == site for r in rows),
            "source": {"path": "events.csv", "sha256": sha(data), "rows": len(rows)}}


@server.tool()
def get_doc(doc_id: str) -> dict:
    """依 index.json 的文件 ID 取文件內容。"""
    _, index = read("index.json")
    entry = json.loads(index).get(doc_id)
    if entry is None:
        raise ToolError(f"查無文件 ID：{doc_id}")
    return read_file(entry)


@server.tool()
def read_file(path: str) -> dict:
    """讀資料目錄內的一個文字檔。"""
    _, data = read(path)
    text = data.decode("utf-8", errors="replace")
    result = {"path": path, "sha256": sha(data), "text": text[:MAX_CHARS]}
    if len(text) > MAX_CHARS:
        result["truncated"] = f"原長 {len(text)} 字，只回前 {MAX_CHARS} 字"
    return result


if __name__ == "__main__":
    server.run("stdio")
'''

EVENTS_CSV = """id,site,date,type
E01,A,2026-10-01,blocked_exit
E02,A,2026-10-02,no_helmet
E03,B,2026-10-02,no_helmet
E04,A,2026-10-03,no_helmet
"""
INDEX = {"SOP-01": "docs/ladder.md", "SOP-02": "docs/exit.md", "SOP-03": "../secret.txt"}
DOCS = {
    "docs/ladder.md": "# 合梯作業\n使用合梯前確認開閉防滑裝置已扣上，作業高度 2 公尺以上須另行評估。\n",
    "docs/exit.md": "# 逃生通道\n" + "逃生通道寬度內不得堆放物料，每日收工前巡查一次。\n" * 12,
}
SECRET = "這是資料目錄外的檔案：不該被讀到。\n"


def build_workspace(base):
    """在暫存資料夾裡排出：允許的 data/、旁邊的 secret.txt、字首相同的 data-private/、指向外面的 symlink。"""
    root = base / "data"
    (root / "docs").mkdir(parents=True)
    (root / "events.csv").write_text(EVENTS_CSV, encoding="utf-8")
    (root / "index.json").write_text(json.dumps(INDEX, ensure_ascii=False), encoding="utf-8")
    for name, text in DOCS.items():
        (root / name).write_text(text, encoding="utf-8")
    (base / "secret.txt").write_text(SECRET, encoding="utf-8")
    (base / "data-private").mkdir()
    (base / "data-private" / "salary.csv").write_text("name,amount\n合成員工,1\n", encoding="utf-8")
    (root / "docs" / "shortcut.md").symlink_to(base / "secret.txt")
    return root


def show(result):
    text = result.content[0].text if result.content else ""
    if result.is_error:
        return "錯誤 " + text.splitlines()[0].replace("Error executing tool ", "")
    data = json.loads(text)
    if isinstance(data, dict) and "text" in data:
        data["text"] = data["text"].splitlines()[0] + ("…" if "\n" in data["text"].rstrip("\n") else "")
    return json.dumps(data, ensure_ascii=False)


ATTACKS = (
    ("正常檔案", "docs/ladder.md"),
    ("上層跳出", "../secret.txt"),
    ("絕對路徑", "{base}/secret.txt"),
    ("同字首目錄", "../data-private/salary.csv"),
    ("符號連結", "docs/shortcut.md"),
)
GUARDS = ("concat", "prefix", "normpath", "safe")


def outcome(result):
    if not result.is_error:
        return "讀到"
    text = result.content[0].text
    return "拒絕" if "拒絕" in text else "查無" if "查無" in text else "錯誤"


async def demo(base):
    root = build_workspace(base)
    server_file = base / "server.py"
    server_file.write_text(SERVER_SOURCE, encoding="utf-8")

    def params(guard):
        return StdioServerParameters(command=sys.executable, args=[str(server_file), str(root), guard])

    async with Client(params("safe")) as client:
        print("== 一、資料來源：server 讀的是磁碟上的檔案 ==")
        for item in json.loads((await client.call_tool("list_sources", {})).content[0].text)["files"]:
            print(f"  {item['path']:<16} {item['bytes']:>4} bytes  sha256 {item['sha256']}")

        print("\n== 二、改檔後，同一條連線的結果跟著變 ==")
        print("  修改前", show(await client.call_tool("count_events", {"site": "A"})))
        with open(root / "events.csv", "a", encoding="utf-8") as f:
            f.write("E05,A,2026-10-04,blocked_exit\n")
        print("  修改後", show(await client.call_tool("count_events", {"site": "A"})))

        print("\n== 三、文件 ID 與回傳上限 ==")
        for doc_id in ("SOP-01", "SOP-02", "SOP-99", "SOP-03"):
            print(f"  {doc_id:<7}", show(await client.call_tool("get_doc", {"doc_id": doc_id})))

    print("\n== 四、路徑防護：四種寫法 × 五種路徑 ==")
    table = {}
    for guard in GUARDS:
        async with Client(params(guard)) as client:
            for label, path in ATTACKS:
                table[label, guard] = outcome(await client.call_tool("read_file", {"path": path.format(base=base)}))
    # 每格都是全形字，終端機裡才對得齊
    print(("  " + " " * 9 + "".join(label + "　" * (6 - len(label)) for label, _ in ATTACKS)).rstrip("　"))
    for g in GUARDS:
        cells = [table[label, g] + ("！" if table[label, g] == "讀到" and label != "正常檔案" else "　") + "　" * 3
                 for label, _ in ATTACKS]
        print(("  " + g.ljust(9) + "".join(cells)).rstrip("　"))
    leaks = {g: sum(table[l, g] == "讀到" for l, _ in ATTACKS[1:]) for g in GUARDS}
    print("  目錄外的檔案被讀到：" + "、".join(f"{g} {n}/4" for g, n in leaks.items()))


def main():
    import logging
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(demo(Path(tmp)))


if __name__ == "__main__":
    main()
