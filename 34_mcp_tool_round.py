"""第 34 節：LLM 與 MCP 的完整工具回合。

模型看到工具 → 提出呼叫 → host 檢查 → MCP server 執行 → 結果「回到模型」→ 模型寫最終答覆。
第 13 節用劇本扮演模型、第 32／33 節只有 MCP；本節把兩者接起來，模型是本機的
Qwen2.5-0.5B-Instruct。

    python 34_mcp_tool_round.py          # 預設：重播示範機錄下的模型輸出（fixtures），只需要 mcp
    python 34_mcp_tool_round.py --live   # 真的跑本機模型（需要 transformers 與已下載的權重）

模型的每一句輸出在 trace 都標來源：[錄製] 是示範機實際跑出、凍結成 fixtures 的文字；
[即時] 是這次執行產生的；[構造] 是教師為了測試 host 分支手寫的，不是任何模型產生的。
"""
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Literal

from mcp.client.client import Client
from mcp.server.mcpserver import MCPServer

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "data" / "34-tool-round-fixtures.json"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"   # 與第 02、05、06 節相同

SYSTEM = "你是工地安全查詢助理。需要數字時呼叫工具，只根據工具結果回答，使用繁體中文。"
EXPOSED = ("count_events", "list_events")   # host 決定給模型看哪些工具；delete_event 不給
MAX_ROUNDS = 3                               # 模型最多可以要求幾輪工具
MAX_RESULT_CHARS = 240                       # 回給模型的工具結果上限（字元）

# 合成事件資料（不是任何真實工地）
EVENTS = [
    {"id": f"E{i:02d}", "site": site, "date": f"2026-10-{day:02d}", "type": kind}
    for i, (site, day, kind) in enumerate([
        ("A", 1, "blocked_exit"), ("A", 2, "no_helmet"), ("B", 2, "no_helmet"), ("A", 3, "no_helmet"),
        ("B", 3, "blocked_exit"), ("A", 4, "no_helmet"), ("A", 5, "blocked_exit"), ("B", 5, "blocked_exit"),
        ("A", 6, "no_helmet"), ("A", 7, "no_helmet"),
    ], start=1)
]
Site = Literal["A", "B"]
Kind = Literal["blocked_exit", "no_helmet"]


def build_server(events):
    server = MCPServer("site-events", version="0.1.0")

    @server.tool()
    def count_events(site: Site, type: Kind) -> dict:
        """計算某工地某類事件的筆數（唯讀）。type：blocked_exit＝逃生通道堵塞，no_helmet＝未戴安全帽。"""
        return {"site": site, "type": type, "count": sum(e["site"] == site and e["type"] == type for e in events)}

    @server.tool()
    def list_events(site: Site) -> list:
        """列出某工地的全部事件（唯讀）。"""
        return [e for e in events if e["site"] == site]

    @server.tool()
    def delete_event(id: str) -> str:
        """刪除一筆事件（寫入）。"""
        events[:] = [e for e in events if e["id"] != id]
        return f"deleted {id}"

    return server


# ---------------------------------------------------------------- 轉接：MCP ↔ 模型

def to_model_tools(mcp_tools):
    """MCP 的 inputSchema 就是 JSON Schema，換個外殼就是模型的 function 格式。"""
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.input_schema}}
            for t in mcp_tools if t.name in EXPOSED]


CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_reply(text, round_no):
    """Qwen 把呼叫寫成 <tool_call>{json}</tool_call>；沒有 call ID，由 host 配發。"""
    calls, problems = [], []
    for i, raw in enumerate(CALL.findall(text), start=1):
        try:
            data = json.loads(raw)
            calls.append({"id": f"call_{round_no}_{i}", "name": data["name"], "arguments": data.get("arguments", {})})
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            problems.append(f"第 {i} 個呼叫無法解析：{type(error).__name__}")
    if "<tool_call>" in text and not calls and not problems:
        problems.append("有 <tool_call> 開頭但沒有結尾")
    content = CALL.sub("", text).strip()
    return content, calls, problems


def serialize(result):
    """MCP 結果 → 回給模型的文字。錯誤也要回去，並標明是錯誤；太長就截斷並說明。"""
    parts = []
    for c in result.content:
        if getattr(c, "text", None) is None:
            continue
        try:                                           # JSON 壓成一行，省 token 也方便讀 trace
            parts.append(json.dumps(json.loads(c.text), ensure_ascii=False, separators=(",", ":")))
        except json.JSONDecodeError:                   # 錯誤訊息：去掉給人看的說明網址，模型用不到
            parts.append("\n".join(line for line in c.text.splitlines()
                                   if not line.strip().startswith("For further information")))
    text = "\n".join(parts)
    payload = {"is_error": bool(result.is_error), "content": text}
    if len(text) > MAX_RESULT_CHARS:
        payload["content"] = text[:MAX_RESULT_CHARS]
        payload["truncated"] = f"原長 {len(text)} 字，只回前 {MAX_RESULT_CHARS} 字"
    return json.dumps(payload, ensure_ascii=False)


def prompt_key(messages, tools):
    blob = json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 模型（可替換）

class Replay:
    """重播 fixtures：以「完整提示的雜湊」查表。host 改了任何一則訊息，雜湊就對不上，直接報錯。"""
    label = "錄製"

    def __init__(self, path=FIXTURES):
        self.data = json.loads(path.read_text(encoding="utf-8"))

    def __call__(self, messages, tools):
        key = prompt_key(messages, tools)
        if key not in self.data["replies"]:
            raise LookupError("fixtures 沒有這個提示（host 送給模型的訊息變了）；請在有模型的機器用 --live 重錄")
        return self.data["replies"][key]


class LocalQwen:
    """本機 CPU 推論，greedy（不抽樣），所以同一台機器重跑結果相同。"""
    label = "即時"

    def __init__(self):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.record = {}

    def __call__(self, messages, tools):
        inputs = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                                    return_tensors="pt", return_dict=True)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=120, do_sample=False)
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        self.record[prompt_key(messages, tools)] = text
        return text


class Scripted:
    """教師手寫的回覆，用來測 host 的分支（模型不太會自己做出這些事）。"""
    label = "構造"

    def __init__(self, replies, repeat_last=False):
        self.replies, self.repeat_last = list(replies), repeat_last

    def __call__(self, messages, tools):
        return self.replies.pop(0) if len(self.replies) > 1 or not self.repeat_last else self.replies[0]


# ---------------------------------------------------------------- host：一個完整回合

NUMERAL = dict(zip("一二兩三四五六七八九十", (1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10)))


def check_answer(answer, results):
    """host 對最終答覆做的便宜檢查。模型拿到錯誤或截斷的結果，仍可能寫出很肯定的答案。"""
    flags = []
    ok = [json.loads(r) for r in results if not json.loads(r)["is_error"]]
    # 0.5B 模型常回簡體字，所以繁簡都要比對
    admits = re.search(r"[無无]法|查[無无]|失[敗败]|[沒没]有.{0,6}[資资]料|不支援|只(提供|接受)", answer)
    if results and not ok and not admits:
        flags.append("所有工具呼叫都失敗，答覆卻沒有說明查詢失敗")
    counts = {str(json.loads(r["content"]).get("count")) for r in ok if r["content"].startswith("{") and "truncated" not in r}
    for raw in re.findall(r"(\d+|[一二兩三四五六七八九十])\s*次", answer):
        value = str(NUMERAL.get(raw, raw))
        if value not in counts:
            flags.append(f"答覆說「{raw} 次」，工具結果裡沒有這個數字")
    if any("truncated" in r for r in ok) and not re.search(r"部分|截斷|前\s*\d+|不完整", answer):
        flags.append("工具結果被截斷，答覆沒有說明只看到一部分")
    return flags


def root_cause(error):
    """在 async with Client(...) 裡丟出的例外，會被 anyio 包成 ExceptionGroup；取出最裡面那個。"""
    while getattr(error, "exceptions", None):
        error = error.exceptions[0]
    return error


async def run_turn(client, model, question, trace):
    tools = to_model_tools((await client.list_tools()).tools)
    exposed = {t["function"]["name"] for t in tools}
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    trace.append(("user", question))
    results = []                                        # 本回合所有成功送到 MCP 的結果
    for round_no in range(1, MAX_ROUNDS + 1):
        text = model(messages, tools)
        content, calls, problems = parse_reply(text, round_no)
        if problems:
            trace.append(("host", "停止：模型輸出無法解析（" + "；".join(problems) + "）"))
            return "無法解析"
        if not calls:                                   # 無工具呼叫：這就是最終答覆
            messages.append({"role": "assistant", "content": content})
            trace.append((f"model[{model.label}]", content))
            flags = check_answer(content, results)
            for flag in flags:
                trace.append(("host", f"答覆檢查：{flag}"))
            return "完成，但答覆未通過檢查（不可直接給使用者）" if flags else "完成"
        messages.append({"role": "assistant", "content": content, "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
        for call in calls:
            args = json.dumps(call["arguments"], ensure_ascii=False)
            trace.append((f"model[{model.label}]", f"要求 {call['name']}({args})  → host 配發 {call['id']}"))
            if call["name"] not in exposed:             # 沒給模型看的工具，server 上有也不能執行
                body = json.dumps({"is_error": True, "content": f"工具 {call['name']} 不在本回合允許清單"}, ensure_ascii=False)
                trace.append(("host", f"拒絕：{call['name']} 不在允許清單（未送到 MCP）"))
            else:
                body = serialize(await client.call_tool(call["name"], call["arguments"]))
                results.append(body)
                trace.append(("mcp", f"{call['id']} → {body}"))
            messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": body})
    trace.append(("host", f"停止：已達 {MAX_ROUNDS} 輪工具上限，沒有最終答覆"))
    return "達到上限"


def ids_paired(trace):
    asked = re.findall(r"host 配發 (call_\d+_\d+)", "\n".join(t for _, t in trace))
    answered = re.findall(r"^(call_\d+_\d+) →", "\n".join(t for _, t in trace), re.M)
    rejected = sum(t.startswith("拒絕：") for _, t in trace)
    return len(asked) == len(answered) + rejected


RECORDED = (
    ("1 完整回合", "A 工地有幾次未戴安全帽？"),
    ("2 換一類事件", "B 工地有幾次逃生通道堵塞？"),
    ("3 無效參數：錯誤回給模型", "C 工地有幾次未戴安全帽？"),
    ("4 結果太長：截斷後回給模型", "列出 A 工地的所有事件。"),
    ("5 不需要工具", "你好，請用一句話介紹你能做什麼。"),
)
CONSTRUCTED = (
    ("6 要求沒開放的工具", "幫我刪掉 E02。",
     Scripted(['<tool_call>\n{"name": "delete_event", "arguments": {"id": "E02"}}\n</tool_call>',
               "無法刪除：本系統只提供查詢。"])),
    ("7 模型一直要工具", "A 工地狀況如何？",
     Scripted(['<tool_call>\n{"name": "count_events", "arguments": {"site": "A", "type": "no_helmet"}}\n</tool_call>'],
              repeat_last=True)),
    ("8 呼叫格式壞掉", "A 工地狀況如何？",
     Scripted(['<tool_call>\n{"name": "count_events", "arguments": {"site": "A"\n</tool_call>'])),
)


async def demo(model):
    print(f"模型：{MODEL_ID}｜來源：{'示範機錄製的 fixtures（重播）' if model.label == '錄製' else '本機即時推論'}")
    print(f"開放給模型的工具：{list(EXPOSED)}｜工具輪數上限 {MAX_ROUNDS}｜結果上限 {MAX_RESULT_CHARS} 字")
    for title, question, *scripted in RECORDED + CONSTRUCTED:
        events = [dict(e) for e in EVENTS]
        trace = []
        try:
            async with Client(build_server(events)) as client:
                status = await run_turn(client, scripted[0] if scripted else model, question, trace)
        except Exception as error:
            raise root_cause(error) from None
        print(f"\n=== {title} ===")
        for who, text in trace:
            print(f"  {who:<9} " + text.replace("\n", "\n" + " " * 12))
        print(f"  狀態：{status}｜call ID 配對：{'一致' if ids_paired(trace) else '不一致'}｜事件數仍為 {len(events)}")


def main(argv=()):
    import logging
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    model = LocalQwen() if "--live" in argv else Replay()
    try:
        asyncio.run(demo(model))
    except LookupError as error:
        if __name__ != "__main__":
            raise
        sys.exit(f"LookupError: {error}")
    if "--live" in argv and FIXTURES.exists():
        same = all(Replay().data["replies"].get(k) == v for k, v in model.record.items())
        print(f"\n即時輸出與 fixtures 逐字相同：{same}（{len(model.record)} 次模型呼叫）")
    return model


if __name__ == "__main__":
    main(sys.argv[1:])
