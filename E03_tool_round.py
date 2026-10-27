"""第 16 節：原生 Function Calling 的完整工具回合（只用 Python 標準函式庫）。

一個成功完整回合有四步，少一步都不算；協定重播若達回合上限，工具結果不會再送模型，不會有最終答覆：
  1. 模型提出 tool call（名稱＋參數＋call ID）
  2. **應用程式**決定要不要執行：allowlist、參數驗證、權限
  3. 執行，把結果連同**同一個 call ID** 回給模型
  4. 模型根據結果寫出最終答覆

預設路徑的模型由 `ScriptedModel` 扮演：它照劇本回傳訊息，訊息格式仿照
常見的原生 tool calling 協定。**這是協定重播（fixtures），不是即時模型。** 第三節另有明確選做的 NVIDIA Build live 路徑，只傳送這裡的合成問題；live 路徑需網路與 `NVIDIA_API_KEY`，但不會把模型輸出的工具呼叫直接執行。
"""

import json

# 只有 --nvidia-live 路徑才會使用；預設重播不載入任何網路依賴。
import os
import sys
import urllib.error
import urllib.request

# 合成事件資料，唯讀。
EVENTS = (
    {"id": "E1", "site": "A", "date": "2026-10-01", "type": "blocked_exit"},
    {"id": "E2", "site": "A", "date": "2026-10-03", "type": "no_helmet"},
    {"id": "E3", "site": "B", "date": "2026-10-03", "type": "no_helmet"},
    {"id": "E4", "site": "A", "date": "2026-10-04", "type": "no_helmet"},
)

TOOLS = {
    "count_events": {
        "description": "計算某工地在某期間的事件數（唯讀）",
        "parameters": {
            "type": "object",
            "required": ["site", "type"],
            "properties": {
                "site": {"type": "string", "enum": ["A", "B"]},
                "type": {"type": "string", "enum": ["blocked_exit", "no_helmet"]},
                "since": {"type": "string"},
            },
        },
    },
}

MAX_TOOL_ROUNDS = 3


def count_events(site, type, since="0000-00-00"):
    return {"site": site, "type": type, "since": since,
            "count": sum(1 for e in EVENTS if e["site"] == site and e["type"] == type
                         and e["date"] >= since)}


EXECUTORS = {"count_events": count_events}


def validate_args(name, args):
    """Validate the JSON value's object shape and the tool's declared schema."""
    if name not in TOOLS:
        return [f"工具 {name!r} 不在 allowlist"]
    if not isinstance(args, dict):
        return ["arguments 必須是 JSON object（JSON 物件）"]
    spec = TOOLS[name]["parameters"]
    errors = [f"缺少必填參數 {k}" for k in spec["required"] if k not in args]
    for key, value in args.items():
        rule = spec["properties"].get(key)
        if rule is None:
            errors.append(f"未定義的參數 {key}")
        elif not isinstance(value, str):
            errors.append(f"{key} 必須是字串")
        elif "enum" in rule and value not in rule["enum"]:
            errors.append(f"{key}={value!r} 不在允許清單 {rule['enum']}")
    return errors


def execute(call):
    """應用程式這一側。回傳要送回模型的 tool 訊息——**一定帶同一個 call ID**。"""
    name = call["function"]["name"]
    if name not in TOOLS:
        content = {"error": f"工具 {name} 不在 allowlist"}
    else:
        try:
            args = json.loads(call["function"]["arguments"])
        except (json.JSONDecodeError, TypeError) as error:
            detail = error.msg if isinstance(error, json.JSONDecodeError) else str(error)
            content = {"error": f"參數不是合法 JSON：{detail}"}
        else:
            problems = validate_args(name, args)
            if problems:
                content = {"error": "；".join(problems)}
            else:
                try:
                    content = EXECUTORS[name](**args)
                except (TypeError, ValueError) as error:
                    # Keep malformed arguments inside the tool protocol instead of
                    # crashing the host application's request loop.
                    content = {"error": f"工具參數或值無效：{error}"}
    return {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(content, ensure_ascii=False)}


class ScriptedModel:
    """照劇本出牌的「模型」。每次被呼叫就回傳劇本的下一則 assistant 訊息。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        if not self.script:
            return {"role": "assistant", "content": None,
                    "tool_calls": [{"id": f"call_loop_{self.calls}", "type": "function",
                                    "function": {"name": "count_events",
                                                 "arguments": '{"site":"A","type":"no_helmet"}'}}]}
        return self.script.pop(0)


def tool_call(call_id, name, arguments):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": arguments}}]}


def final(text):
    return {"role": "assistant", "content": text}


def run_turn(model, question):
    messages = [{"role": "user", "content": question}]
    for _ in range(MAX_TOOL_ROUNDS):
        reply = model(messages)
        if not isinstance(reply, dict) or reply.get("role") != "assistant":
            return messages, "協定錯誤：模型回覆格式無效，停止執行"
        calls = reply.get("tool_calls") or []
        if not isinstance(calls, list):
            return messages, "協定錯誤：tool_calls 必須是清單，停止執行"
        if not calls:
            messages.append(reply)
            return messages, "完成"
        validated = []
        seen_ids = set()
        for call in calls:
            if (not isinstance(call, dict) or not isinstance(call.get("id"), str)
                    or not call["id"] or call["id"] in seen_ids
                    or not isinstance(call.get("function"), dict)
                    or not isinstance(call["function"].get("name"), str)
                    or not isinstance(call["function"].get("arguments"), str)):
                return messages, "協定錯誤：tool call 結構或 ID 無效，停止執行"
            seen_ids.add(call["id"])
            validated.append(call)
        messages.append(reply)
        messages.extend(execute(call) for call in validated)
    return messages, f"達到回合上限 {MAX_TOOL_ROUNDS}，強制停止；最後工具結果未再送回模型，尚無最終答覆"


def nvidia_model(messages):
    """以 NVIDIA Build OpenAI 相容 Chat Completions API 呼叫工具。

    僅送合成資料；模型只提出呼叫，回傳訊息由本機 host 驗證後才執行。
    """
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("選做 live 需在本機設定 NVIDIA_API_KEY；不可寫入程式或命令列")
    request_body = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": messages,
        "tools": [{"type": "function", "function": {"name": name,
                   "description": spec["description"], "parameters": spec["parameters"]}}
                  for name, spec in TOOLS.items()],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 160,
    }
    req = urllib.request.Request(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        data=json.dumps(request_body).encode("utf-8"), method="POST",
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"NVIDIA API 回傳 HTTP {exc.code}；勿貼出憑證或完整請求") from None
    except urllib.error.URLError:
        raise RuntimeError("NVIDIA API 連線失敗或逾時；不自動重試，以免重複請求") from None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("API 回應缺少 choices；未執行任何工具")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("API 回應缺少 assistant message；未執行任何工具")
    # 不原樣執行：將供應商訊息轉成本教材狹窄協定形狀；再由 run_turn/execute 驗證。
    normalized = {"role": "assistant", "content": message.get("content")}
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise RuntimeError("tool_calls 型別錯誤；未執行任何工具")
    normalized_calls = []
    for index, call in enumerate(calls, 1):
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise RuntimeError("tool call 結構錯誤；未執行任何工具")
        function = call["function"]
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, str):
            raise RuntimeError("tool name/arguments 型別不符；未執行任何工具")
        normalized_calls.append({"id": call.get("id") or f"nvidia_call_{index}",
                                 "type": "function", "function": {"name": name, "arguments": arguments}})
    if normalized_calls:
        normalized["tool_calls"] = normalized_calls
    return normalized


def call_ids_match(messages):
    asked = [c["id"] for m in messages if m["role"] == "assistant" for c in m.get("tool_calls") or []]
    answered = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    return asked == answered and len(set(asked)) == len(asked)


def show(title, messages, status):
    print(f"=== {title} ===")
    for message in messages:
        if message["role"] == "user":
            print(f"  user      : {message['content']}")
        elif message["role"] == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                print(f"  assistant : 要求呼叫 {call['function']['name']}"
                      f"({call['function']['arguments']})  id={call['id']}")
        elif message["role"] == "tool":
            print(f"  tool      : id={message['tool_call_id']} → {message['content']}")
        else:
            print(f"  assistant : {message['content']}")
    print(f"  狀態：{status}｜訊息 {len(messages)} 則｜call ID 配對：{'一致' if call_ids_match(messages) else '不一致'}")
    print()


SCENARIOS = (
    ("1 完整回合", "A 工地 10/04 起有幾次未戴安全帽？", [
        tool_call("call_1", "count_events", '{"site":"A","type":"no_helmet","since":"2026-10-04"}'),
        final("A 工地自 2026-10-04 起有 1 次未戴安全帽紀錄（工具查詢結果）。"),
    ]),
    ("2 無效參數 → 錯誤回給模型 → 模型據實回報", "C 工地有幾次未戴安全帽？", [
        tool_call("call_1", "count_events", '{"site":"C","type":"no_helmet"}'),
        final("查詢失敗：工具只接受 A 或 B 工地，資料中沒有 C 工地。"),
    ]),
    ("3 要求不在 allowlist 的工具", "幫我刪掉 E2。", [
        tool_call("call_1", "delete_event", '{"id":"E2"}'),
        final("無法執行：刪除不在允許的工具清單內，本系統只提供唯讀查詢。"),
    ]),
    ("4 模型一直要求工具（不會自己停）", "A 工地狀況如何？", []),
)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--nvidia-live" in argv:
        if os.environ.get("NVIDIA_API_KEY") is None:
            raise SystemExit("選做 live 需先在本機環境設定 NVIDIA_API_KEY；不可把 key 貼進對話、筆記或命令列")
        question = "A 工地 10/04 起有幾次未戴安全帽？只可用唯讀工具查詢；若資料不足請說明。"
        messages, status = run_turn(nvidia_model, question)
        show("NVIDIA Build live（僅合成資料）", messages, status)
        return
    if argv:
        raise SystemExit("用法：python3 E03_tool_round.py [--nvidia-live]")
    print(f"工具 allowlist：{sorted(TOOLS)}｜回合上限：{MAX_TOOL_ROUNDS}｜事件 {len(EVENTS)} 筆（合成、唯讀）")
    print()
    for title, question, script in SCENARIOS:
        messages, status = run_turn(ScriptedModel(script), question)
        show(title, messages, status)

    print("=== 只印出函式名稱，為什麼不夠？ ===")
    messages, _ = run_turn(ScriptedModel(SCENARIOS[0][2]), SCENARIOS[0][1])
    truth = count_events("A", "no_helmet", "2026-10-04")["count"]
    total = count_events("A", "no_helmet")["count"]
    answer = messages[-1]["content"]
    print(f"  工具實際回傳 count={truth}；最終答覆：{answer}")
    print(f"  答覆中的數字與工具結果一致：{str(truth) + ' 次' in answer}"
          f"（若忽略 since 會得到 {total}，答案就錯了）")
    print("  只看到「呼叫了 count_events」證明不了這件事——要看到結果有被送回、也被正確使用。")


if __name__ == "__main__":
    main()