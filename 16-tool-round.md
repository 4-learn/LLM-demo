# 原生 Function Calling 完整工具回合

> **新版正文：20E07-E03-05｜對應新版 LLM 第 16 節：原生 Function Calling 完整工具回合。**
> 不用任何 orchestration 框架，親手跑完一個工具回合：**模型提出 → 程式決定 → 執行 → 結果回模型 → 最終答覆**。核心主張：**模型不執行函式，應用程式才執行。** 必做路徑只用 Python 標準函式庫與協定重播（fixtures）；另提供選做 NVIDIA Build live 對照，實際工具執行仍由本機程式控制。

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先修第 12 節的權限邊界與第 15 節的 schema 驗證。

| 時間 | 內容 |
| --- | --- |
| 0–8 分鐘 | 「模型呼叫工具」這句話哪裡錯了 |
| 8–20 分鐘 | 完整回合的四步、tool call ID、訊息順序 |
| 20–28 分鐘 | 應用程式的責任：allowlist、參數驗證、唯讀執行器、回合上限 |
| 28–35 分鐘 | 跑 Demo：一個成功、三個失敗情境 |
| 35–42 分鐘 | 選做：以 NVIDIA Build 觀察真實工具提議，沿用本機 host 驗證與唯讀執行器 |
| 42–50 分鐘 | Workshop：自己加一個工具並跑無效參數 |

驗收能力：交付一份**含模型 tool call、程式結果、相符 call ID 與最終答覆**的完整紀錄——**不只印出函式名稱**。必做 fixture 只驗證應用程式協定流程；ScriptedModel 預先寫定每步訊息，不是模型依工具結果生成的閉環。

## 一、「模型呼叫工具」這句話哪裡錯了

模型**不會**呼叫任何東西。它輸出的是一段文字（或結構化資料），內容是「我想呼叫 `count_events`，參數是 `{"site":"A"}`」。**真正執行的是你的程式**，而你的程式可以拒絕。

| 誰 | 做什麼 | 能不能信 |
| --- | --- | --- |
| 模型 | 提出要呼叫哪個工具、帶什麼參數 | **不能**——它的輸出是第 15 節的結構化輸出，要驗證 |
| 應用程式 | 決定要不要執行、驗證參數、檢查權限 | 這是你寫的，**你負責** |
| 工具 | 真的去查、去改 | 依權限而定（第 12 節） |

這跟第 01 節的責任表是同一件事：**模型提議，程式決定。**

## 二、完整回合的四步

```text
1. user        → 問題
2. assistant   → tool_calls: [{id: "call_1", name: "count_events", arguments: "{...}"}]
3. tool        → {tool_call_id: "call_1", content: "{...結果...}"}
4. assistant   → 最終答覆（根據第 3 步的結果）
```

每一步都有可以出錯的地方：

| 步驟 | 常見錯誤 | 後果 |
| --- | --- | --- |
| 2 | 參數是字串形式的 JSON；解碼後也可能是 array、null 或 scalar，而非 object | 直接 `**args` 會炸；本 Demo 將非 object 回成工具錯誤、不執行工具 |
| 2→3 | 沒驗證就執行 | 模型給什麼就跑什麼 |
| 3 | **`tool_call_id` 沒對上** | 一次有多個呼叫時，結果對錯了呼叫；很多 API 會直接拒收 |
| 3 | 工具出錯時直接拋例外 | 模型永遠不知道發生了什麼，你也沒有紀錄 |
| 4 | 沒有回合上限 | 模型一直要求工具，程式一直跑 |

> **`arguments` 是一段字串，不是物件。** 先 JSON decode，再拒絕非 JSON object，最後驗證 required／型別／enum／未知欄位；本 demo 不做欄位間語義推理或授權政策驗證，不應稱為三層完整驗證。

## 三、應用程式的四個責任

| 責任 | 本節 Demo 怎麼做 | 為什麼 |
| --- | --- | --- |
| **allowlist** | `TOOLS` 字典裡沒有的名稱一律拒絕 | 模型可能要求一個你根本沒提供的工具（情境 3） |
| **參數驗證** | JSON decode 後要求 object，`validate_args` 檢查必填、型別、允許值、未定義參數 | 模型給的參數就是結構化輸出，會錯（情境 2） |
| **唯讀執行器** | `count_events` 只讀不寫 | 第 12 節：控制在工具層 |
| **回合上限** | `MAX_TOOL_ROUNDS = 3` | 模型不一定會自己停（情境 4） |

還有一個**錯誤回傳**的原則：**工具出錯時，把錯誤當成結果回給模型**，而不是拋例外中斷。這樣模型可以據實告訴使用者「查詢失敗」，而你的紀錄裡也完整保留了失敗原因。

## 四、協定重播：沒有模型也能學協定

本節的 `ScriptedModel` 是**協定重播 fixture**：它依預設劇本依序回傳訊息。訊息形狀仿照常見的原生 tool calling 協定（`role`／`tool_calls`／`tool_call_id`）。工具結果雖會放進後續 messages，但最終答覆也是預先寫在 fixture；這不是模型讀取工具結果再生成答案的模型閉環。

**這是協定重播（fixtures），不是即時模型，也不是模型閉環。** 它只驗證應用程式對固定訊息的處理（工具選擇、參數拒絕、call ID 與回合上限）；不證明任何真實模型會產生這些訊息，也不證明模型會根據工具結果形成最終答覆。

為什麼先用重播？因為**應用程式這一側的邏輯（驗證、allowlist、上限、ID 配對）跟模型無關**，而且是最常寫錯的部分。先把它寫對、寫測試，再接上真的模型。

## 五、可執行 Demo

情境：一個唯讀的合成事件查詢工具 `count_events`。四個情境：完整協定重播、無效參數、要求不在 allowlist 的工具、模型不會自己停。完整回合中的最終答覆為預錄 fixture，不是模型根據工具輸出現場生成。回合上限情境會在最後一組工具結果後停止，該結果尚未再送模型，故沒有最終答覆。

<!-- demo: E03_tool_round.py -->
```python
"""第 16 節：原生 Function Calling 的完整工具回合（只用 Python 標準函式庫）。

一個成功完整回合有四步，少一步都不算；協定重播若達回合上限，工具結果不會再送模型，不會有最終答覆：
  1. 模型提出 tool call（名稱＋參數＋call ID）
  2. **應用程式**決定要不要執行：allowlist、參數驗證、權限
  3. 執行，把結果連同**同一個 call ID** 回給模型
  4. 模型根據結果寫出最終答覆

預設路徑的模型由 `ScriptedModel` 扮演：它照劇本回傳訊息，訊息格式仿照
常見的原生 tool calling 協定。**這是協定重播（fixtures），不是即時模型。** 第三節另有明確選做的 NVIDIA Build live 路徑，只傳送這裡的合成問題；live 路徑需安裝 `openai` SDK、網路與 `NVIDIA_API_KEY`，但不會把模型輸出的工具呼叫直接執行。
"""

import json

# 只有 --nvidia-live 路徑才會使用；預設重播不載入任何網路依賴。
import os
import sys
# nvidia_model 內延遲載入 openai SDK，讓 fixture replay 不需安裝 SDK。

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
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("選做 live 需安裝 openai SDK（pip install 'openai>=1,<3'）；重播不需要") from None
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=45,
        max_retries=0,
    )
    try:
        response = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=messages,
            tools=[{"type": "function", "function": {"name": name,
                    "description": spec["description"], "parameters": spec["parameters"]}}
                   for name, spec in TOOLS.items()],
            tool_choice="auto",
            temperature=0,
            max_tokens=160,
        )
    except Exception as exc:
        raise RuntimeError(f"NVIDIA API 呼叫失敗（{type(exc).__name__}）；不自動重試") from None
    if not response.choices:
        raise RuntimeError("API 回應缺少 choices；未執行任何工具")
    message = response.choices[0].message
    if message is None:
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
```

執行指令（在 `llm-demo` 目錄下；預設離線路徑不需要網路、模型或 SDK）：

```bash
python3 E03_tool_round.py
python3 -m unittest -v test_E03_tool_round.py
```

選做 NVIDIA Build live（只用合成事件；需安裝 `openai` SDK、教師核准、網路與本機環境變數 `NVIDIA_API_KEY`）：

```bash
python3 E03_tool_round.py --nvidia-live
```

程式會請模型提出唯讀工具呼叫，再由同一個本機 `run_turn`／`execute` 路徑驗證和執行。缺少 key 時不發請求；不要將 key 放入命令列、程式、筆記或作業。離線 `ScriptedModel` 路徑是協定重播 fixture，不是模型閉環。

關鍵輸出（Linux／Python 3.10.15，20E07-E03-05；錯誤詳見實際輸出）：

```text
工具 allowlist：['count_events']｜回合上限：3｜事件 4 筆（合成、唯讀）

=== 1 完整回合 ===
  user      : A 工地 10/04 起有幾次未戴安全帽？
  assistant : 要求呼叫 count_events({"site":"A","type":"no_helmet","since":"2026-10-04"})  id=call_1
  tool      : id=call_1 → {"site": "A", "type": "no_helmet", "since": "2026-10-04", "count": 1}
  assistant : A 工地自 2026-10-04 起有 1 次未戴安全帽紀錄（工具查詢結果）。
  狀態：完成｜訊息 4 則｜call ID 配對：一致
```

## 六、結果怎麼讀

### 1. 完整回合是四則訊息，而且 ID 配對

情境 1 的四則訊息就是第二段的四步。注意 `tool` 訊息帶的 `id=call_1` 與 `assistant` 要求的 `id=call_1` **一致**——這是程式碼確實把結果對回那一次呼叫的證據。

### 2. 只印函式名稱不夠

最後一段把這件事算給你看：工具回傳 `count=1`，因為 `since` 參數把 10/01 與 10/03 的紀錄濾掉了。**如果你的程式忽略了 `since`，結果會是 `2`，最終答覆就錯了。**

只看到「模型呼叫了 `count_events`」的紀錄，**證明不了參數有被正確使用、結果有被送回、答覆有根據結果**。這就是課綱驗收寫「不只印出函式名稱」的原因。

### 3. 錯誤也是結果

情境 2 與 3 都沒有拋例外。錯誤被包成 `{"error": ...}`、帶著同一個 call ID 回給模型，模型據實回報。對照第 15 節：**拒絕與錯誤是合法結果，不是要隱藏的東西。**

情境 3 特別重要：模型要求了 `delete_event`——一個**根本不存在**的工具。如果你的程式是 `globals()[name](**args)` 這種寫法，它會去找一個叫這個名字的函式，**而你不知道你的 namespace 裡有什麼**。allowlist 就是把「能被呼叫的東西」限縮成一張你寫下來的清單。

### 4. 回合上限是唯一保證會停的東西

情境 4 的模型劇本是空的，所以它每一輪都要求同一個工具。**沒有上限，這個迴圈會一直跑下去**（真實環境裡則是一直花錢）。本節的上限是 3，程式在 7 則訊息後強制停止，並把狀態寫成「達到回合上限」而不是「完成」。

注意：每次呼叫的 ID 都不同（`call_loop_1`、`_2`、`_3`）。**ID 重複是協定錯誤**，`call_ids_match` 會把它當成不一致。

## 七、常見誤判

| 誤判 | 為什麼錯 |
| --- | --- |
| 「模型呼叫了工具」 | 模型只**提議**；執行的是你的程式，你可以也應該拒絕 |
| 「`arguments` 是物件，直接拿來用」 | 它是字串形式的 JSON，要 parse 也要驗證（第 15 節） |
| 「工具出錯就拋例外」 | 模型與紀錄都看不到失敗原因；應該把錯誤當結果回傳 |
| 「模型會自己停」 | 不一定；回合上限是唯一保證 |
| 「用 `getattr`／`globals()` 動態找函式比較彈性」 | 那等於沒有 allowlist |
| 「log 有印出呼叫了哪個函式，就算驗收」 | 要有參數、結果、配對的 ID 與最終答覆，才能證明整個回合是對的 |
| 「用框架就不用管這些」 | 框架幫你包了這些步驟，**不會幫你決定哪些工具可以被呼叫** |

## Workshop：加一個工具並跑無效參數

### 任務

1. 執行 Demo，先預測情境 4 會產生幾則訊息，再核對。
2. 新增一個唯讀工具 `list_sites`（無參數，回傳所有出現過的工地代號）。把它加進 `TOOLS` 與 `EXECUTORS`，並寫一個劇本：先呼叫 `list_sites`，再呼叫 `count_events`，最後給答覆。**call ID 要用 `call_1`、`call_2`。**
3. 寫一個劇本，讓模型在 `arguments` 裡放一段**不合法的 JSON**（例如少一個引號）。你的程式會怎麼回應？
4. 寫一個劇本，讓模型傳一個**未定義的參數**（例如 `"limit": "10"`）。它會被擋下嗎？如果你把 `validate_args` 裡檢查未定義參數的那段刪掉，會發生什麼事？
5. 把 `MAX_TOOL_ROUNDS` 改成 `1`，重跑情境 1。結果是什麼？**這個上限對正常的回合太嚴格了嗎？**

### 參考判讀

- 第 1 題：**7 則**——1 則 user，加上 3 輪 ×（1 則 assistant ＋ 1 則 tool）。最後一則是 tool 訊息，**沒有最終答覆**，所以狀態是「達到回合上限」而不是「完成」。
- 第 3 題：`execute` 會回傳 `{"error": "參數不是合法 JSON：..."}`，**帶著同一個 call ID**。程式不會崩潰，模型會收到錯誤。
- 第 4 題：會被擋下（`未定義的參數 limit`）。刪掉那段之後，`count_events(**args)` 會因為 `limit` 不是它的參數而拋出 `TypeError`——**這個例外沒有被 `execute` 接住**，整個回合會崩潰。這說明參數驗證不只是安全問題，也是穩定性問題。
- 第 5 題：模型提出呼叫、工具回傳結果，然後**迴圈就結束了**，模型沒有機會寫最終答覆，狀態是「達到回合上限 1」。一個需要工具的正常回合至少需要 **2** 次模型呼叫；上限要大於這個數字。

### 驗收

- 交出一份**完整回合紀錄**：user 問題、assistant 的 tool call（名稱、參數、ID）、tool 結果（帶相同 ID）、最終答覆。
- 至少一個**失敗情境**的紀錄（無效參數、不在 allowlist 或達到上限），並說明程式如何處理。
- 寫明證據等級：用協定重播的要標「fixtures，非即時模型」；用真實端點的要附模型名稱、版本與日期。
- 不得只印出「呼叫了某函式」作為完成證明。

## 教師紀錄與下一步

- 必做的 `ScriptedModel` 路徑**不需要模型、網路或任何 SDK**。`ScriptedModel` 的訊息形狀仿照常見原生 tool calling 協定的欄位（`tool_calls`／`tool_call_id`／`arguments` 為字串），但**不宣稱與任何特定供應商的 API 逐欄位相容**——那是第 17 節的契約測試要驗的。
- 證據等級：**fixtures（協定重播）**。它驗證的是應用程式這一側的邏輯，不是任何模型的行為。
- 撰寫時的實際紀錄：第一版情境 4 每輪都用同一個 `call_loop` 當 ID，而當時的 `call_ids_match` 只比對順序，所以**重複 ID 也被判為「一致」**。已改為每輪不同 ID，並讓配對檢查也拒絕重複。另外，第一版情境 1 的 `since=20E07-E03-02` 剛好跟不加 `since` 得到一樣的答案（都是 2），**證明不了參數有被使用**；改成 `2026-10-04` 之後兩者才有差別（1 對 2）。
- 選做 NVIDIA Build live：執行 `python3 E03_tool_round.py --nvidia-live`，使用合成問題與唯讀 `count_events` 工具；程式需在本機設定 `NVIDIA_API_KEY`。未設定金鑰時安全拒絕，不連網；本教材與測試不會代為呼叫 API。
- live 回應以 `messages` 交回同一個 `run_turn`，工具呼叫仍經 `execute` 的 allowlist、JSON parse、參數驗證與唯讀 executor；模型不得取得其他工具或直接執行程式。
- 指定 `meta/llama-3.1-8b-instruct` 與 OpenAI-compatible `tools` 欄位僅是示例契約，執行前須依當日 NVIDIA 模型文件確認；不支援時用離線 fixture，不能默認所有模型支援。
- 第 17 節可再比較不同供應商的 tool call 格式、arguments 型別與平行呼叫差異；本節 live 路徑只有示範 NVIDIA Build，不代表已驗證該端點目前可用。
