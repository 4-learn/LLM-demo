# Multi-Agent、固定 Workflow 與 Reducer

> **正文改版：20E07-E03-06｜對應新版 LLM 第 38 節：Multi-Agent、固定 Workflow 與 Reducer。**
> 舊教材的 supervisor 範例照抄執行，`completed_agents` 沒有 reducer，**跑不完**：每個代理都把清單蓋成只剩自己，supervisor 永遠看不到全部完成，一直來回派工直到遞迴上限。本節先把這個 bug 跑出來，再修；接著用固定 workflow 做同一件事，比較步數。核心主張：**任務的順序事先知道時，固定 workflow 比 supervisor 更短、更好測。多代理不是升級，是有代價的選擇。不管用哪一種，都要有停止條件，而且收尾要看 state 裡的結果，不能只看路由器說「done」。**
> 本節的代理都是固定字串，不呼叫模型；第五段「會亂派工的路由器」是教師構造的決定序列，用來模擬 LLM supervisor 可能的輸出。

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先修第 27 節（reducer）、第 28 節（分支）、第 32 節（整合）、第 35 節（工具回合與輪數上限）。

| 時間 | 內容 |
| --- | --- |
| 0–10 分鐘 | 固定分工、LLM 動態路由、supervisor：差在「誰決定下一步」 |
| 10–20 分鐘 | 舊教材的 bug：`completed_agents` 沒有 reducer |
| 20–30 分鐘 | 停止條件：遞迴上限、代理預算、未知代理、收尾檢查 |
| 30–35 分鐘 | 跑 Demo |
| 35–50 分鐘 | Workshop：加 reducer 與去重，再和固定 workflow 比較 |

驗收能力：交出**並行合併、重複輸入、停止條件**三種測試的結果，並列出「這個任務不需要多代理」的理由。

## 一、三種分工方式

| 方式 | 誰決定下一步 | 適合 | 代價 |
| --- | --- | --- | --- |
| 固定 workflow | 寫死在 graph 的邊 | 步驟與依賴事先知道 | 換流程要改程式 |
| supervisor（規則） | 一個節點看 state 決定 | 步驟會因 state 而變，但規則寫得出來 | 每派一次工多一步；要防無限迴圈 |
| supervisor（LLM） | 模型看 state 決定 | 步驟真的事先不知道 | 會派錯、漏派、亂派；要預算、要驗證 |

舊教材的情境「法規 → 報告 → 通知」**順序是固定的**：報告需要法規，通知不需要任何人。這種任務寫成固定 workflow 就好，法規和通知還可以並行。

## 二、舊教材的 bug

舊教材的 state：

```python
completed_agents: list[str]    # 已完成的 Agent 清單
```

每個代理回傳 `{"completed_agents": ["regulation"]}` 這種「只有自己」的清單。第 27 節講過：**沒有 reducer 的欄位，後寫的蓋掉先寫的**。所以：

1. 法規完成 → `["regulation"]`
2. supervisor 看到缺報告 → 派報告 → `["report"]`（法規被蓋掉了）
3. supervisor 看到缺法規 → 又派法規 → `["regulation"]`
4. ……一直重複，直到 LangGraph 的遞迴上限（預設 25 步）丟出 `GraphRecursionError`

舊教材的「執行結果」顯示 7 步就完成，那**不是這段程式實際跑出來的**。本節用 langgraph 1.2.12 實際執行，見 Demo 第一段。

修法有兩種：

| reducer | 效果 | 問題 |
| --- | --- | --- |
| `operator.add` | 串接，不會丟 | 同一個代理完成兩次會記兩次（第四段） |
| 去重合併（`union`） | 集合聯集後排序 | 結果順序是字母序，不是完成順序；要完成順序就另外記 `log` |

## 三、停止條件

supervisor 會回到自己，所以一定要有停止條件。本節示範四層，越前面越粗：

1. **遞迴上限**（LangGraph 內建，`recursion_limit`）：最後防線。到了就丟例外，中間的結果都沒了。
2. **代理預算**：supervisor 自己數已經派了幾次，超過就停，回 `stop`。比遞迴上限早停，而且保留已完成的結果。
3. **未知代理**：路由器說要派 `lawyer`，但沒有這個節點，就停下並記錄，不要猜。
4. **收尾檢查**：路由器說 `done`，不代表真的完成。最後要直接看 state，每個代理的結果欄位都在，才算完成。

第 4 層最常被省略，Demo 第五段最後一個情境就是例子：路由器跳過法規直接說 done，報告裡的法規依據是「尚未查詢」。

## 四、可執行 Demo

需要 `langgraph==1.2.12`，不需要模型。

<!-- demo: E07_multi_agent.py -->
```python
"""第 38 節：Multi-Agent、固定 Workflow 與 Reducer。

舊教材的 supervisor 照抄（只把呼叫 LLM 換成固定字串），先跑出它的 bug，再修；
接著把同一件事改成固定 workflow 比較步數，最後加上停止條件與重複輸入。
需要 langgraph==1.2.12；不需要模型、不需要 API key。
"""
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError, InvalidUpdateError
from langgraph.graph import END, START, StateGraph

EVENT = {"id": "EVT-20E07-0042", "category": "高處墜落", "severity": "高"}
AGENTS = ("regulation", "report", "notification")


# ---------------------------------------------------------------- 三種 completed_agents 合併規則

def union(current, update):
    """去重合併：同一個代理完成兩次只記一次；結果排序，與執行順序無關。"""
    return sorted(set(current or []) | set(update or []))


class Plain(TypedDict, total=False):              # 舊教材：沒有 reducer
    event: dict
    next_agent: str
    completed_agents: list
    regulation_result: str
    report_result: str
    notification_result: str
    log: Annotated[list, operator.add]


class Added(Plain, total=False):
    completed_agents: Annotated[list, operator.add]


class Union(Plain, total=False):
    completed_agents: Annotated[list, union]


# ---------------------------------------------------------------- 代理：舊教材的三個 agent，LLM 換成固定字串

def regulation(state):
    return {"regulation_result": f"{state['event']['category']}：營造安全衛生設施標準（示意）",
            "completed_agents": ["regulation"], "log": ["regulation"]}


def report(state):
    basis = state.get("regulation_result", "尚未查詢")
    return {"report_result": f"報告 {state['event']['id']}；法規依據：{basis}",
            "completed_agents": ["report"], "log": ["report"]}


def notification(state):
    targets = {"高": 3, "中": 2}.get(state["event"]["severity"], 1)
    return {"notification_result": f"已通知 {targets} 人",
            "completed_agents": ["notification"], "log": ["notification"]}


NODES = {"regulation": regulation, "report": report, "notification": notification}


# ---------------------------------------------------------------- 模式一：supervisor（舊教材的決策邏輯）

def supervisor_decide(state):
    done = state.get("completed_agents") or []
    nxt = next((a for a in AGENTS if a not in done), "done")
    return {"next_agent": nxt, "log": [f"S→{nxt}"]}


def build_supervisor(schema, decide=supervisor_decide, checkpointer=None):
    graph = StateGraph(schema)
    graph.add_node("supervisor", decide)
    for name, fn in NODES.items():
        graph.add_node(name, fn)
        graph.add_edge(name, "supervisor")
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", lambda s: s["next_agent"],
                                {**{a: a for a in AGENTS}, "done": END, "stop": END})
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------- 模式二：固定 workflow

def build_fixed(schema, checkpointer=None):
    """法規與通知互不依賴 → 並行；報告需要法規 → 接在法規後面。"""
    graph = StateGraph(schema)
    for name, fn in NODES.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "regulation")
    graph.add_edge(START, "notification")
    graph.add_edge("regulation", "report")
    graph.add_edge("report", END)
    graph.add_edge("notification", END)
    return graph.compile(checkpointer=checkpointer)


def start():
    return {"event": dict(EVENT), "completed_agents": [], "log": []}


def run(app, limit=25):
    """回傳 (最終 state 或錯誤字串, 超步數, 節點執行次數)。同一超步內的節點是並行的。"""
    supersteps = nodes = 0
    final = None
    try:
        for mode, chunk in app.stream(start(), {"recursion_limit": limit}, stream_mode=["updates", "values"]):
            if mode == "updates":                    # 每個節點的更新各一筆
                nodes += len(chunk)
            else:                                    # 每個超步結束時一筆（另有一筆初始 state）
                supersteps += 1
                final = chunk
        return final, supersteps - 1, nodes
    except GraphRecursionError:
        return f"GraphRecursionError（{limit} 步內沒結束）", supersteps, nodes
    except InvalidUpdateError as error:
        return "InvalidUpdateError：" + str(error).splitlines()[0][:52], supersteps, nodes


def missing(final):
    """收尾檢查：不看路由器說了什麼，直接看 state 裡有沒有每個代理的結果。"""
    return [a for a in AGENTS if f"{a}_result" not in final]


def describe(final):
    if isinstance(final, str):
        return final
    return f"completed={final['completed_agents']} 缺結果={missing(final) or '無'}"


# ---------------------------------------------------------------- 模式三：會亂派工的路由（模擬 LLM supervisor）

def make_router(script, budget=None):
    """script：路由器依序給出的決定（教師構造，代表 LLM 可能的輸出）；budget：代理呼叫上限。"""
    picks = iter(script)

    def decide(state):
        calls = sum(1 for entry in state.get("log", []) if not entry.startswith("S→"))
        if budget is not None and calls >= budget:
            return {"next_agent": "stop", "log": [f"S→stop(預算 {budget} 用完)"]}
        nxt = next(picks, "done")
        if nxt not in AGENTS and nxt != "done":
            return {"next_agent": "stop", "log": [f"S→stop(未知代理 {nxt})"]}
        return {"next_agent": nxt, "log": [f"S→{nxt}"]}

    return decide


def main():
    import importlib.metadata as metadata
    print("langgraph:", metadata.version("langgraph"))
    print("事件:", EVENT)

    print("\n== 一、舊教材的 supervisor：completed_agents 沒有 reducer ==")
    final, _, _ = run(build_supervisor(Plain), limit=12)
    print("  結果:", describe(final))
    app = build_supervisor(Plain)
    trail = []
    try:
        for update in app.stream(start(), {"recursion_limit": 12}, stream_mode="updates"):
            (node, data), = update.items()
            trail.append(f"{node}:{data.get('completed_agents', data.get('next_agent'))}")
    except GraphRecursionError:
        pass
    print("  前 6 步:", " → ".join(trail[:6]))

    print("\n== 二、加上 reducer ==")
    for schema in (Added, Union):
        final, supersteps, nodes = run(build_supervisor(schema))
        print(f"  {schema.__name__:<6} {describe(final)}")
        print(f"         超步 {supersteps}、節點執行 {nodes} 次  log={final['log']}")

    print("\n== 三、改成固定 workflow（法規 ∥ 通知 → 報告） ==")
    for schema in (Plain, Added, Union):
        final, supersteps, nodes = run(build_fixed(schema))
        print(f"  {schema.__name__:<6} {describe(final)}")
        if not isinstance(final, str):
            print(f"         超步 {supersteps}、節點執行 {nodes} 次  log={final['log']}")

    print("\n== 四、重複輸入：同一事件在同一 thread 送兩次 ==")
    for schema in (Added, Union):
        app = build_fixed(schema, InMemorySaver())
        config = {"configurable": {"thread_id": EVENT["id"]}}
        app.invoke(start(), config)
        final = app.invoke({"event": dict(EVENT), "log": []}, config)
        print(f"  {schema.__name__:<6} 第 2 次後 completed={final['completed_agents']}  log 長度 {len(final['log'])}")

    print("\n== 五、停止條件：路由器亂派工（構造的決定序列） ==")
    loops = ["regulation"] * 30
    cases = (
        ("一直派同一個，無預算", make_router(loops), 25),
        ("一直派同一個，預算 5", make_router(loops, budget=5), 25),
        ("派到不存在的代理", make_router(["regulation", "lawyer"]), 25),
        ("漏派法規就說 done", make_router(["notification", "report", "done"]), 25),
    )
    for label, decide, limit in cases:
        final, _, nodes = run(build_supervisor(Union, decide), limit=limit)
        print(f"  {label}（節點執行 {nodes} 次）")
        if isinstance(final, str):
            print(f"    {final}")
            continue
        verdict = "通過" if not missing(final) else "不通過，不能當成完成"
        print(f"    最後決定 {final['log'][-1]}；收尾檢查：缺 {missing(final) or '無'} → {verdict}")
        if "report_result" in final:
            print(f"    報告內容：{final['report_result']}")


if __name__ == "__main__":
    main()
```

執行（程式也放在 [llm-demo](https://github.com/4-learn/llm-demo)）：

```bash
python E07_multi_agent.py
```

關鍵輸出（示範機 4-learn：Ubuntu 24.04／Python 3.12.3／langgraph 1.2.12，20E07-E03-06；備課機 Python 3.10.15 逐字相同）：

```text
langgraph: 1.2.12
事件: {'id': 'EVT-20E07-0042', 'category': '高處墜落', 'severity': '高'}

== 一、舊教材的 supervisor：completed_agents 沒有 reducer ==
  結果: GraphRecursionError（12 步內沒結束）
  前 6 步: supervisor:regulation → regulation:['regulation'] → supervisor:report → report:['report'] → supervisor:regulation → regulation:['regulation']

== 二、加上 reducer ==
  Added  completed=['regulation', 'report', 'notification'] 缺結果=無
         超步 7、節點執行 7 次  log=['S→regulation', 'regulation', 'S→report', 'report', 'S→notification', 'notification', 'S→done']
  Union  completed=['notification', 'regulation', 'report'] 缺結果=無
         超步 7、節點執行 7 次  log=['S→regulation', 'regulation', 'S→report', 'report', 'S→notification', 'notification', 'S→done']

== 三、改成固定 workflow（法規 ∥ 通知 → 報告） ==
  Plain  InvalidUpdateError：At key 'completed_agents': Can receive only one valu
  Added  completed=['notification', 'regulation', 'report'] 缺結果=無
         超步 2、節點執行 3 次  log=['notification', 'regulation', 'report']
  Union  completed=['notification', 'regulation', 'report'] 缺結果=無
         超步 2、節點執行 3 次  log=['notification', 'regulation', 'report']

== 四、重複輸入：同一事件在同一 thread 送兩次 ==
  Added  第 2 次後 completed=['notification', 'regulation', 'report', 'notification', 'regulation', 'report']  log 長度 6
  Union  第 2 次後 completed=['notification', 'regulation', 'report']  log 長度 6

== 五、停止條件：路由器亂派工（構造的決定序列） ==
  一直派同一個，無預算（節點執行 25 次）
    GraphRecursionError（25 步內沒結束）
  一直派同一個，預算 5（節點執行 11 次）
    最後決定 S→stop(預算 5 用完)；收尾檢查：缺 ['report', 'notification'] → 不通過，不能當成完成
  派到不存在的代理（節點執行 3 次）
    最後決定 S→stop(未知代理 lawyer)；收尾檢查：缺 ['report', 'notification'] → 不通過，不能當成完成
  漏派法規就說 done（節點執行 5 次）
    最後決定 S→done；收尾檢查：缺 ['regulation'] → 不通過，不能當成完成
    報告內容：報告 EVT-20E07-0042；法規依據：尚未查詢
```

## 五、結果怎麼讀

### 1. 舊教材的 supervisor 跑不完

第一段：前 6 步就看得到 `regulation → report → regulation`，`completed_agents` 每次都只剩一個。12 步上限到了就丟 `GraphRecursionError`。

### 2. 兩種 reducer 都能跑完

第二段：7 步完成（4 次 supervisor、3 次代理）。`Added` 的 `completed` 是完成順序，`Union` 是字母序，要看完成順序用 `log`。

### 3. 固定 workflow 更短

第三段：同樣三個代理，**2 個超步、3 次節點執行**，supervisor 版要 7 個超步、7 次。法規與通知在第 1 個超步並行，報告在第 2 個。

沒有 reducer 的 `Plain` 在這裡直接 `InvalidUpdateError`：兩個節點在同一個超步寫同一個欄位，LangGraph 不知道要留哪一個。循序執行時默默蓋掉，並行時才報錯（同第 27 節）。

### 4. 重複輸入

第四段：同一事件在同一個 thread 送兩次（例如通報系統重送）。`Added` 的 `completed` 變成 6 筆，`Union` 仍是 3 筆。兩者的 `log` 都是 6 筆：代理確實執行了兩次。**reducer 去重只讓「記錄」不重複，不會讓「動作」不重複。** 通知真的發了兩次；要擋住重複的動作，要靠第 29 節的冪等鍵。

### 5. 停止條件

第五段：

- 沒有預算時，一直派法規，到遞迴上限才停，結果全丟。
- 預算 5：派了 5 次就停（另有 6 次 supervisor 決策，共 11 次），保留了法規結果，收尾檢查「不通過」。
- 未知代理：停下並記錄 `lawyer`。
- 漏派法規：路由器說 done，**收尾檢查抓到缺法規**，報告內容也證實它沒有依據。

## 六、什麼時候不需要多代理

本節的情境就是例子。下面幾種情況，先寫成固定 workflow：

- 步驟和依賴關係事先知道（本節：法規 → 報告；通知獨立）。
- 每一步的輸入只來自前一步，不需要「討論」。
- 需要可重現的測試：固定 workflow 的路徑是固定的，supervisor（尤其 LLM）的路徑每次可能不同。
- 預算有限：每多一次 supervisor 決策，如果是 LLM，就是多一次模型呼叫。

需要 supervisor 的訊號：下一步取決於前一步的**內容**（例如分析結果決定要查哪一類法規），而且分支多到寫成條件邊（第 28 節）會很難維護。

## 七、常見誤判

| 誤判 | 實際 |
| --- | --- |
| 「每個代理寫自己的欄位，就不會互相覆蓋」 | 舊教材的結果欄位沒問題，`completed_agents` 是**共用**欄位，被蓋掉了 |
| 「循序執行沒報錯，所以 state 沒問題」 | 循序時默默蓋掉，並行時才報 `InvalidUpdateError` |
| 「reducer 去重了，所以不會重複發通知」 | 去重的是記錄，通知照樣發了兩次 |
| 「路由器說 done 就是完成」 | 要看 state 裡的結果欄位 |
| 「遞迴上限就是停止條件」 | 它是最後防線，到了結果全丟；要有更早的預算 |
| 「多代理比較先進」 | 本節的任務固定 workflow 只要 2 個超步 |

## Workshop：加 reducer 與去重，再和固定 workflow 比較

### 任務

1. （3 分鐘）跑一次，從第一段的前 6 步說明 `completed_agents` 為什麼一直只剩一個。
2. （4 分鐘）把 `Plain` 的 `completed_agents: list` 改成 `Annotated[list, union]`，再跑。第一段會變成什麼？第三段的 `Plain` 呢？看完改回來。
3. （4 分鐘）在第五段加一個情境：`make_router(["regulation", "report", "notification", "regulation"], budget=3)`。結果是什麼？收尾檢查通過嗎？
4. （4 分鐘）物流客訴情境（舊教材 Workshop）：分析 → SOP（需要分析）→ 回覆（需要分析與 SOP）。這需要 supervisor 嗎？畫出固定 workflow 的邊。

### 參考判讀

- 第 1 題：每個代理回傳只有自己的清單，沒有 reducer 就整個蓋掉。supervisor 看到的永遠只有最後一個完成的代理。
- 第 2 題：第一段變成正常完成，7 個超步；第三段的 `Plain` 也正常完成，2 個超步。改完之後 `Plain` 和 `Union` 就沒有差別了。
- 第 3 題：預算 3 用完前三個都派完了，第 4 次決策時預算已滿，回 `stop(預算 3 用完)`。收尾檢查**通過**：三個結果都在。這說明預算和收尾檢查是兩件事：停在 stop 不代表失敗，停在 done 也不代表成功，要看 state。
- 第 4 題：不需要。三步完全串列（`START → 分析 → SOP → 回覆 → END`），沒有並行的機會，也沒有依內容改變路徑。舊教材 Workshop 的「進階挑戰：用 LLM 讓 supervisor 判斷下一步」在這個情境只會增加風險。

### 驗收

- 第 2 題改完後的第一段與第三段輸出（並行合併）。
- 第四段兩種 reducer 的對照，加一句：為什麼 `log` 仍是 6 筆（重複輸入）。
- 第 3 題的輸出與判讀（停止條件）。
- 第 4 題的邊，以及「不需要多代理」的理由。

## 教師紀錄與下一步

- 證據等級：**本機實測**（langgraph 1.2.12 真實執行；代理是固定字串、路由決定是構造的序列，都不呼叫模型）。示範機 4-learn（Python 3.12.3）與備課機（Python 3.10.15）輸出逐字相同；備課機連跑 3 次逐字相同。
- 舊教材 [Supervisor](https://hackmd.io/@yillkid/JVCTksbDTRmmoqNJW2Q4-w) 的 `completed_agents: list[str]` 沒有 reducer，照抄執行會 `GraphRecursionError`，該頁的「執行結果」不是實際輸出。**舊頁尚未修改**，建議在頁首加上指向本節的說明。
- LLM supervisor 沒有實跑。第五段是構造的決定序列，代表 LLM 可能的輸出，不是任何模型的實測。
- 下一步：第 39 節用同一組案例比較單代理、固定流程與多代理的品質、延遲與成本。
