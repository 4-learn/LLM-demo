# 41｜Agent 設計模式與記憶

> **正文改版：2026-10-24｜對應新版 LLM 第 41 節：Agent 設計模式與記憶。**本節以純標準庫的模擬執行器辨認 workflow 與 agent 的分界，並實作停止條件及有來源、可拒收的記憶；本節不是自主模型或生產代理框架實測。

[回 51 節必修＋9 節選修課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 流程圖：Agent 設計模式與記憶

```mermaid
flowchart TD
    A[輸入 observations 與記憶] --> B{選擇模式}
    B -- 固定 workflow --> C[依預定步驟執行]
    B -- 受限 agent --> D[觀察當前輸入]
    D --> E{完成／步數上限？}
    E -- 否 --> F[依固定政策選 lookup 或 summarize]
    F --> G{工具在 allowlist？}
    G -- 否 --> H[停止：blocked]
    G -- 是 --> I[執行工具並記錄 trace]
    I --> D
    E -- 是且完成 --> J[停止：done]
    E -- 達上限 --> K[停止：limit]
    L[新增記憶] --> M{來源可信且已核實？}
    M -- 否 --> N[拒收記憶]
    M -- 是 --> O[保存來源與核實狀態]
    C --> P[比較固定路徑與 agent 選擇／停止結果]
    H --> P
    J --> P
    K --> P

## 學習目標與時間

先備：能讀函式、列表及條件分支；不要求模型、工具套件或網路。50 分鐘：概念 35 分鐘（workflow/agent 10、停止與權限 10、記憶污染 15），Workshop 15 分鐘。

## 一、固定流程是 workflow，選擇行動才是 agent

Workflow 由設計者預先固定節點與轉移，例如「驗證輸入→查資料→輸出」，輸入不同可產生不同結果，但路徑骨架已定。Agent 則在受限範圍內依觀察選擇下一個工具或步驟，迴圈可能因新結果而調整。兩者不是優劣二分：固定且可預測的工作優先採 workflow；只有需要根據觀察選擇行動時，才引入 agent 的額外複雜性。即使是 agent，也應將工具白名單、權限與狀態界線寫清楚。選擇行動的能力並不意味著自行授權：讀取與寫入工具應分權，執行前檢查參數和使用者權限，對外寄送、刪除或付款等副作用要有人工核准或可逆方案。設計文件也要說明哪些步驟確定、哪些由策略決定，並以 trace 讓操作者能重建決策路徑。

## 二、停止條件是設計的一部分

每次迴圈都要有可檢查的停止規則，例如完成任務、需要人工確認、達到步數上限或工具失敗。若只依賴模型說「完成」，可能無限呼叫、反覆重試或在未驗證時結束。示例設定最大兩步；耗盡步數時回傳 `limit`，而不是捏造成功。實務還要記錄每步輸入、工具名稱、結果摘要與停止原因，並將不可逆副作用交給明確授權或人工核准。

## 三、記憶要有來源、範圍與有效期

Agent 記憶會影響後續決策，錯誤記憶可能由使用者誤述、模型幻覺、舊資料或惡意提示造成污染。記憶項目至少應帶來源、可信狀態、用途範圍與時間；未核實的說法不可升格為事實，也不可把一個情境的私人資訊帶入另一個情境。最小化保存、可檢視刪除、明確到期與拒絕不可信新增，能降低污染持續擴散的機會，但不能保證完全免疫。更進一步，可將使用者提供內容標為「待核實」，將工具回傳附上來源時間，並以獨立可信資料核對後才升級狀態。記憶檢索也應按任務和租戶範圍過濾，不宜把所有歷史對話全文塞入提示。

## 四、可執行 Demo

程式模擬「觀察→依固定政策選工具→取結果」的受限循環，不呼叫模型。記憶只接受標示為核實的來源；迴圈達上限就明確停止。執行：`python3 41_agent_patterns.py`。

<!-- demo: 41_agent_patterns.py -->
```python
ALLOWED_TOOLS = {"lookup", "summarize"}


TRUSTED_SOURCES = {"policy:v2", "manual:approved"}


def remember(memory, text, source, verified=False):
    if type(verified) is not bool or not verified:
        return False
    if not isinstance(source, str) or source not in TRUSTED_SOURCES:
        return False
    if not isinstance(text, str) or not text.strip():
        return False
    memory.append({"text": text, "source": source, "verified": True})
    return True


def run_agent(observations, max_steps=2):
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be positive")
    if not isinstance(observations, (list, tuple)) or any(not isinstance(x, str) for x in observations):
        raise ValueError("observations must be a list or tuple of strings")
    trace = []
    for step, observation in enumerate(observations[:max_steps], 1):
        tool = "lookup" if "查" in observation else "summarize"
        if tool not in ALLOWED_TOOLS:
            return trace, "blocked"
        trace.append((step, tool, observation))
        if "完成" in observation:
            return trace, "done"
    return trace, "limit"


if __name__ == "__main__":
    memory = []
    print("unverified accepted:", remember(memory, "使用者說資料已核准", "user", False))
    remember(memory, "政策版本 v2 已核實", "policy:v2", True)
    print("memory:", memory)
    print("run:", run_agent(["查詢工單", "整理完成", "額外步驟"]))
    print("limit:", run_agent(["整理摘要", "整理摘要", "完成"], max_steps=2)[1])
```

關鍵輸出（本機實測；標準函式庫、合成觀察）：

```text
unverified accepted: False
memory: [{'text': '政策版本 v2 已核實', 'source': 'policy:v2', 'verified': True}]
run: ([(1, 'lookup', '查詢工單'), (2, 'summarize', '整理完成')], 'done')
limit: limit
```

## Workshop

15 分鐘：5 分鐘把觀察改成「查詢完成」，確認何時停止；5 分鐘把 `max_steps` 設 1，確認未完成時回傳 `limit`；5 分鐘測試記憶污染拒收與工具白名單。參考判讀：未核實記憶不得被保存；工具只從允許集合選擇；上限耗盡不等同完成。可執行驗收：執行 Demo 後加入 `assert not remember([], "未證實傳聞", "user")`、`assert run_agent(["整理摘要"], 1)[1] == "limit"`、`assert run_agent(["查詢完成"], 2)[1] == "done"`；另確認 `max_steps=0` 會引發 `ValueError`。若改成工具輸入外部字串，必須先驗證工具名稱，不可直接執行任意名稱。

## 教師紀錄與下一步

證據級別：本機實測為本教材程式於 2026-12-06 直接執行，環境 Linux、Python 3（標準函式庫，微版本未記錄），無模型 revision，量測方式為 stdout；全部輸入皆為合成觀察。此教學模型不模擬 LLM 決策、提示注入、持久儲存或併發，記憶的 `verified` 布林值只是教學閘門，不是真實可信度證明。尚未實班試教，沒有學生實測。下一節可銜接多代理協作與共享狀態風險；舊教材連結僅供歷史參照，未宣稱同步。
