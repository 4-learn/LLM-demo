# 工具權限與副作用控制

> **新版正文：20E07-E03-05｜對應新版 LLM 第 26 節：工具權限與副作用控制。**
> 第 14 節的工具都是唯讀的。一旦工具會**改變世界**——寫入、通知、刪除——問題就從「答案對不對」變成「這件事該不該發生」。本節的證據只有一種：**執行器有沒有真的執行。** 只用 Python 標準函式庫、只操作沙盒合成資料。

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先修第 09 節（控制在工具層）、第 14 節（完整工具回合）、第 25 節（pipeline）。

| 時間 | 內容 |
| --- | --- |
| 0–8 分鐘 | 副作用分級：讀、寫、通知、刪除 |
| 8–20 分鐘 | 政策表：allowlist、角色、參數範圍、核准、dry-run |
| 20–28 分鐘 | 決定與執行分開；為什麼證據只能是執行器紀錄 |
| 28–35 分鐘 | 跑 Demo：八個請求，看實際寫入次數 |
| 35–50 分鐘 | Workshop：為新工具設政策並寫拒絕測試 |

驗收能力：交付**政策表與拒絕測試**，每一個拒絕都以**執行器紀錄**證明沒有執行；不能用「模型說已拒絕」代替。

## 一、副作用分級

| 等級 | 例子 | 出錯的代價 | 可以撤回嗎 |
| --- | --- | --- | --- |
| **read** | 查詢事件數 | 答錯，但世界沒變 | 不需要 |
| **write** | 建立工單 | 多一張錯的工單 | 通常可以 |
| **notify** | 通知工地停工 | **人已經收到了** | **不行** |
| **delete** | 刪除事件紀錄 | 資料消失 | 看有沒有備份 |

**通知比寫入危險**，這是最常被低估的一點：寫錯的資料可以改，發出去的通知收不回來。所以本節的政策讓 `notify_site` 需要核准，`create_ticket` 不需要。

## 二、政策表

每個工具一列，欄位都是**程式檢查的**，不是寫在提示裡的：

| 欄位 | 意思 | 檢查順序 |
| --- | --- | --- |
| allowlist | 工具在不在表上 | 1 |
| 角色 | 誰可以用 | 2 |
| 參數範圍 | 例如工單優先級只能 low／medium | 3 |
| dry-run | 只回報會做什麼 | 4 |
| 核准 | 需要人工按下同意 | 5 |

**最小能力原則**：先給零，需要再加。本節的 `delete_event` 的可用角色是**空集合**——連 manager 都不行，即使已核准。如果真的需要刪除，那是另一個流程（有備份、有雙人覆核），不是讓 LLM 工具回合去做。

## 三、決定與執行要分開

本節 Demo 把它拆成兩個函式：

```python
decide(tool, args, role, approved, dry_run)  # 純函式，回傳（決定, 原因），不碰任何資料
call(sandbox, tool, args, role, ...)         # 先 decide，只有「執行」才真的呼叫
```

分開的理由：

1. **`decide` 可以單獨測試**，不需要建資料、不會有副作用。你可以對它跑幾百個組合。
2. **執行只有一條路徑**。執行器表 `executors()` 是明確列出的字典，不用 `getattr` 依名稱動態找函式——那等於沒有 allowlist（第 14 節）。
3. **證據來自沙盒**。`Sandbox.writes` 記下每一次真的改變狀態的動作，測試只看它。

## 四、為什麼「模型說已拒絕」不算數

模型的回覆和執行器的行為是**兩條獨立的路徑**。模型可以說「我已拒絕」，同時你的程式裡有一條沒經過 `decide()` 的路徑真的執行了刪除——**兩件事可以同時成立**。本節 Demo 最後一段就是把這個情況印出來。

所以驗收的證據只有一種：**執行器紀錄**。課綱寫「不能用模型說已拒絕代替執行器證據」，就是這個意思。

## 五、可執行 Demo

情境：一個沙盒、四個工具、八個請求。每個請求印出決定與**實際寫入次數**。

<!-- demo: E07_side_effects.py -->
```python
"""第 26 節：工具權限與副作用控制（只用 Python 標準函式庫）。

第 14 節的工具都是唯讀的。一旦工具會**改變世界**（寫入、通知、刪除），
問題就從「答案對不對」變成「這件事該不該發生」。

本節的證據只有一種：**執行器有沒有真的執行**。
模型說「我已拒絕」不算數；`Sandbox.writes` 的長度才算數。
"""

import json

# 每個工具一列政策。副作用等級決定預設行為，不是由模型決定。
POLICY = {
    "count_events":  {"effect": "read",   "roles": {"staff", "manager"}, "approval": False},
    "create_ticket": {"effect": "write",  "roles": {"staff", "manager"}, "approval": False,
                      "limits": {"priority": {"low", "medium"}}},
    "notify_site":   {"effect": "notify", "roles": {"manager"},          "approval": True},
    "delete_event":  {"effect": "delete", "roles": set(),                "approval": True},
}


class Sandbox:
    """合成資料的沙盒。所有副作用都記在這裡，測試只看這裡。"""

    def __init__(self):
        self.events = [{"id": "E1", "site": "A"}, {"id": "E2", "site": "B"}]
        self.tickets = []
        self.notifications = []
        self.writes = []                     # 每一次真的改變狀態，都記一筆

    def count_events(self, site):
        return {"count": sum(1 for e in self.events if e["site"] == site)}

    def create_ticket(self, site, priority):
        ticket = {"id": f"T{len(self.tickets) + 1}", "site": site, "priority": priority}
        self.tickets.append(ticket)
        self.writes.append(("create_ticket", ticket["id"]))
        return ticket

    def notify_site(self, site, message):
        self.notifications.append((site, message))
        self.writes.append(("notify_site", site))
        return {"sent": True}

    def delete_event(self, event_id):
        self.events = [e for e in self.events if e["id"] != event_id]
        self.writes.append(("delete_event", event_id))
        return {"deleted": event_id}


def decide(tool, args, role, approved=False, dry_run=False):
    """純函式：回傳 (決定, 原因)。不碰沙盒——決定和執行要分開，才測得了決定。"""
    rule = POLICY.get(tool)
    if rule is None:
        return "拒絕", "不在 allowlist"
    if role not in rule["roles"]:
        return "拒絕", f"角色 {role} 無權使用 {tool}"
    for key, allowed in rule.get("limits", {}).items():
        if args.get(key) not in allowed:
            return "拒絕", f"{key}={args.get(key)!r} 超出允許範圍 {sorted(allowed)}"
    if dry_run and rule["effect"] != "read":
        return "預演", "dry-run：只回報會做什麼，不執行"
    if rule["approval"] and not approved:
        return "待核准", f"{rule['effect']} 類工具需要人工核准"
    return "執行", rule["effect"]


def executors(sandbox):
    """明確的執行器表。只有這張表上的函式能被呼叫——不用 getattr 動態找。"""
    return {"count_events": sandbox.count_events, "create_ticket": sandbox.create_ticket,
            "notify_site": sandbox.notify_site, "delete_event": sandbox.delete_event}


def call(sandbox, tool, args, role, approved=False, dry_run=False):
    verdict, reason = decide(tool, args, role, approved, dry_run)
    result = executors(sandbox)[tool](**args) if verdict == "執行" else None
    return verdict, reason, result


CASES = (
    ("唯讀查詢",               "count_events",  {"site": "A"},                        "staff",   False, False),
    ("建立工單（範圍內）",      "create_ticket", {"site": "A", "priority": "low"},     "staff",   False, False),
    ("建立工單（越權參數）",    "create_ticket", {"site": "A", "priority": "urgent"},  "staff",   False, False),
    ("通知工地（staff）",       "notify_site",   {"site": "A", "message": "停工"},     "staff",   False, False),
    ("通知工地（未核准）",      "notify_site",   {"site": "A", "message": "停工"},     "manager", False, False),
    ("通知工地（dry-run）",     "notify_site",   {"site": "A", "message": "停工"},     "manager", False, True),
    ("通知工地（已核准）",      "notify_site",   {"site": "A", "message": "停工"},     "manager", True,  False),
    ("刪除事件（manager 核准）", "delete_event",  {"event_id": "E2"},                   "manager", True,  False),
)


def main():
    print("=== 政策表 ===")
    print(f"  {'工具':<14} {'副作用':<7} {'可用角色':<18} {'需核准'}")
    for tool, rule in POLICY.items():
        roles = "、".join(sorted(rule["roles"])) or "（無人）"
        print(f"  {tool:<14} {rule['effect']:<7} {roles:<18} {'是' if rule['approval'] else '否'}")
    print()

    sandbox = Sandbox()
    print("=== 八個請求 ===")
    for label, tool, args, role, approved, dry_run in CASES:
        before = len(sandbox.writes)
        verdict, reason, _ = call(sandbox, tool, args, role, approved, dry_run)
        executed = len(sandbox.writes) - before
        print(f"  {label:<22} {verdict:<4} 實際寫入 {executed}  ← {reason}")
    print()

    print("=== 執行器證據 ===")
    print(f"  沙盒寫入紀錄：{sandbox.writes}")
    print(f"  工單 {len(sandbox.tickets)} 張、通知 {len(sandbox.notifications)} 則、"
          f"事件剩 {[e['id'] for e in sandbox.events]}")
    print()

    print("=== 「模型說已拒絕」為什麼不算數 ===")
    claim = {"role": "assistant", "content": "我已拒絕這個刪除請求。"}
    bad = Sandbox()
    bad.delete_event(event_id="E1")                      # 一條繞過 decide() 的程式路徑
    print(f"  模型的回覆：{claim['content']}")
    print(f"  同一時間沙盒的寫入紀錄：{bad.writes}")
    print("  兩者可以同時成立。驗收只看右邊。")


if __name__ == "__main__":
    main()
```

執行指令（在 [`llm-demo`](https://github.com/4-learn/llm-demo) 目錄下，**不需要網路、不需要模型**）：

```bash
python3 E07_side_effects.py
```

關鍵輸出（Linux／Python 3.10.15，20E07-E03-05）：

```text
=== 政策表 ===
  工具             副作用     可用角色               需核准
  count_events   read    manager、staff      否
  create_ticket  write   manager、staff      否
  notify_site    notify  manager            是
  delete_event   delete  （無人）               是

=== 八個請求 ===
  唯讀查詢                   執行   實際寫入 0  ← read
  建立工單（範圍內）              執行   實際寫入 1  ← write
  建立工單（越權參數）             拒絕   實際寫入 0  ← priority='urgent' 超出允許範圍 ['low', 'medium']
  通知工地（staff）            拒絕   實際寫入 0  ← 角色 staff 無權使用 notify_site
  通知工地（未核准）              待核准  實際寫入 0  ← notify 類工具需要人工核准
  通知工地（dry-run）          預演   實際寫入 0  ← dry-run：只回報會做什麼，不執行
  通知工地（已核准）              執行   實際寫入 1  ← notify
  刪除事件（manager 核准）       拒絕   實際寫入 0  ← 角色 manager 無權使用 delete_event

=== 執行器證據 ===
  沙盒寫入紀錄：[('create_ticket', 'T1'), ('notify_site', 'A')]
  工單 1 張、通知 1 則、事件剩 ['E1', 'E2']

=== 「模型說已拒絕」為什麼不算數 ===
  模型的回覆：我已拒絕這個刪除請求。
  同一時間沙盒的寫入紀錄：[('delete_event', 'E1')]
  兩者可以同時成立。驗收只看右邊。
```

## 六、結果怎麼讀

### 1. 八個請求，只有兩次真的寫入

```text
沙盒寫入紀錄：[('create_ticket', 'T1'), ('notify_site', 'A')]
```

這一行就是整節的驗收證據。八個請求裡，兩個寫入工具被執行、一個唯讀工具被執行（不產生寫入），其餘五個——越權參數、角色不符、未核准、dry-run、刪除——**實際寫入全是 0**。

### 2. 同一個工具，四種結果

`notify_site` 出現了四次，結果各不相同：

| 情境 | 決定 | 原因 |
| --- | --- | --- |
| staff 呼叫 | 拒絕 | 角色不符——**在核准之前就擋下**，staff 連申請核准的資格都沒有 |
| manager 未核准 | 待核准 | 角色對，但 notify 需要核准 |
| manager dry-run | 預演 | 只回報會做什麼 |
| manager 已核准 | 執行 | 全部條件通過 |

**檢查順序很重要。** 如果先檢查核准再檢查角色，staff 會收到「待核准」——這等於告訴他「找人按一下就可以」，而實際上他根本沒有這個權限。

### 3. 「已核准」不能繞過角色

最後一個請求是 manager **已核准**的刪除，結果仍是**拒絕**。核准是一個**額外的**條件，不是萬能鑰匙。`delete_event` 的可用角色是空集合，所以任何人、任何核准狀態都不能透過這條路徑刪除。

### 4. dry-run 也必須是零寫入

dry-run 回傳「預演」，實際寫入 0。這聽起來理所當然，但很多實作的 dry-run 是「執行，然後 rollback」——**中間如果發了通知，rollback 救不回來**。本節的 dry-run 在 `decide` 層就停住，根本不碰執行器。

## 七、常見誤判

| 誤判 | 為什麼錯 |
| --- | --- |
| 「模型回覆說已拒絕」 | 模型回覆與執行器行為是兩條路徑；驗收只看執行器紀錄 |
| 「在提示裡寫『刪除前要先確認』」 | 那是期望不是控制（第 09 節）；核准要在程式裡檢查 |
| 「寫入比通知危險」 | 通知收不回來；寫錯的資料通常改得回來 |
| 「已核准就可以執行」 | 核准是額外條件；角色與參數範圍仍要先過 |
| 「dry-run 就是執行再 rollback」 | 有外部副作用時 rollback 無效；dry-run 要在執行前停住 |
| 「用 `getattr(obj, name)` 找工具比較彈性」 | 彈性就是問題：任何方法名都能被呼叫 |
| 「先給完整權限，有問題再收」 | 最小能力：先給零，`delete_event` 的角色是空集合 |

## Workshop：為新工具設政策並寫拒絕測試

### 任務

1. 執行 Demo，先預測八個請求各自的決定與寫入次數，再核對。
2. 新增工具 `close_ticket(ticket_id)`（write 類），只有 manager 可用、不需核准。在 `POLICY`、`Sandbox`、`executors()` 三處都加上。**少加一處會發生什麼事？**分別試試看只加 `POLICY` 不加 `executors()`。
3. 把 `decide` 裡「角色」與「核准」兩段檢查的順序對調，重跑。staff 呼叫 `notify_site` 的決定變成什麼？**這為什麼是資訊洩漏？**
4. 寫一個測試：對 `delete_event` 窮舉所有角色（含不存在的角色）× 核准（是／否）× dry-run（是／否），斷言**沙盒寫入次數永遠是 0**。
5. 把 `create_ticket` 的 `limits` 拿掉，重跑。越權參數的請求變成什麼？

### 參考判讀

- 第 2 題：只加 `POLICY` 不加 `executors()` 時，`decide` 回傳「執行」，接著 `executors(sandbox)[tool]` 拋出 `KeyError`。**這是好的失敗**——政策說可以，但沒有執行器，所以什麼都沒發生。反過來（只加執行器不加政策）則會在 `decide` 被擋為「不在 allowlist」。兩邊都要加，才是完整的上架。
- 第 3 題：對調後 staff 會得到「**待核准**」。這告訴他「這個動作存在、只差核准」，而事實上他沒有權限。在真實系統裡，這會引導使用者去找主管「幫忙按一下」——**社交工程就是從這種訊息開始的**。
- 第 4 題：以 staff、manager、不存在的角色三種計，3 × 2 × 2 = **12 組**，寫入次數全為 0。如果你的測試只測「manager 已核准」這一組，你就沒有測到 dry-run 與未核准的路徑。
- 第 5 題：`priority='urgent'` 的請求變成「**執行**」，沙盒多一張 urgent 工單。參數範圍是政策的一部分，不是可有可無的驗證。

### 驗收

- 交出**政策表**：每個工具的副作用等級、可用角色、參數範圍、是否需核准。
- 交出**拒絕測試**：至少涵蓋越權角色、越權參數、未核准三種，每一個都斷言**沙盒寫入次數為 0**。
- 至少一個測試證明 dry-run 不產生寫入。
- 不得用模型回覆的文字作為拒絕的證據。
- 只能操作沙盒合成資料；不得對真實系統執行任何寫入、通知或刪除。

## 教師紀錄與下一步

- 本節**完全不需要模型、網路或任何 SDK**。`Sandbox` 是記憶體內的合成資料，證據等級：**本機實測（執行器在政策下的行為）**。
- 撰寫時的實際紀錄：第一版 `call()` 用 `getattr(sandbox, tool)` 依名稱找執行器——正是本節「常見誤判」表裡警告的寫法。雖然 `decide` 已先擋掉不在 `POLICY` 的名稱，但這讓「允許什麼」同時依賴兩個地方（政策表、以及沙盒恰好有哪些方法），改成明確的 `executors()` 字典。
- 第 29 節接上：重試時同一個寫入可能被執行兩次，需要冪等鍵；第 30 節接上：「待核准」之後的中斷、恢復與拒絕分支。
- 待教師確認：真實課堂若接 LangGraph 的 tool node，政策檢查應放在 tool node 之前的獨立節點，還是包在每個工具函式裡——兩者在第 30 節的 checkpoint 恢復時行為不同。
