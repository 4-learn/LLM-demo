"""第 25 節：工具權限與副作用控制（只用 Python 標準函式庫）。

第 13 節的工具都是唯讀的。一旦工具會**改變世界**（寫入、通知、刪除），
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
