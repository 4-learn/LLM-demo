"""第 28 節：重試、退避與冪等（只用 Python 標準函式庫）。

三個問題，三個機制：
  - 哪些錯誤值得重試？      → 錯誤分類：暫時 vs 永久
  - 重試要等多久、試幾次？  → 指數退避＋jitter＋重試預算
  - 重試會不會做兩次？      → 冪等鍵（idempotency key）

時間不是真的流逝：`FakeClock` 只記錄「本來會睡多久」，所以 Demo 瞬間跑完、結果可重現。
"""

import random


class TransientError(Exception):
    """限流、逾時、暫時網路故障。等一下可能就好。"""


class PermanentError(Exception):
    """授權失敗、schema 錯誤、資源不存在。等多久都不會好。"""


class FakeClock:
    def __init__(self):
        self.slept = []

    def sleep(self, seconds):
        self.slept.append(round(seconds, 3))


class TicketService:
    """模擬寫入服務。`failures` 是一個劇本：每次呼叫依序取出一個要發生的事。

    關鍵細節：`"timeout_after_write"` 表示**寫入已經成功，但回應在路上遺失**——
    呼叫端只看到逾時，不知道寫入其實發生了。這是重複寫入的真正來源。
    """

    def __init__(self, failures=(), honour_keys=True):
        self.failures = list(failures)
        self.honour_keys = honour_keys
        self.tickets = []
        self.seen_keys = {}
        self.calls = 0

    def create(self, site, key):
        self.calls += 1
        event = self.failures.pop(0) if self.failures else "ok"
        if event == "rate_limited":
            raise TransientError("429 限流")
        if event == "unauthorized":
            raise PermanentError("401 未授權")
        if self.honour_keys and key in self.seen_keys:
            return self.seen_keys[key]          # 同一把鍵：回傳上次的結果，不再寫
        ticket = {"id": f"T{len(self.tickets) + 1}", "site": site}
        self.tickets.append(ticket)
        if self.honour_keys:
            self.seen_keys[key] = ticket
        if event == "timeout_after_write":
            raise TransientError("逾時（但寫入其實已完成）")
        return ticket


def backoff(attempt, base=0.5, cap=8.0, rng=None):
    """第 attempt 次重試前要等多久：指數成長、有上限、加 full jitter。"""
    ceiling = min(cap, base * (2 ** attempt))
    return rng.uniform(0, ceiling) if rng else ceiling


def with_retry(operation, max_attempts, clock, rng=None):
    log = []
    for attempt in range(max_attempts):
        try:
            result = operation()
            log.append(f"第 {attempt + 1} 次：成功")
            return result, log
        except PermanentError as error:
            log.append(f"第 {attempt + 1} 次：{error} → 永久錯誤，不重試")
            return None, log
        except TransientError as error:
            if attempt + 1 == max_attempts:
                log.append(f"第 {attempt + 1} 次：{error} → 預算用完，放棄")
                return None, log
            wait = backoff(attempt, rng=rng)
            clock.sleep(wait)
            log.append(f"第 {attempt + 1} 次：{error} → 等 {wait:.3f} 秒後重試")
    return None, log


SCENARIOS = (
    ("1 暫時錯誤兩次後成功", ("rate_limited", "rate_limited"), True, 4),
    ("2 永久錯誤", ("unauthorized",), True, 4),
    ("3 一直限流，預算用完", ("rate_limited",) * 10, True, 4),
    ("4 寫入成功但回應遺失，沒有冪等鍵", ("timeout_after_write",), False, 4),
    ("5 寫入成功但回應遺失，有冪等鍵", ("timeout_after_write",), True, 4),
)


def main():
    print("=== 退避時間（不含 jitter）===")
    print("  " + "、".join(f"第 {n + 1} 次重試前 {backoff(n):.1f}s" for n in range(6)))
    print()
    for title, failures, keys, budget in SCENARIOS:
        service = TicketService(failures, honour_keys=keys)
        clock = FakeClock()
        rng = random.Random(20261005)   # 每個情境各自固定種子，不受情境順序影響
        key = "req-site-A-001"          # 同一個請求的每次重試都用同一把鍵
        result, log = with_retry(lambda: service.create("A", key), budget, clock, rng)
        print(f"=== {title} ===")
        for line in log:
            print(f"  {line}")
        print(f"  服務被呼叫 {service.calls} 次｜實際建立工單 {len(service.tickets)} 張"
              f"｜累計等待 {sum(clock.slept):.3f} 秒｜回傳 {result}")
        print()

    print("=== 同一把鍵，不同的請求？ ===")
    service = TicketService()
    first = service.create("A", "req-001")
    second = service.create("B", "req-001")          # 呼叫端錯用了同一把鍵
    print(f"  第一次 site=A → {first}")
    print(f"  第二次 site=B → {second}")
    print("  第二個請求被當成重試而吞掉了。冪等鍵必須代表「一個請求」，不是一個使用者或一個工地。")


if __name__ == "__main__":
    main()
