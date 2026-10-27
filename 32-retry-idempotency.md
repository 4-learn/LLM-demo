# 重試、退避與冪等

> **新版正文：20E07-E03-05｜對應新版 LLM 第 32 節：重試、退避與冪等。**
> 呼叫外部服務一定會失敗。本節回答三個問題：**哪些錯誤值得重試？要等多久、試幾次？重試會不會把同一件事做兩次？** 核心主張：**最危險的不是失敗，是「其實成功了但你不知道」。** 只用 Python 標準函式庫，時間由假時鐘記錄。

[回 51 節必修＋9 節選修課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 流程圖：重試、退避與冪等

```mermaid
flowchart TD
    A[送出工單請求與冪等鍵] --> B{分類錯誤}
    B -- 永久錯誤 --> C[停止並回報，不重試]
    B -- 暫時錯誤 --> D{重試預算尚有？}
    D -- 否 --> E[停止並回報失敗]
    D -- 是 --> F[FakeClock 記錄退避等待]
    F --> G[以相同冪等鍵重送]
    B -- 成功 --> H[服務端執行一次副作用並記錄鍵與結果]
    G --> I{服務端見過此鍵？}
    I -- 是 --> J[回傳既有結果，不重做副作用]
    I -- 否 --> H
    H --> K[回傳結果]
    J --> K

## 學習目標與時間

先修第 26 節（副作用分級）、第 27、28 節（state 與分支）。

| 時間 | 內容 |
| --- | --- |
| 0–8 分鐘 | 錯誤分類：暫時 vs 永久 |
| 8–18 分鐘 | 指數退避、jitter、重試預算 |
| 18–28 分鐘 | 「寫入成功但回應遺失」與冪等鍵 |
| 28–35 分鐘 | 跑 Demo：五個故障注入情境 |
| 35–50 分鐘 | Workshop：改預算、拿掉 jitter、錯用冪等鍵 |

驗收能力：交付**故障注入測試**，證明暫時錯誤可恢復、永久錯誤不盲目重試、**副作用最多發生一次**。

## 一、錯誤分類：值不值得再試一次

| 類別 | 例子 | 等一下會好嗎 | 處理 |
| --- | --- | --- | --- |
| **暫時** | 429 限流、503、逾時、連線重設 | 可能 | 有限次數重試 |
| **永久** | 401 未授權、400 schema 錯誤、404 | **不會** | **立刻停**，回報錯誤 |

對永久錯誤重試不只浪費：對 401 重試十次，可能觸發帳號鎖定；對 schema 錯誤重試，每一次都在付費。

> 分類要寫在程式裡（本節用兩個例外類別），不要讓模型判斷「要不要再試一次」。這跟第 26 節是同一個原則：**控制在程式，不在模型。**

## 二、等多久、試幾次

**指數退避**：第 n 次重試前等 `base × 2ⁿ` 秒，並設上限。本節是 0.5、1、2、4、8、8……

**jitter**：在 `[0, 上限]` 之間隨機取。沒有 jitter 的話，一百個客戶端在同一秒被限流，就會在同一秒重試——又一起被限流。這叫**驚群（thundering herd）**。

**重試預算**：最多試幾次。本節是 4 次（1 次原始＋3 次重試）。預算用完就放棄並回報，**不能無限重試**。

## 三、最危險的情況：其實成功了

```text
客戶端 ──請求──▶ 服務：寫入成功 ✓
客戶端 ◀──回應── ✗（在路上遺失／逾時）
客戶端：「逾時了，再試一次」──請求──▶ 服務：又寫入一次
```

從客戶端看，這就是一次逾時——**跟「根本沒送到」長得一模一樣**。如果這是第 26 節的 `notify_site`，工地會收到兩次停工通知。

**冪等鍵（idempotency key）**解決這件事：客戶端為**每一個請求**產生一把鍵，**每次重試都帶同一把**。服務端記住看過的鍵，再次看到時直接回傳上次的結果，不再執行。

| | 沒有冪等鍵 | 有冪等鍵 |
| --- | --- | --- |
| 原始請求 | 寫入 | 寫入，記住鍵 |
| 重試 | **再寫入一次** | 看到同一把鍵，回傳上次結果 |
| 結果 | 2 筆 | 1 筆 |

注意：**冪等要服務端配合**。客戶端自己產生一把鍵，但服務端不檢查，就沒有任何效果。

## 四、可執行 Demo

情境：一個模擬工單服務，用劇本注入故障。`FakeClock` 只記錄「本來會睡多久」，所以 Demo 瞬間跑完、每次結果一樣（每個情境各自固定 jitter 種子 `20261005`）。

<!-- demo: E07_retry_idempotency.py -->
```python
"""第 29 節：重試、退避與冪等（只用 Python 標準函式庫）。

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
```

執行指令（在 [`llm-demo`](https://github.com/4-learn/llm-demo) 目錄下，**不需要網路、不需要模型**）：

```bash
python3 E07_retry_idempotency.py
```

關鍵輸出（Linux／Python 3.10.15，20E07-E03-05）：

```text
=== 退避時間（不含 jitter）===
  第 1 次重試前 0.5s、第 2 次重試前 1.0s、第 3 次重試前 2.0s、第 4 次重試前 4.0s、第 5 次重試前 8.0s、第 6 次重試前 8.0s

=== 1 暫時錯誤兩次後成功 ===
  第 1 次：429 限流 → 等 0.138 秒後重試
  第 2 次：429 限流 → 等 0.296 秒後重試
  第 3 次：成功
  服務被呼叫 3 次｜實際建立工單 1 張｜累計等待 0.434 秒｜回傳 {'id': 'T1', 'site': 'A'}

=== 2 永久錯誤 ===
  第 1 次：401 未授權 → 永久錯誤，不重試
  服務被呼叫 1 次｜實際建立工單 0 張｜累計等待 0.000 秒｜回傳 None

=== 3 一直限流，預算用完 ===
  第 1 次：429 限流 → 等 0.138 秒後重試
  第 2 次：429 限流 → 等 0.296 秒後重試
  第 3 次：429 限流 → 等 0.309 秒後重試
  第 4 次：429 限流 → 預算用完，放棄
  服務被呼叫 4 次｜實際建立工單 0 張｜累計等待 0.743 秒｜回傳 None

=== 4 寫入成功但回應遺失，沒有冪等鍵 ===
  第 1 次：逾時（但寫入其實已完成） → 等 0.138 秒後重試
  第 2 次：成功
  服務被呼叫 2 次｜實際建立工單 2 張｜累計等待 0.138 秒｜回傳 {'id': 'T2', 'site': 'A'}

=== 5 寫入成功但回應遺失，有冪等鍵 ===
  第 1 次：逾時（但寫入其實已完成） → 等 0.138 秒後重試
  第 2 次：成功
  服務被呼叫 2 次｜實際建立工單 1 張｜累計等待 0.138 秒｜回傳 {'id': 'T1', 'site': 'A'}

=== 同一把鍵，不同的請求？ ===
  第一次 site=A → {'id': 'T1', 'site': 'A'}
  第二次 site=B → {'id': 'T1', 'site': 'A'}
  第二個請求被當成重試而吞掉了。冪等鍵必須代表「一個請求」，不是一個使用者或一個工地。
```

## 五、結果怎麼讀

### 1. 情境 4 與 5 是本節的核心對照

兩者的故障劇本、重試次數、等待時間**完全相同**，唯一差別是服務端有沒有檢查冪等鍵：

| 情境 | 服務被呼叫 | 實際建立工單 | 回傳 |
| --- | --- | --- | --- |
| 4 沒有冪等鍵 | 2 次 | **2 張** | `T2` |
| 5 有冪等鍵 | 2 次 | **1 張** | `T1` |

注意情境 4 的回傳值是 `T2`——**客戶端以為它建立了一張工單（T2），但系統裡其實有兩張，而且它根本不知道 T1 的存在。** 重試日誌也看不出問題：它只記錄了「逾時、重試、成功」。

### 2. 「服務被呼叫次數」不等於「副作用次數」

情境 5 呼叫了 2 次，但只寫入 1 次。驗收要看的是**副作用次數**（實際建立的工單），不是呼叫次數。課綱寫「副作用最多一次」，指的就是這個。

### 3. 永久錯誤只試一次

情境 2 遇到 401，**呼叫 1 次、等待 0 秒**就停了。如果把它當暫時錯誤，會在 4 次嘗試中浪費 3 次，而且每一次都一樣會失敗。

### 4. 預算用完要明確失敗

情境 3 一直限流，4 次後放棄、回傳 `None`。呼叫端必須處理這個 `None`——把它當成「失敗」往上回報，**而不是當成「沒有工單」繼續流程**。這與第 28 節的原則相同：未知結果必須走保守路徑。

### 5. 同一把鍵用在不同請求上

最後一段：呼叫端把 `req-001` 用在兩個不同工地的請求上，**第二個請求被當成重試、靜默吞掉了**。site=B 的工單永遠不會被建立，而回傳值看起來一切正常（只是內容是 site=A 的）。

冪等鍵必須代表「這一個請求」。常見的錯誤是用使用者 ID、工地代號或日期當鍵——它們都會讓不同的請求撞在一起。

## 六、常見誤判

| 誤判 | 為什麼錯 |
| --- | --- |
| 「失敗就重試」 | 永久錯誤重試只是浪費，還可能觸發鎖定 |
| 「固定每次等 1 秒」 | 沒有 jitter，所有客戶端同時重試，一起再被限流 |
| 「逾時代表沒寫入」 | 寫入可能已完成、只是回應遺失（情境 4） |
| 「重試日誌顯示成功就沒問題」 | 情境 4 的日誌一切正常，系統裡卻有兩張工單 |
| 「客戶端產生冪等鍵就好」 | 服務端不檢查，鍵沒有任何作用 |
| 「用使用者 ID 當冪等鍵」 | 同一使用者的不同請求會互相吞掉 |
| 「重試到成功為止」 | 必須有預算；放棄時要明確回報失敗 |

## Workshop：故障注入

### 任務

1. 執行 Demo，先預測情境 1～5 各自的「服務被呼叫次數」與「實際建立工單」，再核對。
2. 新增一個情境：劇本是 `("rate_limited", "timeout_after_write")`、有冪等鍵、預算 4。預測呼叫次數與工單數，再跑。
3. 把情境 1 的預算改成 `2`，重跑。結果是什麼？**重試預算太小的代價是什麼？**
4. 把 `backoff` 呼叫時的 `rng` 拿掉（不加 jitter），重跑情境 3。每次等待時間是多少？如果有 100 個客戶端同時被限流，它們會在什麼時候重試？
5. 寫一個測試：對劇本 `("timeout_after_write",)` 在**有冪等鍵**的情況下重複呼叫 `with_retry` 三次（都用同一把鍵），斷言工單總數為 1。

### 參考判讀

- 第 2 題：第 1 次限流（未寫入）、第 2 次寫入成功但逾時、第 3 次看到同一把鍵直接回傳 → **呼叫 3 次、工單 1 張**。
- 第 3 題：第 1 次限流、第 2 次限流後預算用完 → **呼叫 2 次、工單 0 張、回傳 `None`**。這個請求本來第 3 次就會成功。預算太小會把可恢復的錯誤變成失敗；太大則在服務真的掛掉時拖很久。
- 第 4 題：不加 jitter 時等待時間固定為 **0.5、1.0、2.0 秒**（累計 3.5 秒）。100 個客戶端會在**完全相同的時間點**重試，一起再被限流。
- 第 5 題：工單總數為 1。三次呼叫都帶 `req-site-A-001`，服務端第一次寫入並記住，之後都回傳同一張。

### 驗收

- 交出故障注入測試，至少包含：暫時錯誤後成功、永久錯誤不重試、預算用完放棄。
- 至少一個測試**證明副作用最多一次**：注入「寫入成功但回應遺失」，斷言實際寫入次數為 1（不是斷言呼叫次數）。
- 寫明你的冪等鍵怎麼產生、為什麼能代表「一個請求」。
- 只操作模擬服務；不得對真實 API 注入故障。

## 教師紀錄與下一步

- 本節**完全不需要模型、網路或任何 SDK**。`TicketService` 是記憶體內的模擬服務，`FakeClock` 不真的睡。證據等級：**本機實測（重試邏輯在注入故障下的行為）**。
- 撰寫時的實際紀錄：第一版用**一個**亂數產生器跑完所有情境，結果每個情境的 jitter 值取決於前面情境用掉幾個亂數——情境 4、5 的等待時間不同（0.056 對 0.144 秒），讓「唯一差別是冪等鍵」的對照不乾淨。改為每個情境各自固定種子之後，4、5 的等待時間一致。
- 真實 LLM API 的重試：多數供應商的 SDK 內建重試，**要先查清楚它重試哪些錯誤、幾次**，否則你自己外面再包一層，實際嘗試次數會是兩者相乘。供應商回傳的 `Retry-After` 標頭應優先於自算的退避時間。
- 第 30 節接上：HITL 恢復時節點可能**重跑**，所以核准後的副作用也必須冪等——跟本節情境 5 是同一個機制。
- 待教師確認：課堂是否要示範某個 SDK 內建重試的實際設定（需鎖定版本與查核日期）。
