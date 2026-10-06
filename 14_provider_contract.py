"""第 14 節：多供應商介面、契約測試與 API 成本。

「OpenAI 相容」只保證請求長得像，不保證每個參數都有效、usage 都可信、價格都一樣。
本程式只用標準函式庫，讀兩份資料：教師用 LiteLLM 產生的能力快照 data/provider-snapshot.json，
以及教師手動到官方價目頁核對的 data/provider-prices-checked.json；
不呼叫任何付費 API。真實呼叫的部分在講義的 Workshop 由教師示範或以 fixtures 重播。
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = ("reported", "estimated", "simulated")   # 供應商回報／本地估算／離線模擬


def load(name):
    for base in (HERE / "data", HERE.parent / "llm" / "data"):
        path = base / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"找不到 data/{name}")


def price_drift(snapshot, checked):
    """LiteLLM 內建價目 vs 教師核對的官方價。回傳 {代號: (litellm, 官方, 是否相同)}。"""
    out = {}
    for alias, m in snapshot["models"].items():
        lite = (m["usd_per_1m_input"], m["usd_per_1m_output"])
        official = (checked["models"][alias]["usd_per_1m_input"], checked["models"][alias]["usd_per_1m_output"])
        out[alias] = (lite, official, lite == official)
    return out


def mark(value):
    """True／False／None 三種都要分開：None 是「沒寫」，不是「不支援」。"""
    return {True: "有", False: "無", None: "?"}[value]


def capability_matrix(models):
    rows = []
    for alias, m in models.items():
        rows.append([alias, mark(m["flags"]["supports_function_calling"]),
                     mark(m["flags"]["supports_response_schema"]), mark(m["flags"]["supports_vision"]),
                     mark(m["openai_params"]["seed"]), mark(m["openai_params"]["logprobs"]),
                     mark(m["openai_params"]["n"]), f'{m["max_input_tokens"]:,}'])
    return rows


def contract_check(model, required_params, required_flags=()):
    """請求需要的參數，這個供應商是否全部支援？回傳缺少的清單（空 = 通過）。"""
    missing = [p for p in required_params if not model["openai_params"].get(p)]
    unknown = [f for f in required_flags if model["flags"].get(f) is None]
    missing += [f for f in required_flags if model["flags"].get(f) is False]
    return missing, unknown


def cost_usd(model, input_tokens, output_tokens):
    return (input_tokens * model["usd_per_1m_input"] + output_tokens * model["usd_per_1m_output"]) / 1e6


class Usage:
    """一筆用量必須帶來源。不同來源的數字不能混加成同一個「實際花費」。"""

    def __init__(self, input_tokens, output_tokens, source):
        if source not in SOURCES:
            raise ValueError(f"source 必須是 {SOURCES} 之一，收到 {source!r}")
        self.input_tokens, self.output_tokens, self.source = input_tokens, output_tokens, source


class Budget:
    """送出前就擋：用「輸入 + max_tokens 上限」算最壞情況，而不是送出後才記帳。"""

    def __init__(self, max_usd, max_requests):
        self.max_usd, self.max_requests = max_usd, max_requests
        self.spent = {source: 0.0 for source in SOURCES}
        self.requests = 0

    def admit(self, model, input_tokens, max_output_tokens):
        worst = cost_usd(model, input_tokens, max_output_tokens)
        if self.requests >= self.max_requests:
            return False, "request_limit"
        if sum(self.spent.values()) + worst > self.max_usd:
            return False, f"budget（最壞 ${worst:.6f}，已用 ${sum(self.spent.values()):.6f}）"
        return True, "ok"

    def record(self, model, usage):
        self.requests += 1
        self.spent[usage.source] += cost_usd(model, usage.input_tokens, usage.output_tokens)


def show(snapshot, checked):
    models = snapshot["models"]
    print("能力快照:", snapshot["source"])
    print("litellm:", snapshot["environment"]["litellm"], "｜ 官方價目核對日期:", checked["checked_on"])

    print("\n== 一、能力矩陣（有／無／?=價目表沒寫）==")
    header = ["代號", "工具", "schema", "影像", "seed", "logprobs", "n", "輸入上限"]
    print(" | ".join(header))
    for row in capability_matrix(models):
        print(" | ".join(row))

    print("\n== 二、契約測試：同一個請求，換供應商還成立嗎？==")
    request = ["tools", "response_format", "seed"]
    print("請求需要:", request, "＋ 影像輸入")
    for alias, m in models.items():
        missing, unknown = contract_check(m, request, ["supports_vision"])
        verdict = "通過" if not missing and not unknown else ("不通過" if missing else "待確認")
        print(f"{alias:<14} {verdict:<4} 缺: {missing or '-'}  未知: {unknown or '-'}")

    print("\n== 三、token 數也要換算：LiteLLM 用哪個 tokenizer 估 ==")
    sample = "工人有沒有戴安全帽？"
    for alias, m in models.items():
        print(f"{alias:<14} {m['token_counter'][sample]:>3} token  ← {m['token_counter_tokenizer']}")

    print("\n== 四、套件內建價目會過期：LiteLLM vs 官方核對 ==")
    for alias, (lite, official, same) in price_drift(snapshot, checked).items():
        flag = "相同" if same else "不同 ←"
        print(f"{alias:<14} LiteLLM {lite}  官方 {official}  {flag}")
        if not same:
            print(f"{'':<14} 官方: {checked['models'][alias]['note']}")
    priced = {alias: dict(m, usd_per_1m_input=checked["models"][alias]["usd_per_1m_input"],
                          usd_per_1m_output=checked["models"][alias]["usd_per_1m_output"])
              for alias, m in models.items()}

    print("\n== 五、成本（官方價）：30 人 × 50 次 × (輸入 800、輸出 300 token) ==")
    calls, tin, tout = 30 * 50, 800, 300
    costs = {alias: cost_usd(m, tin * calls, tout * calls) for alias, m in priced.items()}
    stale = {alias: cost_usd(m, tin * calls, tout * calls) for alias, m in models.items()}
    cheapest = min(costs.values())
    for alias in models:
        note = "" if abs(costs[alias] - stale[alias]) < 1e-12 else f"（用 LiteLLM 舊價會算成 ${stale[alias]:.4f}）"
        print(f"{alias:<14} 全班 ${costs[alias]:.4f}  ×{costs[alias] / cheapest:.1f} {note}")
    longer = {alias: cost_usd(m, tin * calls, tout * 4 * calls) for alias, m in priced.items()}
    order = lambda d: [a for a, _ in sorted(d.items(), key=lambda kv: kv[1])]
    print("便宜→貴（輸出 300）:", order(costs))
    print("便宜→貴（輸出 1200）:", order(longer))
    print("便宜→貴（輸出 1200，LiteLLM 舊價）:", order({a: cost_usd(m, tin * calls, tout * 4 * calls) for a, m in models.items()}))

    print("\n== 六、usage 的來源 ==")
    for item in snapshot["mock_usage"]:
        print(f"mock 回覆 {item['reply_chars']:>3} 字 → prompt {item['prompt_tokens']}、completion {item['completion_tokens']}")
    try:
        Usage(10, 20, "actual")
    except ValueError as error:
        print("沒有來源的 usage →", error)

    print("\n== 七、預算上限：送出前擋 ==")
    m = priced["claude-haiku"]
    budget = Budget(max_usd=0.01, max_requests=5)
    for i in range(7):
        ok, why = budget.admit(m, 800, 1000)
        if not ok:
            print(f"第 {i + 1} 次 拒絕: {why}")
            break
        budget.record(m, Usage(800, 300, "estimated"))
        print(f"第 {i + 1} 次 放行，累計 ${sum(budget.spent.values()):.6f}（{budget.spent['estimated']:.6f} 為估算）")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost", nargs=3, metavar=("MODEL", "IN", "OUT"), help="算單次費用，例：--cost deepseek-chat 1200 400")
    args = parser.parse_args(argv)
    snapshot, checked = load("provider-snapshot.json"), load("provider-prices-checked.json")
    if args.cost:
        alias, tin, tout = args.cost[0], int(args.cost[1]), int(args.cost[2])
        price = checked["models"][alias]
        print(f"{alias} 輸入 {tin}、輸出 {tout} → ${cost_usd(price, tin, tout):.6f}"
              f"（estimated，官方價 {checked['checked_on']}）")
        return
    show(snapshot, checked)


if __name__ == "__main__":
    main()
