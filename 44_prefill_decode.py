"""第 44 節：Prefill、Decode 與 Token 成本。

在 CPU 上用 Qwen2.5-0.5B 量每個 token 出現的時間，把一次生成拆成：
  prefill（處理整段輸入，到第一個 token 出現：TTFT）與 decode（之後每個 token：inter-token latency）。
固定輸出 32 token，只改輸入長度；冷載入、第一次（暖機）與之後的重複分開記。
最後用第 14 節教師核對的價目，算同樣 token 數在雲端的計費。

    python 44_prefill_decode.py          # 重播示範機 4-learn 的量測紀錄（不需要模型）
    python 44_prefill_decode.py --live   # 本機實測（需要 torch、transformers、Qwen 快取）；加 --save 覆寫紀錄
"""
import json
import re
import statistics
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE / "data" / "44-latency-record.json"
OUTPUT_TOKENS = 32
REPEATS = 3
THREADS = 2
CONTEXT_COPIES = {"短": 0, "中": 1, "長": 4}     # 在問題前面放幾份「通報紀錄」當背景


def lesson(nn):
    """同第 31 節：先找 NN_*.py，再找講義 NN-*.md；讀原始碼後 exec。"""
    found = [(p, p.read_text(encoding="utf-8")) for p in sorted(HERE.glob(f"{nn}_*.py"))]
    for md in sorted(HERE.glob(f"{nn}-*.md")):
        found += [(md, src) for _, src in re.findall(r"<!-- demo: ([\w.]+) -->\n```python\n(.*?)\n```",
                                                     md.read_text(encoding="utf-8"), re.S)]
    if not found:
        raise FileNotFoundError(f"找不到第 {nn} 節的程式（{nn}_*.py 或 {nn}-*.md）")
    path, source = found[0]
    module = types.ModuleType(f"lesson{nn}")
    module.__file__ = str(path)
    exec(compile(source, path.name, "exec"), module.__dict__)
    return module


def find_data(name):
    for base in (HERE / "data", HERE.parent / "data"):
        if (base / name).exists():
            return base / name
    raise FileNotFoundError(f"找不到 data/{name}")


def build_prompt(copies):
    items = json.loads(find_data("eval-set.json").read_text(encoding="utf-8"))["items"]
    log = "\n".join(f"{i['id']}：{i['text']}" for i in items)
    context = "\n\n".join(f"第 {n + 1} 份通報紀錄：\n{log}" for n in range(copies))
    question = "請用三句話說明工地巡檢時，安全帽通報要注意什麼。"
    return [{"role": "user", "content": f"{context}\n\n{question}" if context else question}]


class Stamps:
    """transformers 的 streamer 介面：generate 先 put 一次整段輸入，之後每產生一個 token put 一次。"""

    def __init__(self):
        self.times, self.first = [], True

    def put(self, value):
        if self.first:
            self.first = False
            return
        self.times.append(time.perf_counter())

    def end(self):
        pass


# ---------------------------------------------------------------- 實測（--live）

def measure():
    import importlib.metadata as metadata
    import platform
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    smoke = lesson("06")
    torch.set_num_threads(THREADS)
    path = snapshot_download(smoke.MODEL_ID, revision=smoke.REVISION, local_files_only=True)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    load_seconds = time.perf_counter() - started
    for name in ("temperature", "top_p", "top_k"):
        setattr(model.generation_config, name, None)

    def once(messages):
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)
        stamps = Stamps()
        begin = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=OUTPUT_TOKENS, min_new_tokens=OUTPUT_TOKENS,
                                 do_sample=False, streamer=stamps)
        end = time.perf_counter()
        produced = out.shape[1] - inputs["input_ids"].shape[1]
        return {"input_tokens": int(inputs["input_ids"].shape[1]), "output_tokens": int(produced),
                "ttft": round(stamps.times[0] - begin, 4),
                "gaps": [round(b - a, 4) for a, b in zip(stamps.times, stamps.times[1:])],
                "total": round(end - begin, 4)}

    runs = {}
    for label, copies in CONTEXT_COPIES.items():
        messages = build_prompt(copies)
        runs[label] = [once(messages) for _ in range(REPEATS + 1)]      # 第 0 次是這個長度的第一次
    return {
        "note": "第 44 節重播用：示範機實測的每個 token 時間。不同機器、負載、執行緒數都會不同，不要當成穩定效能。",
        "model": smoke.MODEL_ID, "revision": smoke.REVISION, "threads": THREADS,
        "load_seconds": round(load_seconds, 2), "runs": runs,
        "versions": {p: metadata.version(p) for p in ("torch", "transformers")},
        "recorded_on": f"{platform.node()}｜{platform.machine()}｜Python {platform.python_version()}",
    }


# ---------------------------------------------------------------- 報告

def summarize(run):
    gaps = run["gaps"]
    decode = sum(gaps)
    return {"ttft": run["ttft"], "itl": statistics.median(gaps),
            "decode_tps": len(gaps) / decode, "total": run["total"],
            "prefill_tps": run["input_tokens"] / run["ttft"]}


def report(record, source):
    print(f"模型 {record['model']}@{record['revision'][:7]}｜CPU {record['threads']} 執行緒｜float32｜greedy｜"
          f"輸出固定 {OUTPUT_TOKENS} token")
    print(f"來源：{source}｜{record['recorded_on']}｜" + "、".join(f"{k} {v}" for k, v in record["versions"].items()))
    print(f"冷載入（讀權重到記憶體）：{record['load_seconds']:.1f} 秒——不算在下面任何一次生成裡")

    print("\n== 一、只改輸入長度（每種長度：第一次另列，之後 3 次取中位數） ==")
    print("輸入 | 輸入 token | 第一次 TTFT | TTFT 中位數 | prefill token/s | ITL 中位數 | decode token/s | 端到端中位數")
    rows = {}
    for label, runs in record["runs"].items():
        first, rest = summarize(runs[0]), [summarize(r) for r in runs[1:]]
        med = {k: statistics.median(r[k] for r in rest) for k in rest[0]}
        rows[label] = (runs[0]["input_tokens"], med)
        print(f"{label} | {runs[0]['input_tokens']} | {first['ttft']:.2f} 秒 | {med['ttft']:.2f} 秒 | "
              f"{med['prefill_tps']:.0f} | {med['itl'] * 1000:.0f} ms | {med['decode_tps']:.1f} | {med['total']:.2f} 秒")

    short, long_ = rows["短"], rows["長"]
    print("\n== 二、輸入變長，哪一段變慢？ ==")
    print(f"  輸入 {short[0]} → {long_[0]} token（×{long_[0] / short[0]:.0f}）："
          f"TTFT ×{long_[1]['ttft'] / short[1]['ttft']:.1f}，ITL ×{long_[1]['itl'] / short[1]['itl']:.2f}，"
          f"端到端 ×{long_[1]['total'] / short[1]['total']:.1f}")
    spread = [summarize(r)["ttft"] for r in record["runs"]["長"][1:]]
    print(f"  「長」的 3 次 TTFT：{'、'.join(f'{t:.2f}' for t in spread)} 秒（同一台機器、同一個輸入也會不同）")

    print("\n== 三、同樣的 token 數，雲端怎麼計費（第 14 節教師核對價，美元） ==")
    prices = json.loads(find_data("provider-prices-checked.json").read_text(encoding="utf-8"))
    print(f"  價目核對日 {prices['checked_on']}；以「長」輸入 {long_[0]} token、輸出 {OUTPUT_TOKENS} token、每天 1,000 次計")
    print("  模型 | 每 1M 輸入 | 每 1M 輸出 | 每天輸入費 | 每天輸出費 | 輸入費占比")
    for name, m in prices["models"].items():
        cin = long_[0] * 1000 * m["usd_per_1m_input"] / 1e6
        cout = OUTPUT_TOKENS * 1000 * m["usd_per_1m_output"] / 1e6
        print(f"  {m['official_model']} | {m['usd_per_1m_input']} | {m['usd_per_1m_output']} | "
              f"{cin:.3f} | {cout:.3f} | {cin / (cin + cout):.0%}")
    print("  本機推論不按 token 計費，但輸入越長 TTFT 越久：時間就是本機的成本。")


def main(argv=()):
    if "--live" in argv:
        record = measure()
        if "--save" in argv:
            RECORD.parent.mkdir(exist_ok=True)
            RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        source = "本機實測"
    else:
        record = json.loads(RECORD.read_text(encoding="utf-8"))
        source = "重播示範機紀錄"
    report(record, source)
    return record


if __name__ == "__main__":
    main(sys.argv[1:])
