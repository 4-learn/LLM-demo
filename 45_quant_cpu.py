"""第 45 節：量化與 CPU 實驗。

同一個 Qwen2.5-0.5B（同 revision、同 tokenizer、同提示、greedy），只改精度：
  float32（基準）、bfloat16（16 位元浮點）、
  int8 動態量化（torch 內建；權重存 int8，激活值在執行時也量化成 int8）——除 lm_head 外的全部 Linear，
  以及同樣做法但跳過 24 個 mlp.down_proj（看完激活值統計後才決定，見第四段）。
每個條件在**獨立子程序**裡跑，才量得到各自的峰值記憶體；逾時或失敗就記下來，不重試。
量：峰值 RSS、載入秒數、TTFT 與 decode 速率（第 44 節的「中」輸入）、第 07 節 25 筆 zero-shot 正確數。

    python 45_quant_cpu.py          # 重播示範機 4-learn 的紀錄（不需要模型）
    python 45_quant_cpu.py --live   # 本機實測（需要 torch、transformers、Qwen 快取）；加 --save 覆寫紀錄
"""
import json
import re
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE / "data" / "45-quant-record.json"
CONDITIONS = ("float32", "bfloat16", "int8-dynamic", "int8-skip-down")
THREADS = 2
LATENCY_REPEATS = 3
TIMEOUT_SECONDS = 600     # 單一條件的上限：超過就停，記為逾時


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


# ---------------------------------------------------------------- 子程序：一個條件

def worker(condition):
    import resource
    import statistics
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    l38, l44 = lesson("38"), lesson("44")
    torch.set_num_threads(THREADS)
    path = snapshot_download(l38.MODEL_ID, revision=l38.REVISION, local_files_only=True)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(path)
    dtype = torch.bfloat16 if condition == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    stats = {}
    if condition == "float32":                          # 量化前先看：每個 Linear 收到的輸入有多大
        def hook(name):
            def record(module, inputs, output):
                a = inputs[0].detach().abs()
                old = stats.get(name, (0.0, 0.0))
                stats[name] = (max(old[0], a.max().item()), max(old[1], a.median().item()))
            return record
        hooks = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if type(m) is torch.nn.Linear]
    if condition.startswith("int8"):
        skip = ("lm_head",) if condition == "int8-dynamic" else ("lm_head", "down_proj")
        names = {n for n, m in model.named_modules() if type(m) is torch.nn.Linear and not n.endswith(skip)}
        torch.ao.quantization.quantize_dynamic(model, names, dtype=torch.qint8, inplace=True)
    load_seconds = time.perf_counter() - started
    for name in ("temperature", "top_p", "top_k"):
        setattr(model.generation_config, name, None)
    linear = sum(1 for m in model.modules() if type(m) is torch.nn.Linear)

    def generate(messages, max_new_tokens, streamer=None, fixed=False):
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)
        extra = {"min_new_tokens": max_new_tokens} if fixed else {}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer, **extra)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    items = json.loads(l38.find_data("eval-set.json").read_text(encoding="utf-8"))["items"]
    labels = {}
    for item in items:
        messages = [{"role": "system", "content": l38.TASK}, {"role": "user", "content": f"通報：{item['text']}\n標籤："}]
        labels[item["id"]] = generate(messages, l38.MAX_NEW_TOKENS)

    if condition == "float32":
        for h in hooks:
            h.remove()
    prompt = l44.build_prompt(l44.CONTEXT_COPIES["中"])
    timings = []
    for _ in range(LATENCY_REPEATS + 1):
        stamps = l44.Stamps()
        begin = time.perf_counter()
        generate(prompt, l44.OUTPUT_TOKENS, stamps, fixed=True)
        gaps = [b - a for a, b in zip(stamps.times, stamps.times[1:])]
        timings.append({"ttft": round(stamps.times[0] - begin, 4), "itl": round(statistics.median(gaps), 4)})
    answer = generate([{"role": "user", "content": "用一句話說明：工人進入工地前要做什麼安全檢查？"}], 40)
    print(json.dumps({
        "condition": condition, "load_seconds": round(load_seconds, 2), "plain_linear_left": linear,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
        "rss_end_mib": int(Path("/proc/self/statm").read_text().split()[1]) * resource.getpagesize() // 2 ** 20,
        "labels": labels, "timings": timings, "answer": answer,
        "activation_top": sorted(([n, round(mx, 1), round(md, 3)] for n, (mx, md) in stats.items()),
                                 key=lambda r: -r[1])[:6],
    }, ensure_ascii=False))


def version(metadata, package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "未安裝"


def measure():
    import importlib.metadata as metadata
    import platform
    results = {}
    for condition in CONDITIONS:
        try:
            done = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", condition],
                                  capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            results[condition] = {"status": f"逾時（超過 {TIMEOUT_SECONDS} 秒）"}
            continue
        if done.returncode != 0:
            tail = done.stderr.strip().splitlines()[-1:] or [f"exit {done.returncode}"]
            results[condition] = {"status": f"失敗：{tail[0][:120]}"}
            continue
        results[condition] = dict(json.loads(done.stdout.strip().splitlines()[-1]), status="完成")
    return {
        "note": "第 45 節重播用：示範機實測。每個條件在獨立子程序執行；不同機器、負載、torch 版本都可能不同。",
        "threads": THREADS, "results": results,
        "versions": {p: version(metadata, p) for p in ("torch", "transformers")},
        "recorded_on": f"{platform.node()}｜{platform.machine()}｜Python {platform.python_version()}",
    }


# ---------------------------------------------------------------- 報告

def report(record, source):
    import statistics
    l38 = lesson("38")
    items = json.loads(l38.find_data("eval-set.json").read_text(encoding="utf-8"))["items"]
    gold = {i["id"]: i["gold"] for i in items}
    results = record["results"]
    print(f"模型 {l38.MODEL_ID}@{l38.REVISION[:7]}｜CPU {record['threads']} 執行緒｜greedy｜同提示、同 parse")
    print(f"來源：{source}｜{record['recorded_on']}｜" + "、".join(f"{k} {v}" for k, v in record["versions"].items()))

    print("\n== 一、同一個模型，只改精度 ==")
    print("條件 | 狀態 | 峰值 RSS | 結束時 RSS | 載入 | TTFT 中位數 | ITL 中位數 | 25 筆正確 | 解析失敗")
    base = results.get("float32", {})
    for name in CONDITIONS:
        r = results.get(name, {"status": "未執行"})
        if r["status"] != "完成":
            print(f"{name} | {r['status']} | — | — | — | — | — | — | —")
            continue
        rest = r["timings"][1:]
        ttft = statistics.median(t["ttft"] for t in rest)
        itl = statistics.median(t["itl"] for t in rest)
        parsed = {i: l38.parse(t) for i, t in r["labels"].items()}
        right = sum(parsed[i] == gold[i] for i in gold)
        fails = sum(v is None for v in parsed.values())
        print(f"{name} | 完成 | {r['peak_rss_mib']:,} MiB | {r['rss_end_mib']:,} MiB | {r['load_seconds']:.1f} 秒 | {ttft:.2f} 秒 | "
              f"{itl * 1000:.0f} ms | {right} | {fails}")

    if base.get("status") == "完成":
        print("\n== 二、跟 float32 比：逐筆標籤有幾筆不同？ ==")
        for name in CONDITIONS[1:]:
            r = results.get(name, {})
            if r.get("status") != "完成":
                print(f"  {name}：沒有結果，不能比")
                continue
            diff = [i for i in gold if l38.parse(r["labels"][i]) != l38.parse(base["labels"][i])]
            detail = "、".join(f"{i}（{l38.parse(base['labels'][i])}→{l38.parse(r['labels'][i])}）" for i in diff)
            print(f"  {name}：{len(diff)} 筆不同{'：' + detail if diff else ''}")

        print("\n== 三、開放式回答（40 token）==")
        for name in CONDITIONS:
            r = results.get(name, {})
            if r.get("status") == "完成":
                print(f"  {name:<13}{r['answer']}")

        print("\n== 四、為什麼全部量化會壞：每層收到的輸入有多大（float32，25 筆分類時） ==")
        print("  層 | 最大絕對值 | 中位數")
        for name, peak, median in base["activation_top"]:
            print(f"  {name} | {peak:,.1f} | {median}")
        print("  動態量化用「最大值」決定每個 int8 刻度：最大值 1,800 時，刻度約 14，中位數 0.05 的值全部變成 0")
        for name in CONDITIONS[2:]:
            r = results.get(name, {})
            if r.get("status") == "完成":
                print(f"  {name}：仍是浮點 Linear 的層數 {r['plain_linear_left']}／{base['plain_linear_left']}")

def main(argv=()):
    if "--worker" in argv:
        worker(argv[argv.index("--worker") + 1])
        return None
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
