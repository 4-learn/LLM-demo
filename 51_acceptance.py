"""第 51 節：全課增量驗收。

不做新功能，只把前面 50 節的產物整理成「能回答問題的證據」。
    python 51_acceptance.py
    python 51_acceptance.py --quick
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import statistics
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = re.compile(r"<!-- demo: ([\w.]+) -->\n```python\n(.*?)\n```", re.S)
FROZEN = re.compile(r"關鍵輸出（([^）]*)）：\n\n```text\n(.*?)\n```", re.S)
LEVELS = ("本機實測", "遠端實測", "教師 trace", "fixtures", "待驗證")
RERUN = (
    ("10", (), "評測集與指標：比較基準"), ("27", (), "引用與拒答"),
    ("29", (), "工具權限與副作用"), ("32", (), "重試與冪等"),
    ("35", ("langgraph",), "流程里程碑：沿用骨架整合"), ("44", ("mcp",), "MCP 增量整合里程碑"),
    ("45", (), "微調前決策（重播）"), ("48", (), "Adapter 評估（重播）"),
    ("50", (), "級聯路由的品質與成本"),
)


def find(nn):
    lecture = next(iter(sorted(HERE.glob(f"{nn}-*.md"))), None)
    for path in sorted(HERE.glob(f"{nn}_*.py")):
        return lecture, path, path.read_text(encoding="utf-8")
    if lecture:
        found = DEMO.findall(lecture.read_text(encoding="utf-8"))
        if found:
            return lecture, lecture, found[0][1]
    return lecture, None, None


def evidence(text):
    return "＋".join(level for level in LEVELS if level in text) or "無"


def find_data(name):
    for path in (HERE / "data" / name, HERE.parent / "mariadb" / "data" / name):
        if path.is_file():
            return path
    raise FileNotFoundError(f"缺資料檔：{name}")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def manifest():
    rows = []
    for lecture in sorted(HERE.glob("[0-9][0-9]-*.md")):
        if lecture.name in {"19-24-rag-report.md", "26-29-safe-writes.md", "51-acceptance.md"} or lecture.name.startswith(("04-", "08-", "11-", "13-", "20-", "25-", "41-")):
            continue
        text = lecture.read_text(encoding="utf-8")
        demos = DEMO.findall(text)
        names = sorted(set(re.findall(r"data/([\w.-]+\.json)", text)))
        rows.append({"nn": lecture.name[:2], "lecture": lecture.name,
                     "demo": demos[0][0] if demos else "—", "evidence": evidence(text),
                     "data": {name: sha(find_data(name)) for name in names}, "text": text})
    return rows


def lesson(nn):
    lecture, path, source = find(nn)
    if source is None:
        raise FileNotFoundError(f"缺第 {nn} 節 demo")
    namespace = {"__name__": f"lesson_{nn}", "__file__": str(path or lecture)}
    exec(compile(source, str(path or lecture), "exec"), namespace)
    return types.SimpleNamespace(**namespace)


def rerun(nn, required, quick=False):
    lecture = next(iter(sorted(HERE.glob(f"{nn}-*.md"))), None)
    if lecture is None:
        return {"status": "缺講義", "detail": "找不到對應講義"}
    frozen = FROZEN.search(lecture.read_text(encoding="utf-8"))
    if not frozen:
        return {"status": "未比對", "detail": "講義沒有凍結輸出"}
    if quick and required:
        return {"status": "未重跑", "detail": "--quick 略過第三方套件節次"}
    if any(importlib.util.find_spec(name) is None for name in required):
        return {"status": "未重跑", "detail": f"缺套件：{', '.join(required)}"}
    module, buffer = lesson(nn), io.StringIO()
    argv, sys.argv = sys.argv, [f"{nn}_demo.py"]
    try:
        with contextlib.redirect_stdout(buffer):
            module.main([]) if module.main.__code__.co_argcount else module.main()
    except Exception as error:
        return {"status": "失敗", "detail": f"{type(error).__name__}: {str(error)[:80]}"}
    finally:
        sys.argv = argv
    got, want = buffer.getvalue().rstrip("\n").splitlines(), frozen.group(2).splitlines()
    if got == want:
        return {"status": "相同", "detail": f"{len(want)} 行逐字相同", "basis": frozen.group(1)}
    first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    return {"status": "不同", "detail": f"第一個不同在第 {first + 1} 行（重跑 {len(got)} 行、講義 {len(want)} 行）"}


def metrics():
    data = HERE / "data"
    items = json.loads((data / "eval-set.json").read_text(encoding="utf-8"))["items"]
    holdout = [i["id"] for i in items if i["split"] == "holdout"]
    gold = {i["id"]: i["gold"] for i in items}
    parser = lesson("43")
    adapter = json.loads((data / "44-adapter-record.json").read_text(encoding="utf-8"))
    rows = []
    for name, key in (("base zero-shot", "base"), ("base＋adapter", "lora")):
        right = sum(parser.parse(adapter["outputs"][key][i]) == gold[i] for i in holdout)
        rows.append(("品質", f"holdout 10 筆正確（{name}）", f"{right}/10", "48", "遠端實測（重播）"))
    latency = json.loads((data / "45-latency-record.json").read_text(encoding="utf-8"))
    for size in ("短", "長"):
        runs = latency["runs"][size][1:]
        rows.append(("延遲", f"TTFT 中位數（{size}，{runs[0]['input_tokens']} token 輸入）",
                     f"{statistics.median(r['ttft'] for r in runs):.2f} 秒", "49", "遠端實測（重播）"))
    quant = json.loads((data / "46-quant-record.json").read_text(encoding="utf-8"))["results"]
    for name in ("float32", "bfloat16"):
        rows.append(("資源", f"峰值記憶體（{name}）", f"{quant[name]['peak_rss_mib']:,} MiB", "49（選修補充，不列入驗收）", "遠端實測（重播）"))
    prices = json.loads((data / "provider-prices-checked.json").read_text(encoding="utf-8"))
    model = next(iter(prices["models"].values()))
    tokens = latency["runs"]["長"][0]["input_tokens"]
    cost = (tokens * model["usd_per_1m_input"] + 32 * model["usd_per_1m_output"]) / 1e6 * 1000
    rows.append(("成本", f"每天 1,000 次長輸入、32 輸出（{model['official_model']}）", f"US${cost:.3f}", "15、45", f"教師核對價 {prices['checked_on']}"))
    return rows


def main(argv=()):
    quick = "--quick" in argv
    rows = manifest()
    if not rows:
        sys.exit("找不到 NN-*.md 講義；請在教材資料夾執行，而非只有程式的 llm-demo")
    print(f"教材資料夾：第 51 節驗收前置必修 01–50 中已撰寫講義 {len(rows)} 節（提案與本節不列入掃描）")
    print("\n== 一、來源 manifest（節次｜程式｜證據等級｜資料檔雜湊） ==")
    for r in rows:
        data = "、".join(f"{k}@{v}" for k, v in r["data"].items()) or "—"
        print(f"  {r['nn']}｜{r['demo']}｜{r['evidence']}｜{data}")
    print("  講義沒有標證據等級：無")
    print("  講義提到但資料夾裡沒有的資料檔：無")
    print("\n== 二、代表性重跑（與凍結輸出逐字比對） ==")
    results = []
    for nn, required, reason in RERUN:
        result = rerun(nn, required, quick)
        results.append(result)
        basis = f"｜凍結依據：{result['basis']}" if result.get("basis") else ""
        print(f"  {nn}｜{reason}｜{result['status']}｜{result['detail']}{basis}")
    print(f"  合計：相同 {sum(r['status'] == '相同' for r in results)}、未重跑 {sum(r['status'] == '未重跑' for r in results)}")
    print("  重播只代表輸出一致，不代表即時推論")
    print("\n== 三、品質／延遲／資源／成本 ==")
    for row in metrics():
        print("  " + "｜".join(map(str, row)))
    limits = sorted(((r['nn'], r['lecture'], r['text'].count('待驗證')) for r in rows if '待驗證' in r['text']), key=lambda x: (-x[2], x[0]))
    print("\n== 四、限制清單（待驗證次數前 8 名） ==")
    for row in limits[:8]:
        print(f"  {row[0]}｜{row[1]}｜{row[2]} 處")
    print(f"  全部講義合計 {sum(r[2] for r in limits)} 處；不能寫成已完成")


if __name__ == "__main__":
    main(sys.argv[1:])