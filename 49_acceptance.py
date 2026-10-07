"""第 49 節：全課增量驗收。

不做新功能，只把前面 48 節的產物整理成「能回答問題的證據」：
  一、來源 manifest：每節的程式、證據等級、用到的資料檔與雜湊。
  二、代表性重跑：挑幾節，**現在**重跑一次，與講義凍結的輸出逐字比對；缺套件就標「未重跑」，不算通過。
  三、品質／延遲／成本表：直接從示範機紀錄讀數字，每一列都標來源與證據等級。
  四、限制清單：各講義「待驗證」與「沒有測」的項目。

    python 49_acceptance.py             # 全部（約 1 分鐘；第 31、39 節需要 langgraph、mcp）
    python 49_acceptance.py --quick     # 只重跑只需要標準函式庫的節次
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
# 代表性重跑：(節次, 需要的套件, 為什麼挑它)
RERUN = (
    ("07", (), "評測集與指標：後面所有比較的基準"),
    ("23", (), "引用與拒答"),
    ("25", (), "工具權限與副作用"),
    ("28", (), "重試與冪等"),
    ("31", ("langgraph",), "第一個整合里程碑"),
    ("39", ("mcp",), "MCP 增量：換掉工具層，行為不變"),
    ("40", (), "微調前的決策（重播示範機模型輸出）"),
    ("43", (), "adapter 前後評估（重播示範機訓練紀錄）"),
    ("47", (), "級聯路由的品質與成本"),
)


def find(nn):
    """回傳 (講義路徑或 None, 程式路徑, 程式原始碼)。先找 NN_*.py，再找講義裡的 demo 區塊。"""
    lecture = next(iter(sorted(HERE.glob(f"{nn}-*.md"))), None)
    for path in sorted(HERE.glob(f"{nn}_*.py")):
        return lecture, path, path.read_text(encoding="utf-8")
    if lecture:
        found = DEMO.findall(lecture.read_text(encoding="utf-8"))
        if found:
            return lecture, lecture, found[0][1]
    return lecture, None, None


def lesson(nn):
    _, path, source = find(nn)
    if source is None:
        raise FileNotFoundError(f"找不到第 {nn} 節的程式")
    module = types.ModuleType(f"lesson{nn}")
    module.__file__ = str(path)
    exec(compile(source, path.name, "exec"), module.__dict__)
    return module


def find_data(name):
    """同第 38 節：資料可能在 data/，也可能在 MariaDB 課的資料夾（兩課共用 SOP 語料）。"""
    for base in (HERE / "data", HERE, HERE.parent / "mariadb" / "data"):
        if (base / name).exists():
            return base / name
    return None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path else "缺檔"


def evidence(text):
    """從「教師紀錄」的證據等級那一行，找出提到哪幾個等級。"""
    teacher = re.search(r"^## 教師(?:紀錄|實測).*", text, re.M)            # 只看教師紀錄：Workshop 也會出現「證據等級」
    line = re.search(r"證據等級[：:](.*(?:\n  .*)*)", text[teacher.start():] if teacher else text)
    if not line:
        return ["（講義沒有標）"]
    words = line.group(1).replace("構造", "合成").replace("套件快照", "fixtures")
    found = [level for level in LEVELS if level in words]
    if "合成" in words or "模擬" in words:
        found.append("合成／模擬")
    return found or ["（無法判讀）"]


# ---------------------------------------------------------------- 一、manifest

def manifest():
    rows = []
    for lecture in sorted(HERE.glob("[0-9][0-9]-*.md")):
        if lecture.name.startswith("49-"):               # 不把本節自己算進去：本節的文字會引用「待驗證」等字
            continue
        text = lecture.read_text(encoding="utf-8")
        demo = DEMO.findall(text)
        data = sorted(set(re.findall(r"data/([\w.-]+\.json)", text)))
        rows.append({
            "nn": lecture.name[:2], "lecture": lecture.name,
            "demo": demo[0][0] if demo else "—", "evidence": evidence(text),
            "data": {name: sha(find_data(name)) for name in data},
            "pending": len(re.findall(r"待驗證", text)),
        })
    return rows


# ---------------------------------------------------------------- 二、代表性重跑

def rerun(nn, needs):
    missing = [p for p in needs if importlib.util.find_spec(p) is None]
    if missing:
        return {"status": "未重跑", "detail": "缺套件 " + "、".join(missing)}
    lecture, _, source = find(nn)
    if source is None:
        return {"status": "未重跑", "detail": "找不到程式"}
    frozen = FROZEN.search(lecture.read_text(encoding="utf-8")) if lecture else None
    if not frozen:
        return {"status": "未比對", "detail": "講義沒有凍結輸出"}
    module = lesson(nn)
    buffer, started = io.StringIO(), time.perf_counter()
    argv, sys.argv = sys.argv, [f"{nn}_demo.py"]
    try:
        with contextlib.redirect_stdout(buffer):
            takes_args = module.main.__code__.co_argcount > 0
            module.main([]) if takes_args else module.main()
    except Exception as error:                                # 失敗也要記下來，不中斷整份報告
        return {"status": "失敗", "detail": f"{type(error).__name__}: {str(error)[:80]}"}
    finally:
        sys.argv = argv
    seconds = time.perf_counter() - started
    got, want = buffer.getvalue().rstrip("\n").splitlines(), frozen.group(2).splitlines()
    if got == want:
        return {"status": "相同", "detail": f"{len(want)} 行逐字相同", "seconds": seconds, "basis": frozen.group(1)}
    first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    return {"status": "不同", "detail": f"第一個不同在第 {first + 1} 行（重跑 {len(got)} 行、講義 {len(want)} 行）",
            "seconds": seconds, "basis": frozen.group(1)}


# ---------------------------------------------------------------- 三、品質／延遲／成本

def metrics():
    rows = []
    data = HERE / "data"
    items = json.loads((data / "eval-set.json").read_text(encoding="utf-8"))["items"]
    holdout = [i["id"] for i in items if i["split"] == "holdout"]
    gold = {i["id"]: i["gold"] for i in items}

    l38 = lesson("38")
    adapter = json.loads((data / "43-adapter-record.json").read_text(encoding="utf-8"))
    for name, key in (("base zero-shot", "base"), ("base＋adapter", "lora")):
        right = sum(l38.parse(adapter["outputs"][key][i]) == gold[i] for i in holdout)
        rows.append(("品質", f"holdout 10 筆正確（{name}）", f"{right}/10", "43", "遠端實測（重播）"))

    latency = json.loads((data / "44-latency-record.json").read_text(encoding="utf-8"))
    for size in ("短", "長"):
        runs = latency["runs"][size][1:]
        ttft = statistics.median(r["ttft"] for r in runs)
        rows.append(("延遲", f"TTFT 中位數（{size}，{runs[0]['input_tokens']} token 輸入）",
                     f"{ttft:.2f} 秒", "44", "遠端實測（重播）"))

    quant = json.loads((data / "45-quant-record.json").read_text(encoding="utf-8"))["results"]
    for name in ("float32", "bfloat16"):
        rows.append(("資源", f"峰值記憶體（{name}）", f"{quant[name]['peak_rss_mib']:,} MiB", "45", "遠端實測（重播）"))

    prices = json.loads((data / "provider-prices-checked.json").read_text(encoding="utf-8"))
    model = next(iter(prices["models"].values()))
    tokens_in = latency["runs"]["長"][0]["input_tokens"]
    cost = (tokens_in * model["usd_per_1m_input"] + 32 * model["usd_per_1m_output"]) / 1e6 * 1000
    rows.append(("成本", f"每天 1,000 次「長」輸入、輸出 32 token（{model['official_model']}）", f"US${cost:.3f}",
                 "14、44", f"教師核對價 {prices['checked_on']}"))
    return rows


# ---------------------------------------------------------------- 報告

def main(argv=()):
    quick = "--quick" in argv
    rows = manifest()
    if not rows:
        sys.exit("找不到任何講義（NN-*.md）。本節要比對講義裡的凍結輸出，請在放講義的資料夾執行"
                 "（llm-demo 只有程式，沒有講義：把 HackMD 講義下載成 .md 放進同一個資料夾）")
    print(f"教材資料夾：第 01–48 節中已改版的講義 {len(rows)} 節（新增節次後，本節輸出會跟著變）")

    print("\n== 一、來源 manifest（節次｜程式｜證據等級｜資料檔雜湊前 12 碼） ==")
    for r in rows:
        data = "、".join(f"{k}@{v}" for k, v in r["data"].items()) or "—"
        print(f"  {r['nn']}｜{r['demo']}｜{'＋'.join(r['evidence'])}｜{data}")
    unlabeled = [r["nn"] for r in rows if r["evidence"][0].startswith("（")]
    missing = sorted({k for r in rows for k, v in r["data"].items() if v == "缺檔"})
    print(f"  講義沒有標證據等級：{'、'.join(unlabeled) or '無'}")
    print(f"  講義提到但資料夾裡沒有的資料檔：{'、'.join(missing) or '無'}")

    print("\n== 二、代表性重跑（現在執行，與講義凍結輸出逐字比對） ==")
    counts = {}
    for nn, needs, why in RERUN:
        if quick and needs:
            result = {"status": "未重跑", "detail": "--quick 略過需要第三方套件的節次"}
        else:
            result = rerun(nn, needs)
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        basis = f"｜凍結依據：{result['basis']}" if "basis" in result else ""
        print(f"  {nn}｜{why}｜{result['status']}｜{result['detail']}{basis}")
    print("  合計：" + "、".join(f"{k} {v}" for k, v in counts.items()))
    print("  「相同」只代表這次重跑與講義一致；第 40、43 節是重播示範機紀錄，不是即時推論")

    print("\n== 三、品質／延遲／資源／成本（每列都標來源節次與證據等級） ==")
    for kind, what, value, source, level in metrics():
        print(f"  {kind}｜{what}｜{value}｜第 {source} 節｜{level}")

    print("\n== 四、限制清單（講義中「待驗證」出現次數，前 8 名） ==")
    for r in sorted(rows, key=lambda r: -r["pending"])[:8]:
        print(f"  {r['nn']}｜{r['lecture']}｜{r['pending']} 處")
    print(f"  全部講義合計 {sum(r['pending'] for r in rows)} 處；這些項目不能寫進「已完成」")
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
