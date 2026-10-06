"""第 24 節：Orchestration 與 LangChain LCEL。

把第 23 節的 RAG 流程（合格過濾 → 檢索 → 上下文 → 生成 → 驗證 → 處置）原封不動地重用，
只換「串接方式」：一版是普通 Python 函式呼叫，一版是 LCEL（Runnable 用 | 串起來、invoke 執行）。
同一組 9 個案例、同一份契約測試跑兩版，比較：輸出、錯誤怎麼傳遞、框架有沒有替你多做什麼。

需要 langchain-core==1.6.6（裝 langgraph 1.2.12 時已一起裝）；不需要模型、不需要 API key。

    python 24_lcel.py
"""
import re
import sys
import types
from pathlib import Path

from langchain_core.runnables import RunnableLambda

HERE = Path(__file__).resolve().parent


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


L23 = lesson("23")


# ---------------------------------------------------------------- 共用的步驟（輸入、輸出都是 dict）
# 每一步只做一件事，明確寫出「讀哪些欄位、加哪些欄位」：這就是契約。

def step_allowed(state):
    """讀 case、corpus；加 allowed。"""
    case = state["case"]
    return {**state, "allowed": L23.eligible(state["corpus"], case["category_id"]) | case.get("extra_allowed", set())}


def step_retrieve(state):
    """讀 index、case、allowed；加 ranked。空範圍不檢索。"""
    if not state["allowed"]:
        return {**state, "ranked": None}
    return {**state, "ranked": state["index"].search(state["case"]["query"], state["allowed"])}


def step_context(state):
    """讀 ranked；加 picked、used、block。"""
    if state["ranked"] is None:
        return {**state, "picked": [], "used": 0, "block": ""}
    picked, used, block = L23.build_context(state["index"], state["ranked"])
    return {**state, "picked": picked, "used": used, "block": block}


def step_generate(state):
    """讀 generator、case、block；加 answer。合格範圍為空時不呼叫生成。"""
    if state["ranked"] is None:
        return {**state, "answer": None}
    case = state["case"]
    return {**state, "answer": state["generator"].generate(case["case_id"], case["query"], state["block"])}


def step_disposition(state):
    """讀 answer、picked、used；回傳與第 23 節 handle() 相同形狀的結果。"""
    if state["answer"] is None:
        return {"處置": "合格範圍為空 → 轉人工（不呼叫生成）", "上下文": []}
    verdict = L23.verify(state["answer"], state["index"], set(state["picked"]))
    if verdict != "通過":
        return {"處置": f"驗證失敗 → 不送出（{verdict}）", "上下文": state["picked"], "字數": state["used"]}
    return {"處置": L23.DISPOSITION[state["answer"]["status"]], "上下文": state["picked"], "字數": state["used"]}


STEPS = (step_allowed, step_retrieve, step_context, step_generate, step_disposition)


# ---------------------------------------------------------------- 兩種串接

def native(state):
    """純 Python：依序呼叫。檢索故障在這裡攔，轉成處置。"""
    try:
        for step in STEPS:
            state = step(state)
        return state
    except ConnectionError as error:
        return {"處置": f"檢索故障（{error}）→ 稍後重試，不可回答查無資料", "上下文": []}


def retrieval_failed(state_with_error):
    error = state_with_error["exception"]
    return {"處置": f"檢索故障（{error}）→ 稍後重試，不可回答查無資料", "上下文": []}


def build_chain(steps=STEPS, catch=True):
    chain = RunnableLambda(steps[0])
    for step in steps[1:]:
        chain = chain | RunnableLambda(step)
    if catch:                         # 只攔 ConnectionError；其他錯誤照樣往外丟
        chain = chain.with_fallbacks([RunnableLambda(retrieval_failed)],
                                     exceptions_to_handle=(ConnectionError,), exception_key="exception")
    return chain


# ---------------------------------------------------------------- 案例與契約測試

def setup():
    corpus, qrels = L23.load("sop_corpus.json"), L23.load("search_cases.json")
    corpus = dict(corpus, documents=corpus["documents"] + L23.EXTRA_DOCS)
    index = L23.LexicalIndex(corpus["chunks"] + L23.EXTRA_CHUNKS)
    by_id = {c["case_id"]: c for c in qrels["cases"]}
    states = []
    for case_id, label, qid, extra in L23.CASES:
        case = dict(by_id[qid], case_id=case_id, extra_allowed=extra or set())
        use = L23.BrokenIndex() if case_id == "E2" else index
        states.append((label, {"case": case, "corpus": corpus, "index": use}))
    return states


def run_all(runner):
    """runner(state) → 結果；每次都換一個新的生成器，才能數呼叫次數。"""
    generator = L23.ScriptedGenerator(L23.SCRIPT)
    out = {}
    for label, state in setup():
        out[state["case"]["case_id"]] = runner(dict(state, generator=generator))
    return out, generator.calls


def contract(results, calls):
    """與第 23 節講義相同的不變量：任何串接方式都要通過。"""
    checks = {
        "A1 送出且附引用": results["A1"]["處置"] == "回覆（附引用）",
        "A2、A3、I1 不送出": all(results[c]["處置"].startswith("驗證失敗") for c in ("A2", "A3", "I1")),
        "N2 照第 23 節送出（驗證器的已知盲點）": results["N2"]["處置"] == "回覆（附引用）",
        "N1 轉人工": results["N1"]["處置"].startswith("查無答案"),
        "C1 列出雙方": results["C1"]["處置"].startswith("資料衝突"),
        "E1 不呼叫生成": results["E1"]["處置"].startswith("合格範圍為空"),
        "E2 不說查無資料": "不可回答查無資料" in results["E2"]["處置"],
        "生成只被呼叫 7 次": calls == 7,
    }
    return checks


def main():
    reference = L23.handle
    print("同一組步驟（第 23 節的函式），三種跑法；9 個案例、同一份契約")
    print()
    rows = {}
    for name, runner in (
        ("第 23 節 handle()", lambda s: reference(s["case"], s["corpus"], s["index"], s["generator"])),
        ("純 Python 串接", native),
        ("LCEL chain", build_chain().invoke),
    ):
        results, calls = run_all(runner)
        checks = contract(results, calls)
        rows[name] = results
        print(f"{name:<16} 契約 {sum(checks.values())}/{len(checks)} 通過｜生成呼叫 {calls} 次")
    print(f"三種輸出逐案相同：{'是' if rows['純 Python 串接'] == rows['LCEL chain'] == rows['第 23 節 handle()'] else '否'}")

    print("\n== 一、LCEL 物件長什麼樣 ==")
    chain = build_chain()
    print(f"  型別：{type(chain).__name__}；裡面包的是 {type(chain.runnable).__name__}，"
          f"共 {len(chain.runnable.steps)} 步：" + " | ".join(s.name for s in chain.runnable.steps))
    print("  `|` 只是把函式依序接起來；每一步拿到的就是上一步回傳的 dict")

    print("\n== 二、錯誤怎麼傳遞 ==")
    no_catch = build_chain(catch=False)
    E2 = dict(next(s for _, s in setup() if s["case"]["case_id"] == "E2"), generator=L23.ScriptedGenerator(L23.SCRIPT))
    try:
        no_catch.invoke(E2)
        print("  沒有 fallback：沒有錯誤？")
    except ConnectionError as error:
        print(f"  沒有 fallback：ConnectionError 直接丟出（{error}）")
    print(f"  有 fallback（只攔 ConnectionError）：{build_chain().invoke(E2)['處置']}")
    A1 = dict(next(s for _, s in setup() if s["case"]["case_id"] == "A1"), generator=L23.ScriptedGenerator(L23.SCRIPT))
    broken = {k: v for k, v in A1.items() if k != "corpus"}
    for name, runner in (("純 Python", native), ("LCEL", build_chain().invoke)):
        try:
            runner(broken)
        except KeyError as error:
            print(f"  {name}：少了 corpus 欄位 → KeyError {error}（fallback 不攔，照樣丟出）")

    print("\n== 三、框架不會替你補權限 ==")
    def allow_everything(state):
        return {**state, "allowed": {c["chunk_id"] for c in state["corpus"]["chunks"]} | {"X01", "X02"}}

    script = dict(L23.SCRIPT, E1={"status": "unknown", "claims": []})     # E1 原本不會呼叫生成，劇本裡沒有

    def runner(state):
        return build_chain((allow_everything,) + STEPS[1:]).invoke(dict(state, generator=generator))

    generator = L23.ScriptedGenerator(script)
    results = {state["case"]["case_id"]: runner(state) for _, state in setup()}
    calls = generator.calls
    failed = [k for k, v in contract(results, calls).items() if not v]
    print(f"  把「合格過濾」換成全部放行：chain 照樣執行、沒有任何錯誤；契約失敗 {len(failed)} 項：" + "、".join(failed))
    changed = [k for k in results if results[k] != rows["LCEL chain"][k]]
    print(f"  結果改變的案例：{'、'.join(changed)}")
    for k in changed:
        print(f"    {k}：{rows['LCEL chain'][k]['上下文'] or '—'} → {results[k]['上下文']}｜{results[k]['處置']}")
    print("  X01、X02 是未審的同學筆記（X02 藏了指令）：原本只有 C1、I1 刻意放行，現在 5 個案例的上下文都混進來")
    print("  E1 的類別 999 不存在：原本不該檢索，現在檢索到別的產品的段落；換成會亂答的模型，就可能被送出")

    print("\n== 四、batch ==")
    states = [dict(s, generator=L23.ScriptedGenerator(L23.SCRIPT)) for _, s in setup()]
    batch = build_chain().batch(states, config={"max_concurrency": 1})
    print(f"  batch 9 案，與逐一 invoke 相同：{'是' if batch == list(rows['LCEL chain'].values()) else '否'}")
    print("  batch 預設會平行執行；生成器有狀態（計數）、檢索可能有副作用時，要確認可以平行，本節設 max_concurrency=1")

if __name__ == "__main__":
    main()
