"""第 40 節：Prompt、RAG 與 SFT 的選擇。

在決定要不要微調之前，先把不用訓練的做法做到合理，再看還剩哪些錯：
  1. zero-shot：第 38 節單代理的提示，原封不動（合理的 baseline，不是刻意寫差的）
  2. 固定 few-shot：從 dev 挑 3 筆（每個標籤 1 筆），所有案例共用
  3. 檢索 few-shot（RAG）：用 bge 從 dev 找最像的 3 筆當範例
  4. embedding 分類器：不生成，直接看 dev 裡最像的 3 筆投票

規則：範例只能來自 dev；評 dev 時把案例自己排除（leave-one-out）；holdout 只看一次。
同一個 Qwen2.5-0.5B、greedy、輸出上限 8 token、同一個 parse。

    python 40_prompt_rag_sft.py          # 重播示範機 4-learn 錄下的模型輸出與檢索結果
    python 40_prompt_rag_sft.py --live   # 真的跑 Qwen 與 bge（需要第 06、16 節的模型快取）；加 --save 覆寫錄製檔
"""
import hashlib
import json
import re
import sys
import types
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE / "data" / "40-prompt-rag-traces.json"
BGE_ID = "BAAI/bge-small-zh-v1.5"
BGE_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
K = 3
FIXED_SHOTS = ("A01", "A02", "B01")          # 從 dev 依標籤各挑第一筆；挑法在看結果之前就定好


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


L38 = lesson("38")
TASK, parse = L38.TASK, L38.parse


def shots_messages(shots):
    """few-shot 用對話輪次給範例：user 是通報，assistant 是標籤。"""
    messages = []
    for item in shots:
        messages += [{"role": "user", "content": f"通報：{item['text']}\n標籤："},
                     {"role": "assistant", "content": item["gold"]}]
    return messages


def build_messages(item, shots):
    return ([{"role": "system", "content": TASK}] + shots_messages(shots)
            + [{"role": "user", "content": f"通報：{item['text']}\n標籤："}])


def key(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 模型與檢索：即時或重播

class Live:
    source = "即時"

    def __init__(self, items):
        import numpy as np
        import torch
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(2)
        bge = SentenceTransformer(BGE_ID, revision=BGE_REVISION, device="cpu",
                                  trust_remote_code=False, model_kwargs={"use_safetensors": True})
        vectors = bge.encode([i["text"] for i in items], normalize_embeddings=True, show_progress_bar=False)
        sims = vectors @ vectors.T
        dev = [n for n, i in enumerate(items) if i["split"] == "dev"]
        self.neighbours = {}
        for n, item in enumerate(items):
            pool = [d for d in dev if d != n]                      # leave-one-out：不能檢索到自己
            ranked = sorted(pool, key=lambda d: (-float(sims[n, d]), items[d]["id"]))[:K]
            self.neighbours[item["id"]] = [[items[d]["id"], round(float(sims[n, d]), 4)] for d in ranked]
        del bge, np
        path = snapshot_download(L38.MODEL_ID, revision=L38.REVISION, local_files_only=True)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        for name in ("temperature", "top_p", "top_k"):
            setattr(self.model.generation_config, name, None)
        self.calls = {}

    def neighbours_of(self, item_id):
        return self.neighbours[item_id]

    def __call__(self, messages):
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                    return_tensors="pt", return_dict=True)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=L38.MAX_NEW_TOKENS, do_sample=False)
        new = out[0][inputs["input_ids"].shape[1]:]
        entry = {"text": self.tokenizer.decode(new, skip_special_tokens=True).strip(),
                 "prompt_tokens": int(inputs["input_ids"].shape[1])}
        self.calls[key(messages)] = entry
        return entry


class Replay:
    source = "錄製"

    def __init__(self, path=RECORD):
        self.data = json.loads(path.read_text(encoding="utf-8"))

    def neighbours_of(self, item_id):
        return self.data["neighbours"][item_id]

    def __call__(self, messages):
        found = self.data["calls"].get(key(messages))
        if found is None:
            raise LookupError("錄製檔沒有這個提示（提示、範例或案例改了）；請在有模型的機器用 --live 重錄")
        return found


# ---------------------------------------------------------------- 四種做法

def run_all(model, items):
    by_id = {i["id"]: i for i in items}
    fixed = [by_id[i] for i in FIXED_SHOTS]
    results = {"zero-shot": {}, "固定 few-shot": {}, "檢索 few-shot": {}, "embedding 分類器": {}}
    tokens = Counter()
    for item in items:
        nearest = [by_id[i] for i, _ in model.neighbours_of(item["id"])]
        plans = (("zero-shot", []),
                 ("固定 few-shot", [s for s in fixed if s["id"] != item["id"]]),   # 評 dev 時排除自己
                 ("檢索 few-shot", nearest))
        for name, shots in plans:
            entry = model(build_messages(item, shots))
            results[name][item["id"]] = parse(entry["text"])
            tokens[name] += entry["prompt_tokens"]
        votes = Counter(s["gold"] for s in nearest)
        top = votes.most_common()
        results["embedding 分類器"][item["id"]] = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else nearest[0]["gold"]
    return results, tokens


def report(results, tokens, items, model):
    gold = {i["id"]: i["gold"] for i in items}
    split = {i["id"]: i["split"] for i in items}
    group = {i["id"]: i["group"] for i in items}
    print(f"模型 {L38.MODEL_ID}（greedy，輸出上限 {L38.MAX_NEW_TOKENS} token）｜檢索 {BGE_ID}，k={K}")
    print(f"來源：{'示範機 4-learn 錄製' if model.source == '錄製' else '本機即時推論'}｜範例只取自 dev；評 dev 時排除案例本身")
    print()
    print("做法 | dev 正確（留一） | holdout 正確 | 解析失敗 | 輸入 token/筆")
    for name, rows in results.items():
        dev = sum(rows[i] == gold[i] for i in rows if split[i] == "dev")
        hold = sum(rows[i] == gold[i] for i in rows if split[i] == "holdout")
        fails = sum(rows[i] is None for i in rows)
        tok = f"{tokens[name] / len(items):.0f}" if name in tokens else "0（不生成）"
        print(f"{name} | {dev}/15 | {hold}/10 | {fails} | {tok}")

    print("\n各組正確數（25 筆）")
    groups = list(dict.fromkeys(group.values()))
    print("做法 | " + " | ".join(f"{g}（{sum(1 for i in group if group[i] == g)}）" for g in groups))
    for name, rows in results.items():
        print(f"{name} | " + " | ".join(str(sum(rows[i] == gold[i] for i in rows if group[i] == g)) for g in groups))

    all_wrong = [i for i in gold if all(rows[i] != gold[i] for rows in results.values())]
    print(f"\n四種做法都錯的案例：{all_wrong or '無'}")
    text = {i["id"]: i["text"] for i in items}
    print("檢索 few-shot 仍然錯的案例（四種做法的標籤依序列出）")
    for i in gold:
        if results["檢索 few-shot"][i] != gold[i]:
            labels = "、".join(f"{rows[i] or '?'}" for rows in results.values())
            print(f"  {i} {split[i]} gold={gold[i]}：{labels}｜{text[i]}")

    print("\n檢索 few-shot 找到的範例（前 3 筆 holdout）")
    for item in [i for i in items if i["split"] == "holdout"][:3]:
        shots = "、".join(f"{n}({s:.2f})" for n, s in model.neighbours_of(item["id"]))
        print(f"  {item['id']} {item['text']} → {shots}")


def main(argv=()):
    items = json.loads(L38.find_data("eval-set.json").read_text(encoding="utf-8"))["items"]
    model = Live(items) if "--live" in argv else Replay()
    results, tokens = run_all(model, items)
    report(results, tokens, items, model)
    if "--save" in argv and isinstance(model, Live):
        import platform
        RECORD.parent.mkdir(exist_ok=True)
        RECORD.write_text(json.dumps({
            "note": "第 40 節重播用：Qwen2.5-0.5B 每個提示的輸出與輸入 token 數，以及 bge 的 dev 近鄰（留一）。不代表其他機器逐字相同。",
            "model": L38.MODEL_ID, "revision": L38.REVISION, "embedding": BGE_ID, "embedding_revision": BGE_REVISION,
            "recorded_on": f"{platform.node()}｜{platform.machine()}｜Python {platform.python_version()}",
            "neighbours": model.neighbours, "calls": dict(sorted(model.calls.items())),
        }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"\n已寫入 {RECORD}（{len(model.calls)} 個提示）")
    return results


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except LookupError as error:
        sys.exit(f"LookupError: {error}")
