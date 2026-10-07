"""第 43 節：微調前後評估與 Adapter 交付。

在 CPU 上對 Qwen2.5-0.5B 做一次很小的 LoRA（q_proj、v_proj，r=8），訓練資料只有第 07 節的 dev 15 筆；
用 holdout 10 筆比較：base zero-shot、base 檢索 few-shot（第 40 節）、base＋adapter。
同一個提示、同樣的解碼與 parse。另外檢查退步（非分類問題）與合併前後是否一致，最後寫出 artifact manifest。

    python 43_adapter_eval.py           # 重播示範機 4-learn 的訓練與評估紀錄（不需要模型）
    python 43_adapter_eval.py --train   # 真的訓練、評估並寫出 adapter（需要 torch、transformers、peft==0.17.1、Qwen 快取）
"""
import hashlib
import json
import re
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE / "data" / "43-adapter-record.json"
OUT = Path("43-adapter")
SEED = 20261006
TRAIN = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "target_modules": ["q_proj", "v_proj"],
         "epochs": 4, "lr": 2e-4, "batch_size": 1, "threads": 4}
PROBES = ("請用一句話說明為什麼工地要戴安全帽。", "把「安全帽」翻譯成英文。")   # 非分類問題：看有沒有退步


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


def load_items():
    return json.loads(L38.find_data("eval-set.json").read_text(encoding="utf-8"))["items"]


def classify_messages(text):
    return [{"role": "system", "content": TASK}, {"role": "user", "content": f"通報：{text}\n標籤："}]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- 訓練與評估（--train）

def train_and_evaluate(items):
    import platform
    import random
    import importlib.metadata as metadata
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    torch.set_num_threads(TRAIN["threads"])
    path = snapshot_download(L38.MODEL_ID, revision=L38.REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path)

    def load_base():
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        for name in ("temperature", "top_p", "top_k"):
            setattr(model.generation_config, name, None)
        return model

    def generate(model, messages, max_new_tokens=L38.MAX_NEW_TOKENS):
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def run_eval(model):
        labels = {i["id"]: generate(model, classify_messages(i["text"])) for i in items}
        probes = [generate(model, [{"role": "user", "content": q}], 40) for q in PROBES]
        return labels, probes

    train_items = [i for i in items if i["split"] == "dev"]
    assert all(i["split"] == "dev" for i in train_items)

    base = load_base()
    base_labels, base_probes = run_eval(base)

    def encode(item):
        prompt = tokenizer.apply_chat_template(classify_messages(item["text"]), add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)["input_ids"][0]
        target = tokenizer(item["gold"] + "<|im_end|>", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        ids = torch.cat([prompt, target])
        labels = torch.cat([torch.full_like(prompt, -100), target])     # 只對回覆算 loss
        return ids.unsqueeze(0), labels.unsqueeze(0)

    config = LoraConfig(r=TRAIN["r"], lora_alpha=TRAIN["lora_alpha"], lora_dropout=TRAIN["lora_dropout"],
                        target_modules=TRAIN["target_modules"], task_type="CAUSAL_LM")
    model = get_peft_model(base, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=TRAIN["lr"])
    encoded = [encode(i) for i in train_items]
    rng = random.Random(SEED)
    losses = []
    started = time.perf_counter()
    model.train()
    for epoch in range(TRAIN["epochs"]):
        order = list(range(len(encoded)))
        rng.shuffle(order)
        total = 0.0
        for n in order:
            ids, labels = encoded[n]
            loss = model(input_ids=ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()
        losses.append(round(total / len(encoded), 4))
    train_seconds = time.perf_counter() - started
    model.eval()
    model.save_pretrained(OUT)

    del model, base
    reloaded = PeftModel.from_pretrained(load_base(), OUT)          # smoke test：從磁碟重新載入 base＋adapter
    lora_labels, lora_probes = run_eval(reloaded)
    merged = reloaded.merge_and_unload()
    merged_labels = {i["id"]: generate(merged, classify_messages(i["text"])) for i in items if i["split"] == "holdout"}

    files = sorted(p.name for p in OUT.iterdir())
    return {
        "note": "第 43 節重播用：示範機實際訓練 LoRA 的紀錄與評估輸出。adapter 檔案本身不放進教材，只記雜湊；要檔案請用 --train 重建。",
        "base": {"model": L38.MODEL_ID, "revision": L38.REVISION},
        "train": dict(TRAIN, seed=SEED, samples=[i["id"] for i in train_items], trainable_params=trainable,
                      epoch_loss=losses, seconds=round(train_seconds, 1)),
        "adapter_files": {name: sha256(OUT / name) for name in files if name.endswith((".json", ".safetensors"))},
        "versions": {p: metadata.version(p) for p in ("torch", "transformers", "peft")},
        "recorded_on": f"{platform.node()}｜{platform.machine()}｜Python {platform.python_version()}",
        "outputs": {"base": base_labels, "lora": lora_labels, "merged_holdout": merged_labels,
                    "probes_base": base_probes, "probes_lora": lora_probes},
    }


# ---------------------------------------------------------------- 報告

def report(record, items, source):
    gold = {i["id"]: i["gold"] for i in items}
    holdout = [i["id"] for i in items if i["split"] == "holdout"]
    dev = [i["id"] for i in items if i["split"] == "dev"]
    out = record["outputs"]
    rag = lesson("40").Replay(L38.find_data("40-prompt-rag-traces.json"))
    rag_labels = lesson("40").run_all(rag, items)[0]["檢索 few-shot"]
    rows = {"base zero-shot": {i: parse(t) for i, t in out["base"].items()},
            "base 檢索 few-shot（第 40 節）": rag_labels,
            "base＋adapter": {i: parse(t) for i, t in out["lora"].items()}}
    t = record["train"]
    print(f"base {record['base']['model']}@{record['base']['revision'][:7]}｜LoRA r={t['r']} α={t['lora_alpha']} "
          f"{'、'.join(t['target_modules'])}｜可訓練參數 {t['trainable_params']:,}")
    print(f"訓練資料：dev {len(t['samples'])} 筆｜{t['epochs']} epoch、lr {t['lr']}、seed {t['seed']}｜"
          f"每 epoch loss {t['epoch_loss']}｜訓練 {t['seconds']} 秒（{record['recorded_on'].split('｜')[0]}）")
    print(f"來源：{source}")

    print("\n== 一、前後評估（同提示、greedy、同 parse） ==")
    print("做法 | holdout 正確（10） | dev 正確（15，訓練過）")
    for name, labels in rows.items():
        hold = sum(labels[i] == gold[i] for i in holdout)
        seen = sum(labels[i] == gold[i] for i in dev)
        note = "" if name != "base 檢索 few-shot（第 40 節）" else "（留一）"
        print(f"{name} | {hold} | {seen}{note}")

    print("\n== 二、holdout 逐筆 ==")
    text = {i["id"]: i["text"] for i in items}
    for i in holdout:
        cells = "｜".join(f"{'✓' if labels[i] == gold[i] else '✗'}{labels[i] or '?'}" for labels in rows.values())
        print(f"  {i} gold={gold[i]}：{cells}｜{text[i]}")

    print("\n== 三、退步檢查：不是分類的問題 ==")
    for q, before, after in zip(PROBES, out["probes_base"], out["probes_lora"]):
        print(f"  問：{q}")
        print(f"    base   ：{before}")
        print(f"    adapter：{after}")

    print("\n== 四、合併後是否一致（merge_and_unload，holdout） ==")
    same = sum(out["merged_holdout"][i] == out["lora"][i] for i in holdout)
    print(f"  合併前後輸出相同 {same}/{len(holdout)}")

    print("\n== 五、artifact manifest ==")
    print(f"  base：{record['base']['model']} revision {record['base']['revision']}")
    for name, digest in record["adapter_files"].items():
        print(f"  {name}：sha256 {digest[:16]}…")
    print(f"  版本：" + "、".join(f"{k} {v}" for k, v in record["versions"].items()))
    print(f"  錄製機器：{record['recorded_on']}")


def main(argv=()):
    items = load_items()
    if "--train" in argv:
        record = train_and_evaluate(items)
        if "--save" in argv:
            RECORD.parent.mkdir(exist_ok=True)
            RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        source = f"本機訓練（adapter 寫在 {OUT}/）"
    else:
        record = json.loads(RECORD.read_text(encoding="utf-8"))
        source = "重播示範機 4-learn 的訓練與評估紀錄（不需要模型）"
    report(record, items, source)
    return record


if __name__ == "__main__":
    main(sys.argv[1:])
