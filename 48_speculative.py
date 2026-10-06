"""第 48 節：推測解碼與服務端推論。

一、推測解碼（真的跑 Qwen2.5-0.5B，CPU）
    draft 不是另一個模型，而是「從輸入裡找相同的 n-gram，把它後面的 token 當候選」（prompt lookup）。
    target 一次 forward 驗證 k 個候選：從頭比對 greedy 選擇，第一個不同的位置之後全部丟掉，換成 target 自己的 token。
    所以輸出和一般 greedy **逐 token 相同**；差別只在 target 要 forward 幾次。
二、continuous batching（標準函式庫模擬，不需要 GPU）
    同一批請求，靜態批次要等整批最長的做完；continuous batching 每一步都把空位補上新請求。

    python 48_speculative.py          # 重播示範機 4-learn 的紀錄（不需要模型）
    python 48_speculative.py --live   # 本機實測（需要 torch、transformers、Qwen 快取）；加 --save 覆寫紀錄
"""
import json
import re
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE / "data" / "48-speculative-record.json"
MAX_NEW_TOKENS = 48
DRAFT_K = 4          # 每次最多提出幾個候選
NGRAM = 2            # 用最後幾個 token 去輸入裡找相同的片段
THREADS = 2
SOP = ("本段為虛構教學資料。教學閘道器 A 與 B 只支援 2.4 GHz 無線網路，不支援 5 GHz。"
       "連線失敗時，請先確認頻段為 2.4 GHz，再確認模擬密碼是否正確，最後重新啟動閘道器。")
TASKS = {
    "照抄段落": f"請把下面這段文字原封不動地重複一次，不要增減任何字：\n{SOP}",
    "摘要改寫": f"請用你自己的話，以一句話摘要下面這段文字：\n{SOP}",
    "開放回答": "請說明工地為什麼要戴安全帽，舉兩個例子。",
}


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


def lookup_draft(tokens, k=DRAFT_K, n=NGRAM):
    """在目前序列（輸入＋已生成）裡，找最近一次出現的「最後 n 個 token」，回傳它後面的最多 k 個 token。"""
    if len(tokens) < n + 1:
        return []
    tail = tokens[-n:]
    for start in range(len(tokens) - n - 1, -1, -1):
        if tokens[start:start + n] == tail:
            return tokens[start + n:start + n + k]
    return []


def accept(draft, target_choices):
    """greedy 驗證：從頭比，相同就接受；第一個不同的位置換成 target 的選擇，之後全部丟掉。

    target_choices 比 draft 多一個：全部接受時，target 還順便給出下一個 token。
    """
    n = 0
    while n < len(draft) and draft[n] == target_choices[n]:
        n += 1
    return draft[:n] + [target_choices[n]], n


# ---------------------------------------------------------------- 實測（--live）

def measure():
    import importlib.metadata as metadata
    import platform
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    l38 = lesson("38")
    torch.set_num_threads(THREADS)
    path = snapshot_download(l38.MODEL_ID, revision=l38.REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])

    def plain(ids):
        """一般 greedy：每個 token 一次 forward。"""
        out = model(input_ids=ids, use_cache=True)
        past, tokens, calls = out.past_key_values, [], 1
        nxt = int(out.logits[0, -1].argmax())
        while True:
            tokens.append(nxt)
            if nxt in eos or len(tokens) >= MAX_NEW_TOKENS:
                return tokens, calls
            out = model(input_ids=torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
            past, calls = out.past_key_values, calls + 1
            nxt = int(out.logits[0, -1].argmax())

    def speculative(ids):
        """推測解碼：draft 提出候選，target 一次 forward 驗證全部候選。"""
        prompt = ids[0].tolist()
        out = model(input_ids=ids, use_cache=True)
        past, tokens, calls, proposed, accepted = out.past_key_values, [], 1, 0, 0
        last = int(out.logits[0, -1].argmax())
        steps = []
        while True:
            tokens.append(last)
            if last in eos or len(tokens) >= MAX_NEW_TOKENS:
                return tokens, calls, proposed, accepted, steps
            draft = lookup_draft(prompt + tokens)[:MAX_NEW_TOKENS - len(tokens)]
            cached = past.get_seq_length()
            out = model(input_ids=torch.tensor([[last] + draft]), past_key_values=past, use_cache=True)
            past, calls = out.past_key_values, calls + 1
            choices = out.logits[0].argmax(-1).tolist()
            new, n = accept(draft, choices)
            proposed, accepted = proposed + len(draft), accepted + n
            steps.append([len(draft), n])
            past.crop(cached + 1 + n)                      # 丟掉被拒絕的候選留在 cache 裡的 K／V
            for token in new[:-1]:
                tokens.append(token)
                if token in eos or len(tokens) >= MAX_NEW_TOKENS:
                    return tokens[:MAX_NEW_TOKENS], calls, proposed, accepted, steps
            last = new[-1]

    results = {}
    with torch.no_grad():
        for name, text in TASKS.items():
            ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                                return_tensors="pt")
            plain(ids)                                      # 暖機一次，不計時
            t0 = time.perf_counter()
            base_tokens, base_calls = plain(ids)
            t1 = time.perf_counter()
            spec_tokens, calls, proposed, accepted, steps = speculative(ids)
            t2 = time.perf_counter()
            results[name] = {
                "input_tokens": int(ids.shape[1]), "same_output": base_tokens == spec_tokens,
                "output_tokens": len(base_tokens), "text": tokenizer.decode(base_tokens, skip_special_tokens=True),
                "plain_calls": base_calls, "plain_seconds": round(t1 - t0, 3),
                "spec_calls": calls, "spec_seconds": round(t2 - t1, 3),
                "proposed": proposed, "accepted": accepted, "steps": steps,
            }
    return {
        "note": "第 48 節重播用：示範機實測。同一台機器、同一份程式，秒數每次仍會略有不同。",
        "model": l38.MODEL_ID, "revision": l38.REVISION, "threads": THREADS, "results": results,
        "versions": {p: metadata.version(p) for p in ("torch", "transformers")},
        "recorded_on": f"{platform.node()}｜{platform.machine()}｜Python {platform.python_version()}",
    }


# ---------------------------------------------------------------- continuous batching 模擬

REQUESTS = [("r1", 0, 30), ("r2", 0, 4), ("r3", 0, 6), ("r4", 0, 5),
            ("r5", 2, 5), ("r6", 3, 4), ("r7", 5, 6), ("r8", 6, 3)]     # (名稱, 到達的步數, 要生成幾個 token)
SLOTS = 4


def static_schedule(requests, slots):
    """靜態批次：整批最長的做完，才收下一批；先做完的請求只能空等，位子也空著。"""
    waiting, done, step, busy = list(requests), {}, 0, 0
    while waiting:
        batch = [r for r in waiting if r[1] <= step][:slots]
        if not batch:
            step += 1
            continue
        for r in batch:
            waiting.remove(r)
            done[r[0]] = step + r[2]
            busy += r[2]
        step += max(r[2] for r in batch)
    return done, step, busy


def continuous_schedule(requests, slots):
    waiting, running, done, step, busy = list(requests), {}, {}, 0, 0
    while waiting or running:
        for r in [r for r in waiting if r[1] <= step]:
            if len(running) >= slots:
                break
            running[r[0]] = r[2]
            waiting.remove(r)
        busy += len(running)
        for name in list(running):
            running[name] -= 1
            if running[name] == 0:
                done[name] = step + 1
                del running[name]
        step += 1
    return done, step, busy


# ---------------------------------------------------------------- 報告

def report(record, source):
    print(f"模型 {record['model']}@{record['revision'][:7]}｜CPU {record['threads']} 執行緒｜greedy｜"
          f"最多生成 {MAX_NEW_TOKENS} token｜draft：輸入裡的 {NGRAM}-gram，每次最多 {DRAFT_K} 個候選")
    print(f"來源：{source}｜{record['recorded_on']}｜" + "、".join(f"{k} {v}" for k, v in record["versions"].items()))

    print("\n== 一、推測解碼：輸出相同，target forward 次數不同 ==")
    print("任務 | 輸入 | 輸出 | 與 greedy 相同 | 接受率 | forward 次數（一般→推測） | 秒數（一般→推測） | 加速比")
    for name, r in record["results"].items():
        rate = f"{r['accepted'] / r['proposed']:.0%}（{r['accepted']}/{r['proposed']}）" if r["proposed"] else "—（沒有候選）"
        print(f"{name} | {r['input_tokens']} | {r['output_tokens']} | {'是' if r['same_output'] else '否！'} | {rate} | "
              f"{r['plain_calls']}→{r['spec_calls']} | {r['plain_seconds']:.2f}→{r['spec_seconds']:.2f} | "
              f"×{r['plain_seconds'] / r['spec_seconds']:.2f}")

    print("\n== 二、每次驗證接受了幾個候選（提出數:接受數，前 12 次） ==")
    for name, r in record["results"].items():
        print(f"  {name}：" + " ".join(f"{p}:{a}" for p, a in r["steps"][:12]))

    print("\n== 三、輸出（兩種方法逐 token 相同，只列一次） ==")
    for name, r in record["results"].items():
        print(f"  {name}：{r['text']}")

    print(f"\n== 四、服務端排程：{len(REQUESTS)} 個請求、{SLOTS} 個位子（每一步每個位子生成 1 個 token） ==")
    arrival = {r[0]: r[1] for r in REQUESTS}
    print("排程 | 全部完成的步數 | 位子使用率 | 平均等待＋生成步數 | r2（短）完成於 | r8（最後到）完成於")
    for label, fn in (("靜態批次", static_schedule), ("continuous batching", continuous_schedule)):
        done, steps, busy = fn(REQUESTS, SLOTS)
        latency = sum(done[n] - arrival[n] for n in done) / len(done)
        print(f"{label} | {steps} | {busy / (steps * SLOTS):.0%} | {latency:.1f} | 第 {done['r2']} 步 | 第 {done['r8']} 步")
    print("  模擬只數步數：沒有算 prefill、記憶體上限與 KV cache 搬移，真實服務要看實測")


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
