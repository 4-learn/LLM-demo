# 推測解碼與服務端推論

> **正文改版：20E07-E03-06｜對應新版 LLM 第 49 節：推測解碼與服務端推論。**
> 推測解碼（speculative decoding）讓 decode 一次前進好幾個 token：先用便宜的方法**猜**接下來幾個 token，再讓目標模型**一次 forward 驗證**全部候選，接受從頭開始連續猜對的部分。本節在示範機 CPU 上真的實作一個最小版本：draft 不是另一個模型，而是「在輸入裡找相同的片段」（prompt lookup），驗證規則是 greedy。重點是**輸出與一般 greedy 逐 token 相同**，加速多少完全取決於接受率。第二部分用標準函式庫模擬服務端的 continuous batching。核心主張：**推測解碼是 token 級的猜測＋驗證，不是小模型先答、大模型再審；效益取決於接受率、配對、batch 與硬體，要量了才知道。**

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先修第 03 節（logits 與 forward）、第 04 節（sampling）、第 45 節（prefill、decode）、第 47 節（KV cache）。

| 時間 | 內容 |
| --- | --- |
| 0–15 分鐘 | token 級流程：提出、驗證、接受、修正；為什麼輸出不變 |
| 15–25 分鐘 | 跑 Demo（重播）：三種任務的接受率與加速比 |
| 25–35 分鐘 | 服務端：靜態批次與 continuous batching |
| 35–50 分鐘 | Workshop：計算接受率與加速比，找出沒有加速的案例 |

驗收能力：畫出 **token 級流程圖**，並用接受率說明效能數字。不需要 GPU，也不需要 vLLM。

## 一、token 級流程

一般 decode：目標模型每 forward 一次，產生 1 個 token。

推測解碼的一輪：

1. **提出**：draft 猜接下來的 k 個 token（本節 k＝4）。
2. **驗證**：目標模型把「上一個 token＋k 個候選」一次送進去 forward。因為有 KV cache（第 47 節），這次 forward 只要處理 k＋1 個新 token，而且它們是**平行**算的。forward 會給出每個位置的下一個 token 分布。
3. **接受**：從第一個候選開始比對：目標模型在這個位置也會選它嗎？相同就接受，**第一個不同的位置停下**，後面的候選全部丟掉。
4. **修正**：在停下的位置，換成目標模型自己的選擇。所以每一輪**至少**前進 1 個 token（全部猜錯時就跟一般 decode 一樣），最多前進 k＋1 個。
5. 被丟掉的候選已經寫進 KV cache，要裁掉（`past.crop`）。

**為什麼輸出不變：** 每一個被接受的 token，都是目標模型在那個位置本來就會選的。本節用 greedy，所以比對就是「argmax 是否相同」。使用取樣時，標準演算法用接受／拒絕取樣與修正分布，保證輸出的**分布**與目標模型相同；那需要 draft 的機率分布，本節的 prompt lookup 沒有分布，所以只示範 greedy。

### 這不是什麼

| 常見混淆 | 差別 |
| --- | --- |
| 小模型先寫完整回答，大模型再審 | 推測解碼是 **token 級**，而且輸出由目標模型決定 |
| 蒸餾（distillation） | 蒸餾是訓練一個小模型去模仿；推測解碼不訓練、不改模型 |
| MoE（mixture of experts） | MoE 是模型架構，每個 token 只用一部分參數；跟猜測與驗證無關 |

### draft 從哪來

- **小的 draft 模型**（同家族、同 tokenizer）：最常見，需要額外記憶體。
- **prompt lookup**（本節）：在輸入裡找最後 n 個 token 出現過的位置，把後面的 token 當候選。不需要額外模型，但只有**輸出會重複輸入內容**的任務才猜得中。

## 二、可執行 Demo

預設重播示範機的紀錄，只需要標準函式庫；同資料夾要有 `data/49-speculative-record.json`。`--live` 需要 `torch`、`transformers`、第 07 節的 Qwen 快取與第 39 節的程式（`E07_*.py` 或講義），示範機約 1 分 30 秒。

<!-- demo: E09_speculative.py -->
```python
"""第 49 節：推測解碼與服務端推論。

一、推測解碼（真的跑 Qwen2.5-0.5B，CPU）
    draft 不是另一個模型，而是「從輸入裡找相同的 n-gram，把它後面的 token 當候選」（prompt lookup）。
    target 一次 forward 驗證 k 個候選：從頭比對 greedy 選擇，第一個不同的位置之後全部丟掉，換成 target 自己的 token。
    所以輸出和一般 greedy **逐 token 相同**；差別只在 target 要 forward 幾次。
二、continuous batching（標準函式庫模擬，不需要 GPU）
    同一批請求，靜態批次要等整批最長的做完；continuous batching 每一步都把空位補上新請求。

    python E09_speculative.py          # 重播示範機 4-learn 的紀錄（不需要模型）
    python E09_speculative.py --live   # 本機實測（需要 torch、transformers、Qwen 快取）；加 --save 覆寫紀錄
"""
import json
import re
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE / "data" / "49-speculative-record.json"
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
    """同第 32 節：先找 NN_*.py，再找講義 NN-*.md；讀原始碼後 exec。"""
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

    l38 = lesson("09")
    l38.MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    l38.REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
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
        "note": "第 49 節重播用：示範機實測。同一台機器、同一份程式，秒數每次仍會略有不同。",
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
```

執行（程式與資料也放在 [llm-demo](https://github.com/4-learn/llm-demo)）：

```bash
python E09_speculative.py
```

關鍵輸出（重播示範機 4-learn 20E07-E03-06 的 `--live --save` 紀錄；當次即時輸出只差「來源」一行的開頭）：

```text
模型 Qwen/Qwen2.5-0.5B-Instruct@7ae5576｜CPU 2 執行緒｜greedy｜最多生成 48 token｜draft：輸入裡的 2-gram，每次最多 4 個候選
來源：重播示範機紀錄｜4-learn｜x86_64｜Python 3.12.3｜torch 2.8.0+cpu、transformers 4.57.3

== 一、推測解碼：輸出相同，target forward 次數不同 ==
任務 | 輸入 | 輸出 | 與 greedy 相同 | 接受率 | forward 次數（一般→推測） | 秒數（一般→推測） | 加速比
照抄段落 | 123 | 48 | 是 | 79%（34/43） | 48→15 | 6.90→3.15 | ×2.19
摘要改寫 | 117 | 48 | 是 | 12%（3/24） | 48→45 | 6.85→6.77 | ×1.01
開放回答 | 42 | 48 | 是 | 29%（8/28） | 48→40 | 6.30→5.62 | ×1.12

== 二、每次驗證接受了幾個候選（提出數:接受數，前 12 次） ==
  照抄段落：0:0 4:4 4:4 4:1 4:0 0:0 0:0 4:4 4:2 4:4 4:4 4:4
  摘要改寫：0:0 4:0 0:0 0:0 0:0 0:0 0:0 0:0 0:0 0:0 0:0 0:0
  開放回答：0:0 0:0 0:0 4:0 0:0 0:0 0:0 0:0 0:0 0:0 0:0 0:0

== 三、輸出（兩種方法逐 token 相同，只列一次） ==
  照抄段落：本段為虛構教學資料。教學閘道器 A 和 B 只支援 2.4 GHz 無線網路，不支援 5 GHz。連線失敗時，請先確認頻段為
  摘要改寫：本段为虚构教学资料。教学设备A和B仅支持2.4GHz无线网络，不支持5GHz。连接失败时，请先确认频段为2.4GHz，再确认模拟密码是否正确，
  開放回答：工地上的安全帽主要目的是为了保护工人在进行高处作业时的安全。戴安全帽的主要目的是防止工人在高处作业时发生坠落事故，以及防止工人在进行高处作业时发生磕碰事故。

== 四、服務端排程：8 個請求、4 個位子（每一步每個位子生成 1 個 token） ==
排程 | 全部完成的步數 | 位子使用率 | 平均等待＋生成步數 | r2（短）完成於 | r8（最後到）完成於
靜態批次 | 36 | 44% | 20.9 | 第 4 步 | 第 33 步
continuous batching | 30 | 52% | 8.9 | 第 4 步 | 第 12 步
  模擬只數步數：沒有算 prefill、記憶體上限與 KV cache 搬移，真實服務要看實測
```

## 三、結果怎麼讀

### 1. 三個任務都和 greedy 逐 token 相同

這是推測解碼最重要的性質：**品質不變**，只改變目標模型要 forward 幾次。如果你看到「推測解碼後答案變了」，那是實作錯了（或用了不同的驗證規則）。

### 2. 接受率決定一切

| 任務 | 接受率 | forward | 示範機加速 |
| --- | --- | --- | --- |
| 照抄段落 | 79% | 48 → 15 | ×2.19 |
| 摘要改寫 | 12% | 48 → 45 | ×1.01 |
| 開放回答 | 29% | 48 → 40 | ×1.12 |

- **照抄段落**：輸出幾乎就是輸入，prompt lookup 一猜就中，常常 4 個全中。
- **摘要改寫**：模型輸出簡體字，和輸入的繁體字不同 token，44 次驗證裡 38 次**根本沒有候選**（找不到相同片段）。
- **開放回答**：輸入很短，能猜的來源很少；29% 的接受率主要來自輸出自己重複的片段（「高处作业时」出現三次）。

### 3. 同一份程式，在另一台機器上變慢

備課機（負載約 4）同日跑：接受率、forward 次數、輸出完全相同，但秒數變成照抄 ×1.45、摘要 **×0.88**、開放 **×0.93**。一次驗證 5 個 token 的 forward 比驗證 1 個慢；接受率低時，多出來的計算沒有換到前進的 token，**加速比小於 1**。在負載較高的機器上，這個額外成本更明顯。**同樣的接受率，加速比仍然取決於硬體與負載。**

### 4. 加速比怎麼算

本節的加速比是**同一台機器、同一次執行**裡，一般 greedy 秒數 ÷ 推測解碼秒數。只有一次量測，不是穩定數字（第 45 節）。理論上限約為「每次 forward 平均前進幾個 token」：照抄段落 48 ÷ 15 ≈ 3.2，實測 2.19，差距來自驗證時多算的 token 與 cache 裁切。

## 四、服務端：continuous batching

同一台 GPU 服務多個請求時，會把它們放進同一個 batch 一起 decode。第四段模擬 8 個請求、4 個位子：

- **靜態批次**：一批 4 個請求一起開始，**要等最長的那個（r1，30 個 token）做完**，才能收下一批。r2 只要 4 步就做完了，但它的位子要空等 26 步。
- **continuous batching**：每一步開始前，有空位就立刻補上已經到達的請求。

結果：全部完成從 36 步變 30 步；平均「等待＋生成」從 20.9 步降到 8.9 步；最後到的 r8 從第 33 步提前到第 12 步。vLLM 等推論服務就是用這種排程，再加上 KV cache 的分頁管理（PagedAttention），讓位子不會被浪費。

這個模擬**只數步數**：沒有算 prefill（新請求加入時要先處理它的輸入）、記憶體上限、batch 變大時每一步變慢等因素。真實服務要看實測：吞吐量、TTFT 的 p50／p90、KV cache 使用率。

## 五、常見誤判

| 誤判 | 實際 |
| --- | --- |
| 「推測解碼會降低品質」 | greedy 驗證下逐 token 相同；取樣時分布相同 |
| 「推測解碼一定更快」 | 接受率低時可能 ×0.88 |
| 「小模型先答、大模型檢查」 | token 級猜測＋驗證，不是語義審查 |
| 「接受率高就一定快很多」 | 還要看驗證的額外成本、硬體與負載 |
| 「continuous batching 讓每個請求都變快」 | 主要改善排隊與使用率；batch 變大時每一步可能變慢 |

## Workshop：接受率與加速比

### 任務

1. （3 分鐘）畫出照抄段落第 2 次驗證（提出 4、接受 4）的流程：送進目標模型的是哪幾個 token？這一輪前進幾個 token？
2. （4 分鐘）第 4 次驗證是「4:1」。這一輪前進幾個 token？KV cache 要裁掉幾個位置？
3. （4 分鐘）用第一段的數字，算三個任務「每次 forward 平均前進幾個 token」，和實測加速比對照。哪一個任務的差距最大？為什麼？
4. （4 分鐘）摘要改寫的接受率只有 12%。提出兩種可能的改善方式，並說明各自的代價。

### 參考判讀

- 第 1 題：送進去 5 個 token（上一個 token＋4 個候選）；4 個全接受，再加上目標模型在最後一個位置給出的下一個 token，前進 5 個。
- 第 2 題：接受 1 個＋目標模型修正 1 個，前進 2 個。送進去 5 個位置，保留「上一個 token＋接受的 1 個」共 2 個，裁掉 3 個。
- 第 3 題：照抄 48÷15≈3.2（實測 2.19）；摘要 48÷45≈1.07（1.01）；開放 48÷40=1.2（1.12）。照抄的差距最大：每次驗證送 5 個 token，forward 比只送 1 個慢，前進越多，這個額外成本越明顯。
- 第 4 題：① 改用小的 draft 模型（同 tokenizer），可以猜「改寫」的內容，代價是額外記憶體與 draft 本身的計算；② 讓模型用繁體輸出（提示或微調），輸出更可能重複輸入的 token，代價是可能影響回答品質，要另外評估。

### 驗收

- 第 1、2 題的 token 級流程圖。
- 第 3 題的計算與判讀。

## 教師紀錄與下一步

- 證據等級：**遠端實測＋模擬**。推測解碼：20E07-E03-06 示範機 4-learn（i7-8565U、8 GB、Python 3.12.3、torch 2.8.0+cpu、2 執行緒）以 `--live --save` 實測，紀錄存為 `data/49-speculative-record.json`；重播與當次即時輸出相同（只差來源標示）。備課機同日：接受率、forward 次數、輸出完全相同，秒數不同（見第三節第 3 點）。continuous batching 是標準函式庫模擬，不是實測。
- 推測解碼是自己寫的最小實作（手動呼叫 forward、裁切 `DynamicCache`），不是 transformers 內建的 `prompt_lookup_num_tokens` 或 `assistant_model`；目的是看得到每一步，不是效能。內建版本與 vLLM **沒有測**。
- 沒有 GPU，也沒有跑 vLLM；課綱說的「預備 vLLM 資料或教師 trace」目前**沒有**，第四段用模擬替代。
- 舊教材 [生成](https://hackmd.io/@yillkid/ruEtEfmUQs-n-rN4vfbTjQ) **尚未修改**。
- 下一步：第 50 節增量驗收。
