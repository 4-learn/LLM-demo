# Prefill、Decode 與 Token 成本

> **正文改版：20E07-E03-06｜對應新版 LLM 第 49 節：Prefill、Decode 與 Token 成本。**
> 一次生成分成兩段：**prefill** 一次處理整段輸入，產生第一個 token；**decode** 之後一個一個產生。本節在示範機的 CPU 上，用 Qwen2.5-0.5B 量出每個 token 出現的時間，固定輸出 32 token、只改輸入長度，看哪一段跟著變慢；冷載入另外記，不混進生成時間。最後用第 15 節教師核對的價目，算同樣的 token 數在雲端怎麼計費。核心主張：**先講清楚量的是哪一段、在什麼環境、重複幾次，數字才能比較；單次數值不是穩定效能。**

[回 51 節必修＋9 節選修課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 流程圖：Prefill、Decode 與 Token 成本

```mermaid
flowchart TD
    A[載入 CPU Qwen 與 tokenizer] --> B[建立短／中／長 prompt]
    B --> C[套用 chat template 並記錄輸入 token 數]
    C --> D[開始計時並 generate 固定 32 tokens]
    D --> E[Streamer 記錄每個 token 時間戳]
    E --> F[第一個 token 時間 - 開始 = TTFT／prefill 指標]
    E --> G[相鄰 token 時間差 = ITL／decode 指標]
    F --> H[重複量測：分開冷載入、暖機與後續樣本]
    G --> H
    H --> I[依 prompt 長度比較 TTFT、ITL 與端到端時間]
    I --> J[依相同 token 數計算教師核對的雲端成本]

## 學習目標與時間

先修第 02 節（token）、第 07 節（CPU 載入與記憶體）、第 15 節（供應商計費）、第 31 節（trace）。

| 時間 | 內容 |
| --- | --- |
| 0–10 分鐘 | 指標定義：TTFT、ITL、tokens/s、端到端；它們各量哪一段 |
| 10–20 分鐘 | 不能混在一起的東西：排隊、網路、冷載入、暖機 |
| 20–35 分鐘 | 跑 Demo（重播），讀延遲表 |
| 35–50 分鐘 | Workshop：寫一份可比較的延遲表 |

驗收能力：交出**含環境與執行模式的延遲表**，冷啟動與暖機分開記。

## 一、指標

| 指標 | 量什麼 | 主要受什麼影響 |
| --- | --- | --- |
| TTFT（time to first token） | 送出請求到第一個 token 出現 | **輸入長度**（prefill）、排隊、網路 |
| ITL（inter-token latency） | 相鄰兩個 token 的間隔 | 模型大小、硬體；上下文變長時略增 |
| decode tokens/s | 每秒產生幾個 token，約 1／ITL | 同上 |
| prefill tokens/s | 輸入 token 數 ÷ TTFT | 硬體、輸入長度 |
| 端到端延遲 | 送出到最後一個 token | TTFT ＋ 輸出 token 數 × ITL |

### 不能混在一起的東西

- **冷載入**：把權重讀進記憶體。只發生一次，本節另外記，不算進任何一次生成。
- **暖機**：第一次生成可能比較慢（記憶體配置、快取）。本節每種長度的第一次另列，不併入中位數。
- **排隊與網路**：雲端 API 的 TTFT 包含這兩段；本機量不到。拿本機數字和雲端比，要先講清楚。
- **輸入／輸出 token**：雲端分開計價，而且輸出通常比較貴（第 15 節）。

## 二、怎麼量

transformers 的 `generate` 可以接一個 streamer：每產生一個 token 就呼叫一次 `put`。Demo 的 `Stamps` 只記下每次的時間戳，第一個時間戳減開始時間就是 TTFT，後面相鄰時間戳的差就是 ITL。`min_new_tokens` 與 `max_new_tokens` 都設 32，確保每次輸出長度相同。

輸入長度用「在問題前面放幾份通報紀錄」控制（第 08 節評測集的 25 筆，當作背景資料）：短 0 份、中 1 份、長 4 份。

## 三、可執行 Demo

預設重播示範機的量測紀錄，只需要標準函式庫；同資料夾要有 `data/eval-set.json`、`data/provider-prices-checked.json`、`data/45-latency-record.json`。`--live` 需要 `torch`、`transformers`、第 07 節的 Qwen 快取與第 07 節的程式（`E02_*.py` 或講義）；示範機約 2 分 30 秒。

<!-- demo: E09_prefill_decode.py -->
```python
"""第 45 節：Prefill、Decode 與 Token 成本。

在 CPU 上用 Qwen2.5-0.5B 量每個 token 出現的時間，把一次生成拆成：
  prefill（處理整段輸入，到第一個 token 出現：TTFT）與 decode（之後每個 token：inter-token latency）。
固定輸出 32 token，只改輸入長度；冷載入、第一次（暖機）與之後的重複分開記。
最後用第 15 節教師核對的價目，算同樣 token 數在雲端的計費。

    python E09_prefill_decode.py          # 重播示範機 4-learn 的量測紀錄（不需要模型）
    python E09_prefill_decode.py --live   # 本機實測（需要 torch、transformers、Qwen 快取）；加 --save 覆寫紀錄
"""
import json
import re
import statistics
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE / "data" / "45-latency-record.json"
OUTPUT_TOKENS = 32
REPEATS = 3
THREADS = 2
CONTEXT_COPIES = {"短": 0, "中": 1, "長": 4}     # 在問題前面放幾份「通報紀錄」當背景


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

    smoke = lesson("07")
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
        "note": "第 45 節重播用：示範機實測的每個 token 時間。不同機器、負載、執行緒數都會不同，不要當成穩定效能。",
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

    print("\n== 三、同樣的 token 數，雲端怎麼計費（第 15 節教師核對價，美元） ==")
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
```

執行（程式與資料也放在 [llm-demo](https://github.com/4-learn/llm-demo)）：

```bash
python E09_prefill_decode.py
```

關鍵輸出（重播示範機 4-learn 20E07-E03-06 的 `--live --save` 紀錄；當次即時輸出只差「來源」一行的開頭）：

```text
模型 Qwen/Qwen2.5-0.5B-Instruct@7ae5576｜CPU 2 執行緒｜float32｜greedy｜輸出固定 32 token
來源：重播示範機紀錄｜4-learn｜x86_64｜Python 3.12.3｜torch 2.8.0+cpu、transformers 4.57.3
冷載入（讀權重到記憶體）：0.8 秒——不算在下面任何一次生成裡

== 一、只改輸入長度（每種長度：第一次另列，之後 3 次取中位數） ==
輸入 | 輸入 token | 第一次 TTFT | TTFT 中位數 | prefill token/s | ITL 中位數 | decode token/s | 端到端中位數
短 | 47 | 0.33 秒 | 0.34 秒 | 137 | 129 ms | 7.7 | 4.36 秒
中 | 396 | 1.94 秒 | 2.00 秒 | 198 | 141 ms | 7.1 | 6.38 秒
長 | 1443 | 11.26 秒 | 12.61 秒 | 114 | 184 ms | 5.4 | 18.34 秒

== 二、輸入變長，哪一段變慢？ ==
  輸入 47 → 1443 token（×31）：TTFT ×36.8，ITL ×1.43，端到端 ×4.2
  「長」的 3 次 TTFT：12.69、12.56、12.61 秒（同一台機器、同一個輸入也會不同）

== 三、同樣的 token 數，雲端怎麼計費（第 15 節教師核對價，美元） ==
  價目核對日 2026-10-06；以「長」輸入 1443 token、輸出 32 token、每天 1,000 次計
  模型 | 每 1M 輸入 | 每 1M 輸出 | 每天輸入費 | 每天輸出費 | 輸入費占比
  gpt-4o-mini | 0.15 | 0.6 | 0.216 | 0.019 | 92%
  deepseek-flash | 0.3 | 1.2 | 0.433 | 0.038 | 92%
  claude-haiku-4-5 | 1.0 | 5.0 | 1.443 | 0.160 | 90%
  gemini-2.5-flash | 0.3 | 2.5 | 0.433 | 0.080 | 84%
  本機推論不按 token 計費，但輸入越長 TTFT 越久：時間就是本機的成本。
```

## 四、結果怎麼讀

### 1. 輸入變長，主要是 TTFT 變慢

輸入從 47 變成 1443 token（31 倍），TTFT 變成 37 倍，ITL 只變成 1.4 倍。輸出固定 32 token 時，「長」的端到端 18 秒裡有 12.6 秒在等第一個 token。**要縮短 TTFT，先縮短輸入**（第 23 節的 chunk、第 47 節的 prefix cache）。

### 2. prefill 在長輸入時反而變慢

prefill tokens/s：短 137、中 198、長 114。短輸入時固定成本占比高；長輸入時 attention 要計算每個 token 與所有前面 token 的關係，成本隨長度超過線性增加。**「prefill 每秒幾個 token」不是一個固定值**，要說明是在多長的輸入下量的。

### 3. 暖機效果不明顯，但還是要分開

示範機每種長度的第一次 TTFT 並沒有比較慢（「長」第一次 11.26 秒，之後 12.56～12.69 秒，反而較快）。冷載入已經另外記（0.8 秒），這個小模型在 CPU 上的暖機效果很小。**分開記不是因為一定有差，而是有差時才看得出來。** 在 GPU、較大的模型或雲端服務上，第一次常常明顯較慢。

### 4. 同一台機器也有變動

「長」的 3 次 TTFT 在 12.56～12.69 秒之間；備課機同樣程式的「長」TTFT 中位數是 13.45 秒、ITL 199 ms。**換機器、換負載、換執行緒數都會改變數字**；延遲表一定要寫環境。

### 5. 雲端計費：輸入占大頭

輸入 1443 token、輸出 32 token 時，四家的輸入費都占 84%～92%。輸出單價雖然比較貴，但這個任務的輸出很短。**輸入長的任務，省輸入 token 最省錢。** 價格是 20E07-E03-06 的核對價，上課前要重新核對。

## 五、常見誤判

| 誤判 | 實際 |
| --- | --- |
| 「這台每秒 7 個 token」 | 那是 decode；prefill 是另一個數字，而且隨輸入長度變 |
| 「輸入變長，每個 token 都變慢」 | 主要是 TTFT 變慢，ITL 只略增 |
| 「量一次就知道效能」 | 同機器也有變動；要重複、取中位數、寫環境 |
| 「載入加生成一起計時」 | 冷載入只發生一次，要分開 |
| 「輸出單價比較貴，所以輸出費比較多」 | 輸入長、輸出短時，輸入費占大頭 |
| 「本機不用錢」 | 時間與硬體就是成本；輸入越長等越久 |

## Workshop：寫一份可比較的延遲表

### 任務

1. （3 分鐘）跑 Demo（重播），說明「長」的端到端 18.34 秒裡，大約有多少秒是 prefill、多少秒是 decode。
2. （4 分鐘）用 `TTFT ＋ 輸出 token × ITL` 估算：如果「長」改成輸出 128 token，端到端大約幾秒？輸入費占比會變多少（以 gpt-4o-mini 計）？
3. （5 分鐘，有模型快取的同學）在自己的電腦跑 `python E09_prefill_decode.py --live`，把結果和示範機並排，填寫：CPU、執行緒數、記憶體、Python 與套件版本、是否有其他程式在跑。
4. （3 分鐘）你的延遲表裡，哪一欄最能說明「縮短輸入」的效益？

### 參考判讀

- 第 1 題：prefill 約 12.6 秒（TTFT），decode 約 31 × 0.184 ≈ 5.7 秒。
- 第 2 題：12.6 ＋ 127 × 0.184 ≈ 36 秒（ITL 在輸出變長時還會略增，所以這是下限估計）。gpt-4o-mini：輸入 1443 × 0.15、輸出 128 × 0.6，輸入費占比約 74%。
- 第 3 題：數字不同是正常的；重點是環境欄位寫齊，別人才能判斷能不能比。
- 第 4 題：TTFT（或 prefill 秒數）。ITL 幾乎不受輸入長度影響。

### 驗收

- 第 3 題的延遲表（環境欄位齊全；沒有模型的同學用示範機紀錄，註明來源）。
- 第 2 題的估算過程。

## 教師紀錄與下一步

- 證據等級：**遠端實測**。20E07-E03-06 示範機 4-learn（i7-8565U、8 GB、Python 3.12.3、torch 2.8.0+cpu、2 執行緒、負載約 0.3）以 `--live --save` 實測，紀錄存為 `data/45-latency-record.json`，重播與當次即時輸出相同（只差來源標示）。備課機同日 `--live`：數字不同（「長」TTFT 中位數 13.45 秒、ITL 199 ms，負載約 4），趨勢相同。
- 撰寫時的實際紀錄：第一版的「長」放 16 份通報紀錄（5638 token），備課機 10 分鐘仍未跑完，改成 4 份（1443 token）。上課時的 CPU 機器不要嘗試太長的輸入。
- 「第一次 TTFT」只代表每種長度的第一次；全程只有短輸入的第一次是真正的第一次生成。
- 沒有量雲端 API 的實際延遲（需要付費帳號，第 15 節未決）。
- 舊教材 [Inference](https://hackmd.io/@yillkid/M1IwAffeRhWVk8nMKD1V7Q) **尚未修改**。
- 下一步：第 46 節，量化與 CPU 實驗。
