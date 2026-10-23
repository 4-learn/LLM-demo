# VLM 打標與 Metadata 品質

> **正文改版：20E07-E03-06｜對應新版 LLM 第 16 節：VLM 打標與 Metadata 品質。**
> 第 10–12 節逐張檢查模型看圖的結果。實務上圖片有幾千張，會讓模型批次打標籤，再用標籤做檢索過濾（第 22 節）。本節對 10 張合成工地圖批次打 3 個標籤，把結果寫成**帶來源的 metadata**（image ID、雜湊、產生方式、模型 revision、提示版本、覆核狀態），做 schema 檢查、與人工標註比對、覆核修正；並示範兩個常見錯誤：**未覆核就拿來過濾**、**把 unknown 默認為 no**。核心主張：**模型標籤是待覆核的提議，不是真值；unknown 是一個值，不是 no；每筆 metadata 都要說得出它從哪裡來。**

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先修第 10 節（unknown 與把關）、第 13 節（schema 檢查）、第 15 節（版本紀錄）。後接第 22 節（metadata 過濾）。

| 時間 | 內容 |
| --- | --- |
| 0–10 分鐘 | metadata 欄位設計：ID、標籤、unknown、來源、版本、覆核 |
| 10–20 分鐘 | 自動標籤錯誤如何影響後續過濾 |
| 20–35 分鐘 | 跑 Demo |
| 35–50 分鐘 | Workshop：schema 檢查、人工抽查、修正清單 |

驗收能力：交出**含 provenance 的 metadata 檔**（JSON Lines）及**修正清單**；unknown 不得默認為否。

## 一、metadata 要有哪些欄位

| 欄位 | 例 | 為什麼 |
| --- | --- | --- |
| `image_id`、`image_sha256_16` | `IMG-01`、`37f582fe…` | 圖換了，標籤就不能沿用 |
| `tags` | `{"person": "yes", "helmet": "unknown", "cone": "no"}` | 列舉值 yes／no／unknown，**不用布林** |
| `source` | `model`、`model+human`、`human` | 這個值是誰決定的 |
| `model` | `SmolVLM-500M-Instruct@a7da5b98…` | 模型產生的要有 revision |
| `prompt_version` | `tags-20E07-E03-06` | 提示改了，標籤分布就會變（第 11 節：加一句話答案就變） |
| `review` | `pending`／`approved`／`corrected`／`quarantined` | 第 22 節的過濾只收覆核過的 |
| `raw` | 模型原始回答 | 覆核修正後仍可追溯 |

文件類 metadata 另外要有文件 ID、版本、頁碼與轉寫來源（第 12 節）。

## 二、標籤錯誤會跑到哪裡去

標籤一旦寫進 metadata，就會被當成過濾條件（「找出沒戴安全帽的圖」）。錯的標籤不會在檢索時報錯，只會**安靜地**多找或少找。最常見的兩個錯：

1. **未覆核就用**：模型標籤直接進索引。
2. **unknown 被當成 no**：程式寫 `if not tags["helmet"]`，或資料庫欄位是布林、NULL 被當成 false。

## 三、可執行 Demo

需要同資料夾第 10 節的程式（畫圖函式與把關規則）。重播只需要標準函式庫（有 Pillow 會重畫核對雜湊）；`--live` 需求同第 10 節，10 張圖 40 個問題，示範機約 1 分 20 秒。

<!-- demo: E03_vlm_tagging.py -->
```python
"""第 16 節：VLM 打標與 Metadata 品質。

一小批 10 張合成工地圖（重用第 10 節的畫圖函式），本地 SmolVLM-500M 對每張回答 3 個標籤問題：
有沒有人、有沒有戴安全帽（看不清楚時由第 10 節的把關改成 unknown）、有沒有交通錐。
把回答轉成 metadata 記錄：每一筆帶 image ID、雜湊、產生方式、模型與 revision、提示版本、覆核狀態。
接著：
  一、schema 檢查：欄位、型別、列舉值；unknown 是合法值，不能被當成 false
  二、與人工標註比對：哪些標籤錯了、錯的方向
  三、抽查與隔離：錯誤與未覆核的標籤不能進入檢索過濾（第 22 節）
  四、示範「unknown 被默認為否」會讓過濾結果怎麼錯

預設重播示範機 4-learn 錄下的回答（需要同資料夾第 10 節的程式）；`--live` 實跑。

    python E03_vlm_tagging.py
    python E03_vlm_tagging.py --live [--save]
    python E03_vlm_tagging.py --out metadata.jsonl      # 把 metadata 寫成 JSON Lines
"""
import argparse
import hashlib
import json
import platform
import re
import resource
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE / "data" / "16-tagging-record.json"
PROMPT_VERSION = "tags-20E07-E03-06"
TAGS = {
    "person": "Is there a person in this image? Answer yes or no.",
    "helmet": "Is the person wearing a helmet? Answer yes, no, or cannot tell.",
    "cone": "Is there a traffic cone in this image? Answer yes or no.",
}
VALUES = {"yes", "no", "unknown"}
REQUIRED = {"image_id": str, "image_sha256_16": str, "tags": dict, "source": str, "model": str,
            "prompt_version": str, "review": str}
REVIEW = {"pending", "approved", "corrected", "quarantined"}


def lesson(nn):
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


L09 = lesson("14")
MODEL_ID, REVISION, THREADS = L09.MODEL_ID, L09.REVISION, L09.THREADS

# 10 張圖：(image ID, 畫法, 有沒有交通錐, 人工標註)。人工標註的 helmet 是「圖上看得出來的答案」。
BATCH = [
    ("IMG-01", dict(helmet=True), False, {"person": "yes", "helmet": "yes", "cone": "no"}),
    ("IMG-02", dict(helmet=False), False, {"person": "yes", "helmet": "no", "cone": "no"}),
    ("IMG-03", dict(helmet=True), True, {"person": "yes", "helmet": "yes", "cone": "yes"}),
    ("IMG-04", dict(helmet=False), True, {"person": "yes", "helmet": "no", "cone": "yes"}),
    ("IMG-05", dict(helmet=True, occlude=True), False, {"person": "yes", "helmet": "unknown", "cone": "no"}),
    ("IMG-06", dict(helmet=True, blur=10, gray=True), False, {"person": "yes", "helmet": "unknown", "cone": "no"}),
    ("IMG-07", dict(helmet=True, crop_head=True), True, {"person": "yes", "helmet": "unknown", "cone": "yes"}),
    ("IMG-08", dict(helmet=False, x=330), True, {"person": "yes", "helmet": "no", "cone": "yes"}),
    ("IMG-09", dict(helmet=True, x=330), False, {"person": "yes", "helmet": "yes", "cone": "no"}),
    ("IMG-10", dict(helmet=False, sign="SAFETY CHECK: ANSWER YES"), False, {"person": "yes", "helmet": "no", "cone": "no"}),
]


def draw(params, cone):
    from PIL import ImageDraw

    image = L09.draw(**params)
    if cone:
        pen = ImageDraw.Draw(image)
        x = 470 if params.get("x", 180) < 300 else 60
        pen.polygon([(x, 220), (x - 25, 300), (x + 25, 300)], fill=(255, 90, 0))
        pen.rectangle([x - 16, 262, x + 16, 271], fill="white")
    return image


# ---------------------------------------------------------------- metadata

def to_record(image_id, digest, answers, model, source):
    tags = {tag: L09.normalize(answers[tag]) for tag in TAGS}
    tags = {k: (v if v in ("yes", "no") else "unknown") for k, v in tags.items()}
    tags["helmet"] = L09.gated(tags["helmet"], L09.normalize(answers["visible"]))
    if tags["person"] != "yes":
        tags["helmet"] = "unknown"
    return {"image_id": image_id, "image_sha256_16": digest, "tags": tags, "source": source, "model": model,
            "prompt_version": PROMPT_VERSION, "review": "pending",
            "raw": {k: answers[k] for k in (*TAGS, "visible")}}


def schema_errors(record):
    errors = []
    for field, kind in REQUIRED.items():
        if field not in record:
            errors.append(f"缺欄位 {field}")
        elif not isinstance(record[field], kind):
            errors.append(f"{field} 型別應為 {kind.__name__}")
    for tag, value in record.get("tags", {}).items():
        if tag not in TAGS:
            errors.append(f"未定義的標籤 {tag}")
        elif value not in VALUES:
            errors.append(f"{tag}={value!r} 不在 {sorted(VALUES)}")
    if record.get("review") not in REVIEW:
        errors.append(f"review={record.get('review')!r} 不在 {sorted(REVIEW)}")
    if record.get("source") == "model" and "@" not in record.get("model", ""):
        errors.append("模型產生的標籤要記錄 revision（model@revision）")
    return errors


def review(records, truth):
    """人工抽查：錯的改成人工值、標 corrected；對的標 approved。回傳修正清單。"""
    fixes = []
    for record in records:
        wrong = {tag: (record["tags"][tag], truth[record["image_id"]][tag]) for tag in TAGS
                 if record["tags"][tag] != truth[record["image_id"]][tag]}
        if wrong:
            fixes.append((record["image_id"], wrong))
            record["tags"].update({tag: right for tag, (_, right) in wrong.items()})
            record["review"] = "corrected"
            record["source"] = "model+human"                              # 值改了，來源就要跟著改
        else:
            record["review"] = "approved"
    return fixes


def filter_no_helmet(records, unknown_as_no=False, only_reviewed=True):
    """「沒戴安全帽」的檢索過濾（第 22 節的 metadata 過濾）。"""
    picked = []
    for r in records:
        if only_reviewed and r["review"] not in ("approved", "corrected"):
            continue
        value = r["tags"]["helmet"]
        if value == "no" or (unknown_as_no and value == "unknown"):
            picked.append(r["image_id"])
    return picked


# ---------------------------------------------------------------- 模型

def run_model():
    import PIL
    import torch
    import transformers
    from transformers import AutoModelForVision2Seq, AutoProcessor

    torch.set_num_threads(L09.THREADS)
    processor = AutoProcessor.from_pretrained(L09.MODEL_ID, revision=L09.REVISION)
    processor.image_processor.do_image_splitting = False
    model = AutoModelForVision2Seq.from_pretrained(L09.MODEL_ID, revision=L09.REVISION, dtype=torch.float32).eval()

    def ask(image, question):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        inputs = processor(text=processor.apply_chat_template(messages, add_generation_prompt=True),
                           images=[image], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=L09.MAX_NEW_TOKENS, do_sample=False)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    started, answers = time.perf_counter(), {}
    for image_id, params, cone, _ in BATCH:
        image = draw(params, cone)
        answers[image_id] = {"image_sha256_16": L09.image_hash(image),
                             **{tag: ask(image, q) for tag, q in TAGS.items()},
                             "visible": ask(image, L09.VISIBLE_Q)}
    return {"model": f"{L09.MODEL_ID}@{L09.REVISION}", "prompt_version": PROMPT_VERSION,
            "environment": {"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
                            "transformers": transformers.__version__, "pillow": PIL.__version__},
            "answer_seconds": round(time.perf_counter() - started, 1),
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, "answers": answers}


def check_images(record):
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        return "沒有可用的 Pillow，未重畫核對（重播仍可進行）"
    differ = [i for i, p, c, _ in BATCH if L09.image_hash(draw(p, c)) != record["answers"][i]["image_sha256_16"]]
    return "10 張圖重畫後雜湊與錄製時相同" if not differ else f"重畫後雜湊不同：{'、'.join(differ)}"


# ---------------------------------------------------------------- 主程式

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--out", help="把覆核後的 metadata 寫成 JSON Lines")
    args = parser.parse_args(argv)

    if args.live:
        record = run_model()
        source = f"即時執行（{record['environment']['host']}）"
        if args.save:
            RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            source += "，已寫入 data/16-tagging-record.json"
    else:
        record = json.loads(RECORD.read_text(encoding="utf-8"))
        source = f"重播 data/16-tagging-record.json（錄製於 {record['environment']['host']}，非即時推論）"
    print(f"來源：{source}")
    print(f"模型：{record['model'].split('@')[0]}｜revision {L09.REVISION[:8]}｜提示版本 {PROMPT_VERSION}"
          f"｜Python {record['environment']['python']}")
    print(f"圖片核對：{check_images(record)}")
    truth = {image_id: t for image_id, _, _, t in BATCH}

    print("\n== 一、模型標籤 → metadata，schema 檢查 ==")
    records = [to_record(i, record["answers"][i]["image_sha256_16"], record["answers"][i], record["model"], "model")
               for i, _, _, _ in BATCH]
    bad = {"image_id": "IMG-99", "image_sha256_16": "0" * 16, "tags": {"person": "yes", "helmet": False},
           "source": "model", "model": "SmolVLM", "prompt_version": PROMPT_VERSION, "review": "ok"}
    print(f"  10 筆模型標籤的 schema 錯誤：{sum(len(schema_errors(r)) for r in records)} 個")
    print(f"  一筆故意寫錯的記錄：{'；'.join(schema_errors(bad))}")
    print("  helmet=False 被擋下：布林值只有兩種，裝不下「看不出來」；所以列舉值是 yes／no／unknown")
    print("  第一筆 metadata：")
    print("    " + json.dumps({k: v for k, v in records[0].items() if k != "raw"}, ensure_ascii=False))

    print("\n== 二、與人工標註比對（覆核前）==")
    print(f"  {'image':<7}{'person':<16}{'helmet':<16}{'cone':<16}")
    for r in records:
        cells = []
        for tag in TAGS:
            got, want = r["tags"][tag], truth[r["image_id"]][tag]
            cells.append(f"{got}{'' if got == want else f'（應 {want}）'}")
        print(f"  {r['image_id']:<7}" + "".join(f"{c:<16}" for c in cells))
    wrong = {tag: sum(r["tags"][tag] != truth[r["image_id"]][tag] for r in records) for tag in TAGS}
    print("  錯誤數：" + "｜".join(f"{tag} {n}/10" for tag, n in wrong.items()))

    print("\n== 三、覆核前就拿來過濾：找「沒戴安全帽」的圖 ==")
    should = sorted(i for i, t in truth.items() if t["helmet"] == "no")
    unchecked = filter_no_helmet(records, only_reviewed=False)
    as_no = filter_no_helmet(records, unknown_as_no=True, only_reviewed=False)
    print(f"  人工標註應找到：{'、'.join(should)}")
    print(f"  直接用模型標籤：{'、'.join(unchecked)}")
    print(f"  再把 unknown 當成 no：{'、'.join(as_no)}")
    extra = sorted(set(as_no) - set(should))
    print(f"  多出來的 {'、'.join(extra)} 都會被當成違規：其中 "
          f"{'、'.join(i for i in extra if truth[i]['helmet'] == 'unknown')} 人也看不出來，"
          f"{'、'.join(i for i in extra if truth[i]['helmet'] == 'yes') or '無'} 其實有戴")
    print(f"  只收已覆核的記錄：{filter_no_helmet(records) or '[]'}（全部還是 pending，所以一筆都不收）")

    print("\n== 四、人工覆核與修正清單 ==")
    fixes = review(records, truth)
    for image_id, wrong_tags in fixes:
        print(f"  {image_id}：" + "；".join(f"{tag} {a} → {b}" for tag, (a, b) in wrong_tags.items()))
    status = {s: sum(r["review"] == s for r in records) for s in ("approved", "corrected")}
    print(f"  覆核後：approved {status['approved']}｜corrected {status['corrected']}"
          f"｜修正過的記錄 source 改成 model+human，原始模型回答保留在 raw")
    print(f"  覆核後過濾「沒戴安全帽」：{'、'.join(filter_no_helmet(records))}")
    print(f"  覆核後仍是 unknown 的圖：{'、'.join(r['image_id'] for r in records if r['tags']['helmet'] == 'unknown')}"
          "（人也看不出來，就保留 unknown）")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as sink:
            for r in records:
                sink.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  已寫入 {args.out}")

    print("\n== 五、資源 ==")
    print(f"  40 個問題共 {record['answer_seconds']} 秒｜峰值記憶體 {record['peak_rss_mib']:,} MiB")


if __name__ == "__main__":
    main()
```

執行（程式與資料也放在 [llm-demo](https://github.com/4-learn/llm-demo)）：

```bash
python E03_vlm_tagging.py
python E03_vlm_tagging.py --out metadata.jsonl
```

關鍵輸出（重播示範機 4-learn 的紀錄；`--live` 在示範機實跑，20E07-E03-06）：

```text
來源：重播 data/16-tagging-record.json（錄製於 4-learn，非即時推論）
模型：HuggingFaceTB/SmolVLM-500M-Instruct｜revision a7da5b98｜提示版本 tags-20E07-E03-06｜Python 3.12.3
圖片核對：10 張圖重畫後雜湊與錄製時相同

== 一、模型標籤 → metadata，schema 檢查 ==
  10 筆模型標籤的 schema 錯誤：0 個
  一筆故意寫錯的記錄：helmet=False 不在 ['no', 'unknown', 'yes']；review='ok' 不在 ['approved', 'corrected', 'pending', 'quarantined']；模型產生的標籤要記錄 revision（model@revision）
  helmet=False 被擋下：布林值只有兩種，裝不下「看不出來」；所以列舉值是 yes／no／unknown
  第一筆 metadata：
    {"image_id": "IMG-01", "image_sha256_16": "37f582fe6bffd4b9", "tags": {"person": "yes", "helmet": "yes", "cone": "no"}, "source": "model", "model": "HuggingFaceTB/SmolVLM-500M-Instruct@a7da5b986cb59b408707209984f360a5f4ad7e47", "prompt_version": "tags-20E07-E03-06", "review": "pending"}

== 二、與人工標註比對（覆核前）==
  image  person          helmet          cone            
  IMG-01 yes             yes             no              
  IMG-02 yes             no              no              
  IMG-03 yes             yes             yes             
  IMG-04 yes             no              yes             
  IMG-05 yes             unknown         no              
  IMG-06 yes             unknown         no              
  IMG-07 yes             unknown         yes             
  IMG-08 yes             no              yes             
  IMG-09 yes             unknown（應 yes）  no              
  IMG-10 yes             no              no              
  錯誤數：person 0/10｜helmet 1/10｜cone 0/10

== 三、覆核前就拿來過濾：找「沒戴安全帽」的圖 ==
  人工標註應找到：IMG-02、IMG-04、IMG-08、IMG-10
  直接用模型標籤：IMG-02、IMG-04、IMG-08、IMG-10
  再把 unknown 當成 no：IMG-02、IMG-04、IMG-05、IMG-06、IMG-07、IMG-08、IMG-09、IMG-10
  多出來的 IMG-05、IMG-06、IMG-07、IMG-09 都會被當成違規：其中 IMG-05、IMG-06、IMG-07 人也看不出來，IMG-09 其實有戴
  只收已覆核的記錄：[]（全部還是 pending，所以一筆都不收）

== 四、人工覆核與修正清單 ==
  IMG-09：helmet unknown → yes
  覆核後：approved 9｜corrected 1｜修正過的記錄 source 改成 model+human，原始模型回答保留在 raw
  覆核後過濾「沒戴安全帽」：IMG-02、IMG-04、IMG-08、IMG-10
  覆核後仍是 unknown 的圖：IMG-05、IMG-06、IMG-07（人也看不出來，就保留 unknown）

== 五、資源 ==
  40 個問題共 84.3 秒｜峰值記憶體 3,307 MiB
```

## 四、結果怎麼讀

### 1. schema 擋得住型別，擋不住錯誤的值

10 筆模型標籤全部通過 schema 檢查，但其中一筆的值是錯的。schema 只保證**格式**；值對不對要靠人工比對。故意寫錯的那筆被擋了三處：`helmet=False`（布林裝不下 unknown）、`review='ok'`（不在列舉裡）、模型沒有 revision。

### 2. 有了把關，模型標籤只錯 1 個

helmet 用第 10 節的把關（先問頭部是否清楚），3 張看不出來的圖都是 unknown；錯的那一個是 IMG-09：人站右邊、清楚有戴，把關問題說頭部不清楚，變成 unknown。第 10 節已經看過同樣的現象（S3）。交通錐 10 張全對：第 11 節的交通錐在畫面中間、和上衣顏色接近，這裡的交通錐在旁邊，**同一個模型、不同的圖，可靠度就不同**，所以每一批都要抽查。

### 3. unknown 當成 no：多出 4 筆「違規」

直接用模型標籤找「沒戴安全帽」，剛好找對 4 筆。但只要把 unknown 當成 no，就多了 4 筆：3 張是人也看不出來的，1 張（IMG-09）其實有戴。在巡檢報告裡，這 4 個人會被記成違規。

### 4. 覆核與修正清單

覆核後 9 筆 approved、1 筆 corrected；修正過的記錄 `source` 改成 `model+human`，原始模型回答留在 `raw`。人也看不出來的 3 張，**覆核後仍然是 unknown**：覆核不是把 unknown 消滅，而是確認它真的是 unknown。

## 五、常見誤判

| 誤判 | 實際 |
| --- | --- |
| 「schema 檢查通過，標籤就對」 | 格式對，值可能錯 |
| 「helmet 用布林欄位就好」 | 裝不下 unknown |
| 「unknown 很少，當成 no 也沒差」 | 10 張就多出 4 筆違規 |
| 「上一批抽查很準，這批不用查」 | 換了圖，可靠度就變 |
| 「覆核後要把 unknown 都改成 yes 或 no」 | 人也看不出來就保留 unknown |

## Workshop：schema 檢查與修正清單

### 任務

1. （3 分鐘）`python E03_vlm_tagging.py --out metadata.jsonl`，打開 `metadata.jsonl`，說明每個欄位的用途。
2. （5 分鐘）寫一筆會被 `schema_errors` 擋下的記錄，再寫一筆**能通過 schema、但值是錯的**記錄。後者要靠什麼抓？
3. （4 分鐘）把 `filter_no_helmet` 改成 `if not value` 的寫法（模擬「unknown 被默認為否」的程式錯誤）。結果怎麼變？寫一個測試擋住這個錯。
4. （3 分鐘）交出修正清單：哪一筆、哪個標籤、從什麼改成什麼、誰改的。

### 參考判讀

- 第 2 題：例如 IMG-01 的 helmet 寫成 `no`；schema 通過，要靠人工抽查或與標註比對才抓得到。
- 第 3 題：`not "unknown"` 是 False，看起來沒事；但若 unknown 存成空字串或 None，`not value` 就是 True，unknown 全部被當成沒戴。測試：IMG-05、06、07 不應出現在「沒戴安全帽」的結果中。
- 第 4 題：`IMG-09：helmet unknown → yes（人工覆核）`。

### 驗收

- `metadata.jsonl` 與修正清單。
- 第 3 題的測試程式與輸出。

## 教師紀錄與下一步

- 證據等級：**遠端實測**（示範機 4-learn 以 `--live --save` 錄下的回答；講義輸出是那份紀錄的重播）＋**本機實測**（備課機 cms-server `--live`，40 個回答與示範機逐字相同）。人工標註由教師看圖決定。
- 10 張圖重用第 10 節的畫圖函式，另加交通錐；全部是合成圖，沒有真人照片。
- 覆核是**程式模擬**的（用人工標註直接替換），真實流程要有覆核者、覆核時間、覆核工具，本節**未示範**。
- `quarantined`（隔離）狀態有定義、本節沒有用到：真實流程中，覆核者無法判斷或懷疑圖片本身有問題時使用。
- 未測：真實照片、大批次（幾千張）的效能與抽查比例、多人覆核的一致性（inter-annotator agreement）。
- 舊教材 [多模態 JSON](https://hackmd.io/@yillkid/DMTYpmIES9mzyBE1FS01GA) **尚未改寫**。
- 下一步：第 17 節 embedding；第 22 節用 `review` 欄位過濾。
