"""第 15 節：VLM 打標與 Metadata 品質。

一小批 10 張合成工地圖（重用第 09 節的畫圖函式），本地 SmolVLM-500M 對每張回答 3 個標籤問題：
有沒有人、有沒有戴安全帽（看不清楚時由第 09 節的把關改成 unknown）、有沒有交通錐。
把回答轉成 metadata 記錄：每一筆帶 image ID、雜湊、產生方式、模型與 revision、提示版本、覆核狀態。
接著：
  一、schema 檢查：欄位、型別、列舉值；unknown 是合法值，不能被當成 false
  二、與人工標註比對：哪些標籤錯了、錯的方向
  三、抽查與隔離：錯誤與未覆核的標籤不能進入檢索過濾（第 21 節）
  四、示範「unknown 被默認為否」會讓過濾結果怎麼錯

預設重播示範機 4-learn 錄下的回答（需要同資料夾第 09 節的程式）；`--live` 實跑。

    python 15_vlm_tagging.py
    python 15_vlm_tagging.py --live [--save]
    python 15_vlm_tagging.py --out metadata.jsonl      # 把 metadata 寫成 JSON Lines
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
RECORD = HERE / "data" / "15-tagging-record.json"
PROMPT_VERSION = "tags-2026-10-06"
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


L09 = lesson("09")

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
    """「沒戴安全帽」的檢索過濾（第 21 節的 metadata 過濾）。"""
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
            source += "，已寫入 data/15-tagging-record.json"
    else:
        record = json.loads(RECORD.read_text(encoding="utf-8"))
        source = f"重播 data/15-tagging-record.json（錄製於 {record['environment']['host']}，非即時推論）"
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
