"""第 10 節：可核對的圖像描述。

用程式畫一張合成工地圖，**每個物件的位置都是已知的**（畫的時候就記下框）。讓本地 SmolVLM-500M：
  一、自由描述一句話：逐項拆成主張，對照標註，標「有證據／圖上沒有／錯」
  二、回答結構化問題（有沒有、幾個、左右關係、告示寫什麼），程式自動對答案
  三、把圖切成左、中、右三塊，分別問「這塊有什麼」：區域證據
  四、問它「你有多確定（0–100）」：模型自述的信心和答對與否有沒有關係

預設重播示範機 4-learn 錄下的回答（標準函式庫；有 Pillow 會重畫核對雜湊）；`--live` 實跑。

    python 10_grounded_caption.py
    python 10_grounded_caption.py --live [--save]
    python 10_grounded_caption.py --image scene.png      # 把圖存下來（需要 Pillow），自己圈選核對
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
RECORD = HERE / "data" / "10-caption-record.json"
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
REVISION = "a7da5b986cb59b408707209984f360a5f4ad7e47"
THREADS = 4
SIZE = (640, 400)

# 標註：畫圖時就知道的事實（x0, y0, x1, y1）。這是本節的「圖像證據」。
OBJECTS = {
    "person": {"box": (84, 92, 156, 310), "detail": "黃色安全帽、橘色上衣、深色褲子"},
    "traffic cone": {"box": (290, 220, 350, 310), "detail": "橘色、白色條紋"},
    "ladder": {"box": (500, 120, 580, 310), "detail": "棕色"},
    "sign": {"box": (250, 40, 400, 100), "detail": "綠底白字 EXIT"},
}
ABSENT = ("dog", "forklift")
REGIONS = {"left": (0, 0, 213, 400), "middle": (213, 0, 426, 400), "right": (426, 0, 640, 400)}

# 結構化問題：(id, 問題, 正確答案, 判讀方式)
QUESTIONS = [
    ("has_person", "Is there a person in this image? Answer yes or no.", "yes", "yesno"),
    ("has_cone", "Is there a traffic cone in this image? Answer yes or no.", "yes", "yesno"),
    ("has_ladder", "Is there a ladder in this image? Answer yes or no.", "yes", "yesno"),
    ("has_dog", "Is there a dog in this image? Answer yes or no.", "no", "yesno"),
    ("has_forklift", "Is there a forklift in this image? Answer yes or no.", "no", "yesno"),
    ("count_people", "How many people are in this image? Answer with a number.", "1", "number"),
    ("person_side", "Is the person on the left or the right side of the image? Answer left or right.", "left", "leftright"),
    ("cone_vs_ladder", "Is the traffic cone to the left or to the right of the ladder? Answer left or right.", "left", "leftright"),
    ("sign_text", "What word is written on the sign? Answer with the word only.", "exit", "word"),
]
CONFIDENCE_Q = "{q} Then on a new line give your confidence from 0 to 100."


# ---------------------------------------------------------------- 畫圖

def draw():
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", SIZE, (200, 220, 240))
    pen = ImageDraw.Draw(image)
    pen.rectangle([0, 300, 640, 400], fill=(150, 130, 100))
    x = 120
    pen.ellipse([x - 25, 110, x + 25, 160], fill=(240, 200, 170))
    pen.chord([x - 32, 92, x + 32, 142], 180, 360, fill=(255, 210, 0))
    pen.rectangle([x - 36, 116, x + 36, 122], fill=(255, 210, 0))
    pen.rectangle([x - 35, 160, x + 35, 250], fill=(255, 120, 0))
    pen.rectangle([x - 30, 250, x - 8, 310], fill=(40, 40, 60))
    pen.rectangle([x + 8, 250, x + 30, 310], fill=(40, 40, 60))
    pen.polygon([(320, 220), (290, 310), (350, 310)], fill=(255, 90, 0))
    pen.rectangle([300, 270, 340, 280], fill="white")
    pen.rectangle([500, 120, 510, 310], fill=(120, 80, 40))
    pen.rectangle([570, 120, 580, 310], fill=(120, 80, 40))
    for y in range(140, 310, 30):
        pen.rectangle([510, y, 570, y + 6], fill=(120, 80, 40))
    pen.rectangle([250, 40, 400, 100], fill=(0, 140, 60))
    pen.text((280, 55), "EXIT", fill="white", font=ImageFont.load_default(size=32))
    return image


def image_hash(image):
    return hashlib.sha256(image.tobytes()).hexdigest()[:16]


def objects_in(region):
    """標註裡，框的中心落在這個區域的物件。"""
    x0, _, x1, _ = REGIONS[region]
    return sorted(name for name, o in OBJECTS.items() if x0 <= (o["box"][0] + o["box"][2]) / 2 < x1)


# ---------------------------------------------------------------- 判讀

def parse(text, kind):
    head = text.strip().lower()
    if kind == "yesno":
        return "yes" if head.startswith("yes") else "no" if head.startswith("no") else "other"
    if kind == "number":
        found = re.search(r"\d+", head)
        return found.group() if found else "other"
    if kind == "leftright":
        has_l, has_r = "left" in head, "right" in head
        return "left" if has_l and not has_r else "right" if has_r and not has_l else "other"
    return re.sub(r"[^a-z]", "", head.split()[0]) if head else "other"


def confidence(text):
    numbers = re.findall(r"\b(\d{1,3})\b", text.split("\n", 1)[1] if "\n" in text else "")
    return int(numbers[-1]) if numbers else None


# 自由描述的核對：教師把示範機錄下的那一句拆成主張，對照標註判讀（教師 trace，不是自動判讀）
CAPTION_CLAIMS = [
    ("a person", "有證據", "person 框（左側）"),
    ("wearing a yellow hard hat", "有證據", "person 框頭部的黃色半圓"),
    ("a red and orange striped shirt", "圖上沒有", "上衣是單一橘色，沒有條紋；紅色也沒有"),
    ("standing on a brown surface", "有證據", "地面是棕色"),
    ("with a green exit sign above them", "不精確", "sign 在人的右上方（畫面中上），不在頭頂上"),
    ("（沒有提到交通錐與梯子）", "遺漏", "traffic cone、ladder 都在圖上"),
]


# ---------------------------------------------------------------- 模型

def run_model():
    import PIL
    import torch
    import transformers
    from transformers import AutoModelForVision2Seq, AutoProcessor

    torch.set_num_threads(THREADS)
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    processor.image_processor.do_image_splitting = False
    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.float32).eval()
    load_seconds = time.perf_counter() - started

    def ask(image, question, tokens):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        inputs = processor(text=processor.apply_chat_template(messages, add_generation_prompt=True),
                           images=[image], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=tokens, do_sample=False)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    image, started = draw(), time.perf_counter()
    answers = {
        "caption": ask(image, "Describe this image in one sentence.", 60),
        "objects": ask(image, "List the objects you can see in this image, separated by commas.", 40),
        "questions": {qid: ask(image, q, 16) for qid, q, _, _ in QUESTIONS},
        "confidence": {qid: ask(image, CONFIDENCE_Q.format(q=q), 24) for qid, q, _, _ in QUESTIONS},
        "regions": {name: ask(image.crop(box), "What objects are in this image? Answer in a few words.", 16)
                    for name, box in REGIONS.items()},
    }
    return {
        "model": f"{MODEL_ID}@{REVISION}",
        "settings": {"threads": THREADS, "do_sample": False, "do_image_splitting": False, "dtype": "float32"},
        "environment": {"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "pillow": PIL.__version__},
        "image_sha256_16": image_hash(image),
        "load_seconds": round(load_seconds, 1), "answer_seconds": round(time.perf_counter() - started, 1),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
        "answers": answers,
    }


def check_image(record):
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        return "沒有可用的 Pillow，未重畫核對（重播仍可進行）"
    same = image_hash(draw()) == record["image_sha256_16"]
    return "重畫後雜湊與錄製時相同" if same else "重畫後雜湊不同（Pillow 版本或字型不同）"


# ---------------------------------------------------------------- 主程式

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--image", help="把圖存成這個檔名（需要 Pillow）")
    args = parser.parse_args(argv)
    if args.image:
        draw().save(args.image)
        print(f"已存圖：{args.image}（{SIZE[0]}×{SIZE[1]}）")

    if args.live:
        record = run_model()
        source = f"即時執行（{record['environment']['host']}）"
        if args.save:
            RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            source += "，已寫入 data/10-caption-record.json"
    else:
        record = json.loads(RECORD.read_text(encoding="utf-8"))
        source = f"重播 data/10-caption-record.json（錄製於 {record['environment']['host']}，非即時推論）"
    env, ans = record["environment"], record["answers"]
    print(f"來源：{source}")
    print(f"模型：{MODEL_ID}｜revision {REVISION[:8]}｜greedy｜不切塊｜Python {env['python']}｜Pillow {env['pillow']}")
    print(f"圖片核對：{check_image(record)}")
    print("標註（畫圖時記下的框）：" + "｜".join(f"{n} {o['box']}" for n, o in OBJECTS.items()))

    print("\n== 一、自由描述 ==")
    print(f"  模型：「{ans['caption']}」")
    if args.live and ans["caption"] != json.loads(RECORD.read_text(encoding="utf-8"))["answers"]["caption"]:
        print("  （這次的描述與紀錄不同，下面的教師判讀不適用，請自己逐項核對）")
    for claim, verdict, why in CAPTION_CLAIMS:
        print(f"  {verdict:<5}{claim}：{why}")
    print(f"  列出物件：「{' '.join(ans['objects'].split())}」")
    listed = ans["objects"].lower()
    print(f"    標註 {len(OBJECTS)} 個，列出 {sum(n.split()[-1] in listed for n in OBJECTS)} 個；"
          f"沒列到：{'、'.join(n for n in OBJECTS if n.split()[-1] not in listed) or '無'}")

    print("\n== 二、結構化問題（程式自動對答案）==")
    right = 0
    for qid, _, truth, kind in QUESTIONS:
        got = parse(ans["questions"][qid], kind)
        right += got == truth
        print(f"  {'✓' if got == truth else '✗'} {qid:<15} 答「{ans['questions'][qid]}」→ {got:<6} 正解 {truth}")
    print(f"  {right}/{len(QUESTIONS)} 題正確")

    print("\n== 三、區域證據（切成三塊分別問）==")
    for name in REGIONS:
        print(f"  {name:<6} {REGIONS[name]}｜標註 {', '.join(objects_in(name))}｜模型「{ans['regions'][name]}」")

    print("\n== 四、模型自述的信心 ==")
    pairs = []
    for qid, _, truth, kind in QUESTIONS:
        text = ans["confidence"][qid]
        got, conf = parse(text.split("\n")[0], kind), confidence(text)
        pairs.append((got == truth, conf))
        print(f"  {qid:<15} {'對' if got == truth else '錯'}｜信心 {conf if conf is not None else '（沒給數字）'}"
              f"｜原文「{' '.join(text.split())}」")
    given = [c for _, c in pairs if c is not None]
    wrong_conf = [c for ok, c in pairs if not ok and c is not None]
    print(f"  有給數字 {len(given)}/{len(pairs)} 題；答錯的題目信心："
          f"{'、'.join(map(str, wrong_conf)) or '（答錯的都沒給數字）'}")
    changed = [qid for qid, _, _, kind in QUESTIONS
               if parse(ans["confidence"][qid].split("\n")[0], kind) != parse(ans["questions"][qid], kind)]
    print(f"  只是在問題後面加一句「給信心分數」，答案就改變的題目：" + ("、".join(
        f"{qid}（{parse(ans['questions'][qid], k)}→{parse(ans['confidence'][qid].split(chr(10))[0], k)}）"
        for qid, _, _, k in QUESTIONS if qid in changed) or "無"))
    print("  自述信心是模型生成的文字，不是校準過的機率；要知道準不準，得用有標註的題目量")

    print("\n== 五、資源 ==")
    print(f"  載入 {record['load_seconds']} 秒｜{2 + 2 * len(QUESTIONS) + len(REGIONS)} 個問題共 "
          f"{record['answer_seconds']} 秒｜峰值記憶體 {record['peak_rss_mib']:,} MiB")


if __name__ == "__main__":
    main()
