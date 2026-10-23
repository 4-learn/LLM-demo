# 可核對的圖像描述

> **正文改版：20E07-E03-06｜對應新版 LLM 第 11 節：可核對的圖像描述。**
> 第 10 節處理「看不出來」。本節處理「說得出來，但說的對不對」。一段流暢的圖像描述裡，常常混著看得到的、看不到的與位置錯的內容。本節用程式畫一張合成工地圖，**每個物件的框在畫圖時就記下來**，於是每句話都能回到圖上核對。讓本地 SmolVLM-500M 自由描述、回答結構化問題、分區域描述，最後請它自報信心分數。核心主張：**描述要拆成可以回到圖上圈選的主張；結構化問題比自由描述好核對；模型自述的信心不是機率。**

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先修第 10 節（evidence／inference／unknown、重播與即時的標示）。

| 時間 | 內容 |
| --- | --- |
| 0–10 分鐘 | 為什麼流暢的描述難核對：主張、位置、遺漏 |
| 10–20 分鐘 | 限定範圍：結構化問題、區域指示 |
| 20–35 分鐘 | 跑 Demo；自述信心 vs 校準過的機率 |
| 35–50 分鐘 | Workshop：在圖上圈出每個主張的證據，刪掉沒有證據的句子 |

驗收能力：交出一份**每句都標了圖上區域的描述**，以及一項誤判分析；不能把「看不到安全帽」寫成「確定沒戴」。

## 一、一句描述裡有幾個主張

「一個戴黃色安全帽的人站在棕色地面上，頭上有綠色出口標誌」包含五個主張：有一個人、戴黃色安全帽、站在棕色地面、有綠色出口標誌、標誌在人的頭上。每一個都要能在圖上**圈出來**，圈不出來就刪掉或改寫。還要檢查**遺漏**：圖上有、但描述沒提的東西（對巡檢來說，漏掉的交通錐可能比多說一句還嚴重）。

| 判讀 | 意思 | 處理 |
| --- | --- | --- |
| 有證據 | 能在圖上圈出來 | 保留，附區域 |
| 不精確 | 物件存在，但位置、數量或屬性說錯 | 改寫成圖上看得到的樣子 |
| 圖上沒有 | 圈不出來 | 刪除 |
| 遺漏 | 圖上有，描述沒提 | 補上，或明寫「描述不完整」 |

## 二、限定範圍讓核對變簡單

| 問法 | 核對方式 | 代價 |
| --- | --- | --- |
| 自由描述 | 人工逐句拆、逐句圈 | 慢；判讀因人而異 |
| 結構化問題（有沒有、幾個、左右、寫什麼） | 程式對答案 | 只問到你想到的問題 |
| 區域指示（切塊再問） | 每塊對照該區標註 | 物件被切到一半會出錯 |

真實照片沒有「畫圖時記下的框」，標註要由人來做（圈框、標物件）。這正是第 16 節 metadata 的工作；本節用合成圖，先把核對流程練熟。

## 三、自述信心不是機率

模型說「信心 90」只是生成的文字，和答對的機率沒有必然關係。**校準**（calibration）的意思是：說 90% 的那些題目，真的有大約 90% 答對；這要用有標註的題目量才知道。本節的小模型連數字都不給。

## 四、可執行 Demo

重播只需要標準函式庫（有 Pillow 會重畫核對雜湊）；`--live` 需求與第 10 節相同。`--image scene.png` 把圖存下來給你圈選。

<!-- demo: E01_grounded_caption.py -->
```python
"""第 11 節：可核對的圖像描述。

用程式畫一張合成工地圖，**每個物件的位置都是已知的**（畫的時候就記下框）。讓本地 SmolVLM-500M：
  一、自由描述一句話：逐項拆成主張，對照標註，標「有證據／圖上沒有／錯」
  二、回答結構化問題（有沒有、幾個、左右關係、告示寫什麼），程式自動對答案
  三、把圖切成左、中、右三塊，分別問「這塊有什麼」：區域證據
  四、問它「你有多確定（0–100）」：模型自述的信心和答對與否有沒有關係

預設重播示範機 4-learn 錄下的回答（標準函式庫；有 Pillow 會重畫核對雜湊）；`--live` 實跑。

    python E01_grounded_caption.py
    python E01_grounded_caption.py --live [--save]
    python E01_grounded_caption.py --image scene.png      # 把圖存下來（需要 Pillow），自己圈選核對
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
RECORD = HERE / "data" / "11-caption-record.json"
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
            source += "，已寫入 data/11-caption-record.json"
    else:
        record = json.loads(RECORD.read_text(encoding="utf-8"))
        source = f"重播 data/11-caption-record.json（錄製於 {record['environment']['host']}，非即時推論）"
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
```

執行（程式與資料也放在 [llm-demo](https://github.com/4-learn/llm-demo)）：

```bash
python E01_grounded_caption.py
python E01_grounded_caption.py --image scene.png
```

關鍵輸出（重播示範機 4-learn 的紀錄；`--live` 在示範機實跑，20E07-E03-06）：

```text
來源：重播 data/11-caption-record.json（錄製於 4-learn，非即時推論）
模型：HuggingFaceTB/SmolVLM-500M-Instruct｜revision a7da5b98｜greedy｜不切塊｜Python 3.12.3｜Pillow 12.3.0
圖片核對：重畫後雜湊與錄製時相同
標註（畫圖時記下的框）：person (84, 92, 156, 310)｜traffic cone (290, 220, 350, 310)｜ladder (500, 120, 580, 310)｜sign (250, 40, 400, 100)

== 一、自由描述 ==
  模型：「A simple, cartoon-style drawing of a person wearing a yellow hard hat and a red and orange striped shirt, standing on a brown surface with a green exit sign above them.」
  有證據  a person：person 框（左側）
  有證據  wearing a yellow hard hat：person 框頭部的黃色半圓
  圖上沒有 a red and orange striped shirt：上衣是單一橘色，沒有條紋；紅色也沒有
  有證據  standing on a brown surface：地面是棕色
  不精確  with a green exit sign above them：sign 在人的右上方（畫面中上），不在頭頂上
  遺漏   （沒有提到交通錐與梯子）：traffic cone、ladder 都在圖上
  列出物件：「1. Person 2. Ladder 3. Sign 4. Ground」
    標註 4 個，列出 3 個；沒列到：traffic cone

== 二、結構化問題（程式自動對答案）==
  ✓ has_person      答「Yes.」→ yes    正解 yes
  ✗ has_cone        答「No.」→ no     正解 yes
  ✓ has_ladder      答「Yes.」→ yes    正解 yes
  ✓ has_dog         答「No.」→ no     正解 no
  ✓ has_forklift    答「No.」→ no     正解 no
  ✗ count_people    答「3.」→ 3      正解 1
  ✓ person_side     答「Left.」→ left   正解 left
  ✓ cone_vs_ladder  答「Left.」→ left   正解 left
  ✓ sign_text       答「EXIT.」→ exit   正解 exit
  7/9 題正確

== 三、區域證據（切成三塊分別問）==
  left   (0, 0, 213, 400)｜標註 person｜模型「Person.」
  middle (213, 0, 426, 400)｜標註 sign, traffic cone｜模型「Exit sign.」
  right  (426, 0, 640, 400)｜標註 ladder｜模型「Ladder.」

== 四、模型自述的信心 ==
  has_person      對｜信心 （沒給數字）｜原文「Yes.」
  has_cone        錯｜信心 （沒給數字）｜原文「No.」
  has_ladder      對｜信心 （沒給數字）｜原文「Yes.」
  has_dog         對｜信心 （沒給數字）｜原文「No.」
  has_forklift    對｜信心 （沒給數字）｜原文「No.」
  count_people    對｜信心 （沒給數字）｜原文「1.」
  person_side     對｜信心 （沒給數字）｜原文「Left.」
  cone_vs_ladder  錯｜信心 （沒給數字）｜原文「Right.」
  sign_text       對｜信心 （沒給數字）｜原文「EXIT.」
  有給數字 0/9 題；答錯的題目信心：（答錯的都沒給數字）
  只是在問題後面加一句「給信心分數」，答案就改變的題目：count_people（3→1）、cone_vs_ladder（left→right）
  自述信心是模型生成的文字，不是校準過的機率；要知道準不準，得用有標註的題目量

== 五、資源 ==
  載入 0.7 秒｜23 個問題共 49.2 秒｜峰值記憶體 3,307 MiB
```

## 五、結果怎麼讀

### 1. 自由描述：一半可以核對

五個主張裡三個有證據；「紅橘條紋上衣」圖上沒有（上衣是單一橘色）；「標誌在他們頭上」不精確（標誌在人的右上方）。交通錐與梯子都沒提。這句話讀起來很通順，不核對的話不會發現問題。

「列出物件」也漏了交通錐：同一個模型，換一種問法，遺漏的東西不一樣。

### 2. 結構化問題：錯在哪裡一目了然

9 題錯 2 題：說**沒有交通錐**（但它就在畫面中間），以及說有 **3 個人**（只有 1 個）。結構化問題的好處是程式就能對答案，壞處是只能抓到你有問的錯。

### 3. 區域證據：切小一點比較準

切成三塊後，左塊說 Person、右塊說 Ladder，都對；中間那塊有交通錐和出口標誌，模型只說 Exit sign，又漏了交通錐。三種問法都沒看見交通錐，代表**這個模型在這張圖上對交通錐不可靠**；這是誤判分析要寫的結論，而不是「模型有時會漏」。

### 4. 自述信心

9 題都沒有給數字。更值得注意的是，只是在問題後面加一句「再給信心分數」，**兩題的答案就變了**：人數從 3 變成 1（變對），交通錐與梯子的左右從 left 變成 right（變錯）。小模型對提示的措辭很敏感；同一張圖、同一個問題，問法不同，答案就不同。所以評測要固定提示（第 08 節），而「模型說它很確定」不能當作證據。

## 六、常見誤判

| 誤判 | 實際 |
| --- | --- |
| 「描述很通順，應該是對的」 | 5 個主張 2 個有問題，還漏了 2 個物件 |
| 「問了『有沒有交通錐』就夠了」 | 三種問法都漏掉同一個物件，要寫成模型限制 |
| 「模型說信心 90，就是 90% 會對」 | 自述信心不是校準過的機率；這個模型甚至不給數字 |
| 「同一個問題，答案應該一樣」 | 加一句話，兩題答案就變了 |
| 「沒看到安全帽 = 沒戴」 | 第 10 節：看不到是證據不足 |

## Workshop：圈出證據

### 任務

1. （3 分鐘）`python E01_grounded_caption.py --image scene.png`，用任何看圖軟體打開。
2. （6 分鐘）把 Demo 的自由描述拆成主張，在圖上圈出每個主張的區域（寫座標或截圖圈框）。圈不出來的刪掉，不精確的改寫。
3. （4 分鐘）寫一段新描述：每句後面括號標區域，例如「一個戴黃色安全帽的人（左側，84–156, 92–310）」。
4. （2 分鐘）誤判分析：選一個模型的錯，寫下它錯在哪種類型（不精確、圖上沒有、遺漏、數量）、可能原因、你會怎麼在流程中擋住。

### 參考判讀

- 第 2 題：刪「red and orange striped」；「exit sign above them」改成「畫面中上方有綠色 EXIT 標誌」。
- 第 3 題：要包含交通錐（中間偏下，290–350, 220–310）與梯子（右側，500–580, 120–310）。
- 第 4 題：例如「交通錐被三種問法都漏掉（遺漏）。可能原因：不切塊時圖被縮小，交通錐顏色接近上衣。流程上：巡檢類問題用結構化清單逐項問，答 no 的高風險項目轉人工確認」。

### 驗收

- 第 3 題的描述（每句附區域）。
- 第 4 題的誤判分析。

## 教師紀錄與下一步

- 證據等級：**遠端實測**（示範機 4-learn 以 `--live --save` 錄下的回答；講義輸出是那份紀錄的重播）＋**本機實測**（備課機 cms-server `--live`，23 個回答與示範機逐字相同）＋**教師 trace**（自由描述拆成主張的判讀是教師人工做的，寫在程式的 `CAPTION_CLAIMS`）。
- 自由描述的判讀**只適用於紀錄中的那一句**；`--live` 得到不同句子時，程式會提示「下面的教師判讀不適用」。
- 標註框是畫圖程式的座標，是真值；真實照片的框要人工標，也會有誤差。
- 撰寫時的實際紀錄：第一版把「exit sign above them」判成「錯」；看圖後標誌確實在人的右上方、高於人頭，改判「不精確」。判讀用詞要先定義（第一段的表），否則不同人會給不同標籤。
- 未測：真實照片、雲端 VLM、物件偵測模型（輸出框座標的模型）、切塊模式下的差異。
- 舊教材 [圖像](https://hackmd.io/@yillkid/Q5pqALZsTp-G79ZO_U2Frg) **尚未改寫**。
- 下一步：第 12 節文件摘要、來源與頁碼。
