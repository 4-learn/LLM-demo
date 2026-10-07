"""第 11 節：文件摘要、來源與頁碼。

一份合成的「教學閘道器設定手冊 v2」掃描檔，3 張圖：PDF 第 1–3 頁，**印刷頁碼是 6、7、9**（第 8 頁缺頁），
第 2 頁掃描模糊。兩條讀取路徑：
  文字層：手冊原檔的文字（born-digital PDF 抽得到的那種），本節當作真值
  VLM 閱讀：SmolVLM-500M 看圖轉寫，分「不切塊」與「切塊」兩種設定（示範機錄下，預設重播）
接著核對一份三點摘要草稿（教師準備的劇本，模擬模型交出的草稿）：每點要附 文件 ID、版本雜湊、PDF 頁、
印刷頁碼、原文片段；程式逐點檢查片段是否真的在那一頁、頁碼是否存在。

    python 11_doc_citations.py
    python 11_doc_citations.py --live [--save]       # 實跑 VLM 轉寫（需 torch、transformers、Pillow、模型快取）
    python 11_doc_citations.py --pages pages/         # 把 3 張頁面圖存下來（需要 Pillow）
"""
import argparse
import difflib
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE / "data" / "11-doc-record.json"
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
REVISION = "a7da5b986cb59b408707209984f360a5f4ad7e47"
THREADS = 4
TRANSCRIBE_Q = "Transcribe all the text on this page."

DOC_ID = "GW-MANUAL"
VERSION = "v2"
# PDF 頁序 → (印刷頁碼, 掃描模糊程度, 文字)
PAGES = {
    1: (6, 0, ["Gateway Setup Guide (v2)", "", "Step 1. Connect to 2.4 GHz Wi-Fi only.",
               "Step 2. 5 GHz is not supported.", "Step 3. Enter the classroom password."]),
    2: (7, 2.5, ["Troubleshooting", "", "If pairing fails, restart the gateway.",
                 "Wait 30 seconds before retrying.", "Do not reset more than 3 times."]),
    3: (9, 0, ["Error codes", "", "E01 means a missing value.", "Other codes: report to the teacher.",
               "Do not guess the meaning of unlisted codes."]),
}


def page_text(pdf_page):
    return "\n".join(line for line in PAGES[pdf_page][2] if line)


def doc_hash():
    joined = "\n\f".join(page_text(n) for n in sorted(PAGES))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def printed_to_pdf():
    return {printed: n for n, (printed, _, _) in PAGES.items()}


# 三點摘要草稿（劇本，模擬模型交出的草稿；故意放進三種常見錯誤）
DRAFT = [
    {"point": "閘道器只能用 2.4 GHz，不支援 5 GHz。", "printed_page": 6,
     "quote": "Step 2. 5 GHz is not supported."},
    {"point": "配對失敗時重開閘道器，等 30 秒再試。", "printed_page": 7,
     "quote": "Wait 30 seconds before retrying."},
    {"point": "重設最多 5 次。", "printed_page": 7,
     "quote": "Do not reset more than 5 times."},                                  # 片段被改過
    {"point": "E02 代表連線逾時。", "printed_page": 8,
     "quote": "E02 means connection timeout."},                                     # 引用缺頁
    {"point": "未列出的錯誤代碼要回報教師。", "printed_page": 3,
     "quote": "Other codes: report to the teacher."},                               # 把 PDF 頁序當印刷頁碼
]


# ---------------------------------------------------------------- 畫頁面

def draw(pdf_page):
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    printed, blur, lines = PAGES[pdf_page]
    image = Image.new("RGB", (595, 842), "white")
    pen = ImageDraw.Draw(image)
    body = ImageFont.load_default(size=20)
    for i, line in enumerate(lines):
        pen.text((50, 60 + i * 34), line, fill="black", font=body)
    pen.text((280, 800), f"- {printed} -", fill="black", font=ImageFont.load_default(size=16))
    return image.filter(ImageFilter.GaussianBlur(blur)) if blur else image


def image_hash(image):
    return hashlib.sha256(image.tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------- 核對

def similarity(a, b):
    return difflib.SequenceMatcher(None, " ".join(a.split()), " ".join(b.split())).ratio()


def missing_lines(truth, got):
    flat = " ".join(got.split()).lower()
    return [line for line in truth.split("\n") if " ".join(line.split()).lower() not in flat]


def check_point(item, pages_text, basis="原文"):
    """回傳 (判定, 理由)。pages_text：PDF 頁 → 用來核對的文字。"""
    to_pdf = printed_to_pdf()
    printed = item["printed_page"]
    if printed not in to_pdf:
        if printed in PAGES:
            return "退回", (f"印刷頁碼 {printed} 不存在；PDF 第 {printed} 頁的印刷頁碼是 {PAGES[printed][0]}"
                          f"（把 PDF 頁序當成頁碼）")
        return "退回", f"印刷頁碼 {printed} 不在這份文件裡（缺頁或不存在），不能引用"
    text = pages_text[to_pdf[printed]]
    if item["quote"] in text:
        return "通過", f"片段在 PDF 第 {to_pdf[printed]} 頁（印刷 {printed}）"
    best = max(text.split("\n"), key=lambda line: similarity(line, item["quote"]))
    return "退回", f"片段不在該頁；最接近的{basis}是「{best}」（相似度 {similarity(best, item['quote']):.2f}）"


# ---------------------------------------------------------------- 模型

def run_model():
    import PIL
    import torch
    import transformers
    from transformers import AutoModelForVision2Seq, AutoProcessor

    torch.set_num_threads(THREADS)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.float32).eval()
    out = {"model": f"{MODEL_ID}@{REVISION}",
           "settings": {"threads": THREADS, "do_sample": False, "max_new_tokens": 90, "dtype": "float32"},
           "environment": {"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
                           "transformers": transformers.__version__, "pillow": PIL.__version__},
           "image_sha256_16": {}, "runs": {}}
    for split in (False, True):
        processor.image_processor.do_image_splitting = split
        name = "切塊" if split else "不切塊"
        out["runs"][name] = {}
        for n in sorted(PAGES):
            image = draw(n)
            out["image_sha256_16"][str(n)] = image_hash(image)
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": TRANSCRIBE_Q}]}]
            inputs = processor(text=processor.apply_chat_template(messages, add_generation_prompt=True),
                               images=[image], return_tensors="pt")
            started = time.perf_counter()
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=90, do_sample=False)
            out["runs"][name][str(n)] = {
                "text": processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip(),
                "input_tokens": int(inputs["input_ids"].shape[1]), "seconds": round(time.perf_counter() - started, 1)}
    out["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    return out


def check_images(record):
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        return "沒有可用的 Pillow，未重畫核對（重播仍可進行）"
    differ = [n for n in PAGES if image_hash(draw(n)) != record["image_sha256_16"][str(n)]]
    return "3 頁重畫後雜湊與錄製時相同" if not differ else f"重畫後雜湊不同：第 {differ} 頁"


# ---------------------------------------------------------------- 主程式

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--pages", help="把 3 張頁面圖存到這個資料夾（需要 Pillow）")
    args = parser.parse_args(argv)
    if args.pages:
        Path(args.pages).mkdir(parents=True, exist_ok=True)
        for n in PAGES:
            draw(n).save(Path(args.pages) / f"pdf-page-{n}.png")
        print(f"已存 3 頁到 {args.pages}")

    if args.live:
        record = run_model()
        source = f"即時執行（{record['environment']['host']}）"
        if args.save:
            RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            source += "，已寫入 data/11-doc-record.json"
    else:
        record = json.loads(RECORD.read_text(encoding="utf-8"))
        source = f"重播 data/11-doc-record.json（錄製於 {record['environment']['host']}，非即時推論）"
    env = record["environment"]
    print(f"來源：{source}")
    print(f"模型：{MODEL_ID}｜revision {REVISION[:8]}｜greedy｜Python {env['python']}｜Pillow {env['pillow']}")
    print(f"圖片核對：{check_images(record)}")
    print(f"文件：{DOC_ID} {VERSION}｜文字層 sha256 {doc_hash()}｜PDF 3 頁｜印刷頁碼 "
          + "、".join(f"PDF {n}→{p[0]}" for n, p in PAGES.items()) + "｜缺第 8 頁")

    print("\n== 一、文字層 vs VLM 轉寫（與文字層的相似度；漏掉的行）==")
    for name, run in record["runs"].items():
        for n in sorted(PAGES):
            r = run[str(n)]
            lost = missing_lines(page_text(n), r["text"])
            blur = "（模糊）" if PAGES[n][1] else ""
            print(f"  {name:<3} PDF {n}{blur:<4}｜{r['input_tokens']:>3} token {r['seconds']:>5.1f} 秒｜"
                  f"相似度 {similarity(page_text(n), r['text']):.2f}｜漏 {len(lost)} 行")
    blurred = record["runs"]["不切塊"]["2"]["text"]
    print(f"  不切塊讀模糊頁的前 80 字：「{' '.join(blurred.split())[:80]}」")
    print(f"  原文：「{' '.join(page_text(2).split())}」")

    print("\n== 二、核對摘要草稿（以文字層為準）==")
    text_layer = {n: page_text(n) for n in PAGES}
    verdicts = []
    for i, item in enumerate(DRAFT, 1):
        verdict, why = check_point(item, text_layer)
        verdicts.append(verdict)
        print(f"  {i}. {verdict}｜印刷第 {item['printed_page']} 頁｜{item['point']}\n     {why}")
    print(f"  通過 {verdicts.count('通過')}／{len(DRAFT)}；能交出的三點摘要："
          f"{'夠' if verdicts.count('通過') >= 3 else '不夠，要補點或寫明證據不足'}")

    print("\n== 三、如果只有 VLM 轉寫可以核對 ==")
    for name, run in record["runs"].items():
        vlm = {n: run[str(n)]["text"] for n in PAGES}
        results = [check_point(item, vlm, "轉寫") for item in DRAFT]
        print(f"  {name}：通過 {sum(v == '通過' for v, _ in results)}／{len(DRAFT)}")
        for i, ((verdict, why), item) in enumerate(zip(results, DRAFT), 1):
            if verdict != verdicts[i - 1]:
                print(f"    {i}. 文字層{verdicts[i - 1]}，轉寫{verdict}｜{why}")
    print("  核對的依據換成轉寫結果，正確的引用也會被退回；轉寫錯一個標點，片段就對不上")
    print("  轉寫要先標品質（與原檔比對、或人工抽查），模糊頁要重掃或人工轉寫；不能讓模型補寫看不清的字")

    print("\n== 四、資源 ==")
    for name, run in record["runs"].items():
        print(f"  {name}：3 頁共 {sum(r['seconds'] for r in run.values()):.1f} 秒，"
              f"每頁約 {run['1']['input_tokens']} 個輸入 token")
    print(f"  峰值記憶體 {record['peak_rss_mib']:,} MiB")


if __name__ == "__main__":
    main()
