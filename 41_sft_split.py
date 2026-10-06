"""第 41 節：SFT 資料與按來源拆分（只用 Python 標準函式庫，不訓練任何模型）。

從 MariaDB 課的 SOP 語料產生合成 SFT 樣本，比較兩種切分：
  - 逐樣本隨機切：同一段落的改寫題同時出現在 train 與 test → 洩漏
  - 按來源群組切：同一份文件的樣本只會在同一個集合
再用近重複檢查找出群組切分漏掉的兩種洩漏——換了編號的複製品、共用模板的拒答——
修正後凍結 test manifest。近重複用回覆的字元 bigram Jaccard，是教學用的粗略指標。
"""

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

BOILERPLATE = "本段為虛構教學資料，不適用真實設備。"
SEED = 20261005
RATIO = {"train": 0.7, "validation": 0.15, "test": 0.15}
NEAR_DUP = 0.6          # 回覆的字元 bigram Jaccard ≥ 此值視為近重複


def load(name):
    here = Path(__file__).resolve().parent
    for base in (here, here.parent / "mariadb", Path.cwd(), Path.cwd().parent / "mariadb"):
        if (base / "data" / name).exists():
            return json.loads((base / "data" / name).read_text(encoding="utf-8"))
    raise FileNotFoundError(f"找不到 data/{name}（在 MariaDB 課的教材目錄）")


TEMPLATES = ("{title}要怎麼做？", "請說明{title}的步驟。", "關於{title}，有什麼要注意的？")


def refusal(style, title, first_rule):
    """三種拒答寫法：同一句模板／只換標題／引用該文件實際寫了什麼。"""
    if style == "template":
        return "文件沒有定義的情況不能推測，請保留紀錄並回報教師。"
    if style == "title":
        return f"《{title}》沒有寫到這個情況，我不能推測；請保留目前的紀錄並回報教師。"
    return f"文件只寫到「{first_rule}」，沒有寫到你問的情況，所以不能推測。"


def build_samples(corpus, refusal_style="template"):
    """每個 active 段落 3 題（同一個目標回覆），每份文件再加 1 題拒答。"""
    titles = {d["document_id"]: d["title"] for d in corpus["documents"] if d["status"] == "active"}
    first_rule = {}
    for chunk in corpus["chunks"]:
        first_rule.setdefault(chunk["document_id"], re.split("[。；]", chunk["text"].replace(BOILERPLATE, ""))[0])
    samples = []
    for chunk in corpus["chunks"]:
        doc = chunk["document_id"]
        if doc not in titles:
            continue
        answer = chunk["text"].replace(BOILERPLATE, "")
        for n, template in enumerate(TEMPLATES, 1):
            samples.append({"id": f"{chunk['chunk_id']}-q{n}", "group": doc, "source": chunk["chunk_id"],
                            "messages": [{"role": "user", "content": template.format(title=titles[doc])},
                                         {"role": "assistant", "content": answer}]})
    for doc, title in sorted(titles.items()):
        samples.append({"id": f"{doc}-refuse", "group": doc, "source": doc,
                        "messages": [{"role": "user", "content": f"{title}的文件沒寫到的情況，可以自己推測嗎？"},
                                     {"role": "assistant", "content": refusal(refusal_style, title, first_rule[doc])}]})
    return samples


# 一份「FAQ 整理」：內容是從 D006 複製改寫來的，卻拿到新的文件編號。
def faq_copy(corpus):
    c012 = next(c for c in corpus["chunks"] if c["chunk_id"] == "C012")
    text = c012["text"].replace(BOILERPLATE, "").replace("資料上傳逾時時", "上傳逾時的時候，")
    return [{"id": "F001-q1", "group": "F001", "source": "F001", "derived_from": "D006",
             "messages": [{"role": "user", "content": "上傳一直沒回應怎麼辦？"},
                          {"role": "assistant", "content": text}]}]


def bigrams(text):
    text = re.sub(r"\s+", "", text)
    return {text[i:i + 2] for i in range(len(text) - 1)}


def jaccard(a, b):
    a, b = bigrams(a), bigrams(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def answer(sample):
    return sample["messages"][-1]["content"]


def split_by_sample(samples):
    order = sorted(samples, key=lambda s: s["id"])
    random.Random(SEED).shuffle(order)
    cut1, cut2 = round(len(order) * RATIO["train"]), round(len(order) * (RATIO["train"] + RATIO["validation"]))
    return {s["id"]: ("train" if i < cut1 else "validation" if i < cut2 else "test") for i, s in enumerate(order)}


def group_of(sample, use_derived=False, template_group=False):
    if template_group and sample["id"].endswith("-refuse"):
        return "TEMPLATE-refuse"           # 同一個模板產生的回覆，視為同一個來源
    if use_derived:
        return sample.get("derived_from") or sample["group"]
    return sample["group"]


def split_by_group(samples, **rule):
    """群組依雜湊排序後輪流分配；同一群組的樣本必在同一集合。"""
    groups = sorted({group_of(s, **rule) for s in samples},
                    key=lambda g: hashlib.sha256(f"{SEED}:{g}".encode()).hexdigest())
    plan = {g: ("test", "validation", "train", "train", "train", "train", "train")[i % 7] for i, g in enumerate(groups)}
    return {s["id"]: plan[group_of(s, **rule)] for s in samples}, plan


def leakage(samples, assign):
    """回傳 (validation／test 中來源段落也出現在別的集合的樣本數, 跨集合的近重複對)。

    validation 也要查：它用來挑 checkpoint，洩漏到 validation 一樣會讓挑選失真。
    """
    sources = defaultdict(set)
    for s in samples:
        sources[s["source"]].add(assign[s["id"]])
    same_source = sum(1 for s in samples if assign[s["id"]] != "train" and len(sources[s["source"]]) > 1)
    pairs = sorted((a["id"], b["id"], round(jaccard(answer(a), answer(b)), 2))
                   for i, a in enumerate(samples) for b in samples[i + 1:]
                   if assign[a["id"]] != assign[b["id"]] and jaccard(answer(a), answer(b)) >= NEAR_DUP)
    return same_source, pairs


def manifest(samples, assign, split="test"):
    rows = sorted((s["id"], hashlib.sha256(json.dumps(s["messages"], ensure_ascii=False, sort_keys=True)
                                           .encode()).hexdigest()[:12]) for s in samples if assign[s["id"]] == split)
    digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]
    return rows, digest


def show_counts(title, samples, assign):
    counts = Counter(assign[s["id"]] for s in samples)
    refuse = Counter(assign[s["id"]] for s in samples if s["id"].endswith("-refuse"))
    groups = defaultdict(set)
    for s in samples:
        groups[assign[s["id"]]].add(s["group"])
    print(f"  {title}")
    for split in ("train", "validation", "test"):
        print(f"    {split:<10} {counts[split]:>2} 筆（拒答 {refuse[split]}）｜文件 {' '.join(sorted(groups[split]))}")


def show_leaks(samples, assign):
    same, pairs = leakage(samples, assign)
    refuse = [p for p in pairs if p[0].endswith("-refuse")]
    other = [p for p in pairs if not p[0].endswith("-refuse")]
    print(f"  validation／test 中，來源段落也在別的集合：{same} 筆")
    print(f"  跨集合近重複對（回覆 Jaccard ≥ {NEAR_DUP}）：{len(pairs)} 對｜其中拒答模板 {len(refuse)} 對")
    for pair in other[:3]:
        print(f"    {pair}")
    return same, pairs


def main():
    corpus = load("sop_corpus.json")
    samples = build_samples(corpus) + faq_copy(corpus)
    print(f"語料 {corpus['dataset_version']}｜合成樣本 {len(samples)} 筆（14 段 × 3 題＋7 題拒答＋1 題 FAQ 複製品）｜seed {SEED}")
    print()

    print("=== 一、逐樣本隨機切 ===")
    naive = split_by_sample(samples)
    show_counts("70／15／15", samples, naive)
    show_leaks(samples, naive)
    print()

    print("=== 二、按文件群組切 ===")
    grouped, _ = split_by_group(samples)
    show_counts("群組依雜湊輪流分配", samples, grouped)
    show_leaks(samples, grouped)
    print("  來源編號檢查是 0，近重複檢查卻不是：F001 換了編號；拒答回覆是同一個模板。")
    print()

    print("=== 三、修正 F001：依 derived_from 歸回 D006 ===")
    derived, _ = split_by_group(samples, use_derived=True)
    show_counts("F001 併入 D006", samples, derived)
    show_leaks(samples, derived)
    print()

    print("=== 四、拒答模板的三種修法 ===")
    lumped, _ = split_by_group(samples, use_derived=True, template_group=True)
    show_counts("(a) 7 題拒答視為同一群組", samples, lumped)
    show_leaks(samples, lumped)
    print("  近重複歸零，但 7 題拒答整組落在同一個集合：train 一題拒答都沒有，模型學不到拒答。")
    titled = build_samples(corpus, "title") + faq_copy(corpus)
    show_counts("(b) 模板裡換上各文件標題", titled, derived)
    show_leaks(titled, derived)
    print("  只換標題，句子其他部分一樣：仍是近重複。")
    samples = build_samples(corpus, "content") + faq_copy(corpus)
    fixed, _ = split_by_group(samples, use_derived=True)
    show_counts("(c) 拒答引用該文件實際寫了什麼", samples, fixed)
    show_leaks(samples, fixed)
    print()

    print("=== 五、凍結 test manifest（採用 (c)）===")
    rows, digest = manifest(samples, fixed)
    print(f"  test {len(rows)} 筆｜manifest 摘要 {digest}")
    print(f"  前 3 列：{rows[:3]}")
    tampered = [dict(s, messages=[s["messages"][0], {"role": "assistant", "content": answer(s) + "。"}])
                if s["id"] == rows[0][0] else s for s in samples]
    _, changed = manifest(tampered, fixed)
    print(f"  {rows[0][0]} 的回覆多加一個句號 → 摘要 {changed}（{'不同' if changed != digest else '相同'}）")
    shots = [s["id"] for s in samples if fixed[s["id"]] == "train"][:2]
    print(f"  few-shot 範例取自 train：{shots}｜與 test 交集 {len(set(shots) & {r[0] for r in rows})} 筆")


if __name__ == "__main__":
    main()
