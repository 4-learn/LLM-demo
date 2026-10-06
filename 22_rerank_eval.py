"""第 22 節：Reranking 與 Retrieval Evaluation（只用 Python 標準函式庫）。

語料與標註都沿用 MariaDB 課：`sop_corpus.json`（16 段）與 `search_cases.json`（8 題 qrels）。

兩個階段，**都不是神經網路**：
  - 候選：字元 bigram 的 TF-IDF cosine（詞彙檢索；站在 bi-encoder 的位置）
  - 重排：同義詞規則，只在候選內加分調整順序（站在 cross-encoder 的位置；不會找新段落）
本節要教的是**評測方法**：候選召回與重排效益分開量、dev 與 holdout 分開看，
不是這兩個規則有多好。
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

CANDIDATES = 4            # 第一階段交給重排的段落數
BOILERPLATE = "本段為虛構教學資料，不適用真實設備。"   # 每段都有，不該影響排序


def load(name):
    here = Path(__file__).resolve().parent
    for base in (here, here.parent / "mariadb", Path.cwd(), Path.cwd().parent / "mariadb"):
        path = base / "data" / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"找不到 data/{name}（在 MariaDB 課的教材目錄）")


def bigrams(text):
    text = re.sub(r"\s+", "", text.replace(BOILERPLATE, ""))
    return Counter(text[i:i + 2] for i in range(len(text) - 1))


class LexicalIndex:
    def __init__(self, chunks):
        self.ids = [c["chunk_id"] for c in chunks]
        self.texts = {c["chunk_id"]: c["text"] for c in chunks}
        grams = {cid: bigrams(self.texts[cid]) for cid in self.ids}
        df = Counter(g for counts in grams.values() for g in counts)
        n = len(self.ids)
        self.idf = {g: math.log((n + 1) / (d + 1)) + 1 for g, d in df.items()}
        self.vectors = {cid: self.weigh(counts) for cid, counts in grams.items()}

    def weigh(self, counts):
        vec = {g: c * self.idf.get(g, 0.0) for g, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {g: v / norm for g, v in vec.items()}

    def search(self, query, allowed=None):
        q = self.weigh(bigrams(query))
        scored = [(sum(w * self.vectors[cid].get(g, 0.0) for g, w in q.items()), cid)
                  for cid in self.ids if allowed is None or cid in allowed]
        # 分數 0 = 與查詢沒有任何共同 bigram：那不是「排在後面」，是「沒被檢出」。
        # 不濾掉的話，0 分段落會依 ID 排出假名次（本節第一版就犯了這個錯）。
        return sorted([x for x in scored if x[0] > 0], key=lambda item: (-item[0], item[1]))


# 本節自編的改寫題：刻意不用原文的字眼。作者為 AI agent，需教師覆核相關性。
# dev 用來調重排規則；holdout 調完才看，只看一次。
PARAPHRASES = (
    ("H01", "dev",     "感測器 F 要怎麼跟面板綁在一起？",             {"C009"}),
    ("H02", "dev",     "上傳卡住了，再按一次會不會算成兩筆？",         {"C012"}),
    ("H03", "dev",     "量出來跟基準對不上，可以直接改掉舊的數字嗎？", {"C008"}),
    ("H04", "dev",     "按了送出按鈕，就算傳到了嗎？",                 {"C011"}),
    ("H05", "holdout", "燈亮了是不是就代表連上了？",                   {"C002"}),
    ("H06", "holdout", "畫面跳出 E07 是什麼意思？",                    {"C016"}),
    ("H07", "holdout", "換成比較快的那個頻率能不能解決連不上的問題？", {"C003", "C004"}),
    ("H08", "holdout", "偏差太大的話，原本的紀錄要保留嗎？",           {"C008"}),
)

# 看 dev 四題失敗原因後寫的同義詞表：查詢用語 → 原文用語。只看過 dev。
SYNONYMS = {"綁": "配對", "基準": "參考值", "改掉": "覆寫", "卡住": "逾時", "兩筆": "第二筆", "傳到": "送達"}


def rerank(query, candidates, texts, weight=0.5):
    """只在候選內調整順序：查詢用語對上原文用語，每對加 weight 分。不會加入新段落。"""
    def hits(cid):
        return sum(1 for said, written in SYNONYMS.items() if said in query and written in texts[cid])
    return sorted(candidates, key=lambda item: (-(item[0] + weight * hits(item[1])), item[1]))


def first_hit(ranked, relevant):
    return next((i for i, cid in enumerate(ranked, 1) if cid in relevant), None)


def eligible_chunks(corpus, category_id, products):
    """MariaDB 課的合格規則：文件 active，且連到至少一個該類別的 active 產品。"""
    active_docs = {d["document_id"] for d in corpus["documents"] if d["status"] == "active"}
    linked = {doc for pid, doc in corpus["links"]
              if products.get(pid, (None, None))[0] == category_id and products[pid][1] == "active"}
    return {c["chunk_id"] for c in corpus["chunks"] if c["document_id"] in active_docs & linked}


# products-v1 的類別與狀態（MariaDB 課 data/seed.sql；此處只取兩欄，測試會核對）
PRODUCTS = {
    "P001": (1, "active"), "P002": (1, "active"), "P003": (1, "inactive"), "P004": (2, "active"),
    "P005": (2, "active"), "P006": (1, "inactive"), "P007": (1, "active"), "P008": (1, "active"),
    "P009": (1, "active"), "P010": (1, "active"), "P011": (3, "active"), "P012": (1, "active"),
}


def recall_at(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked, relevant):
    for position, cid in enumerate(ranked, 1):
        if cid in relevant:
            return 1 / position
    return 0.0


def ndcg_at(ranked, relevant, k):
    dcg = sum(1 / math.log2(i + 2) for i, cid in enumerate(ranked[:k]) if cid in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal


def expand(query):
    """查詢擴充：把同義詞表用在**候選階段**——查詢裡出現的說法，補上原文的寫法。"""
    return query + "".join(f" {written}" for said, written in SYNONYMS.items() if said in query)


def evaluate(index, queries, allowed, k, expanded=False):
    rows = []
    for qid, split, query, relevant in queries:
        full = index.search(expand(query) if expanded else query, allowed)
        before = [cid for _, cid in full[:k]]
        after = [cid for _, cid in rerank(query, full[:k], index.texts)]
        rows.append((qid, split, relevant, [cid for _, cid in full], before, after))
    return rows


def summary(rows, split, k):
    picked = [r for r in rows if r[1] == split]
    cand = sum(recall_at(b, rel, k) for _, _, rel, _, b, _ in picked) / len(picked)
    mrr_b = sum(reciprocal_rank(b, rel) for _, _, rel, _, b, _ in picked) / len(picked)
    mrr_a = sum(reciprocal_rank(a, rel) for _, _, rel, _, _, a in picked) / len(picked)
    return cand, mrr_b, mrr_a


def main():
    corpus, cases = load("sop_corpus.json"), load("search_cases.json")
    index = LexicalIndex(corpus["chunks"])
    print(f"語料 {corpus['dataset_version']}（{len(index.ids)} 段）｜檢索：字元 bigram TF-IDF｜重排：同義詞規則（皆非神經網路）")
    print()

    print(f"=== 一、MariaDB 課的 {cases['annotation_version']}（{cases['provenance']['review_status']}）===")
    scored = []
    for case in cases["cases"]:
        allowed = eligible_chunks(corpus, case["category_id"], PRODUCTS)
        relevant = set(case["relevant_chunk_ids"])
        ranked = [cid for _, cid in index.search(case["query"], allowed)]
        if not relevant:
            print(f"  {case['case_id']} 合格 {len(allowed):>2} 段｜相關＝空集合（{case['kind']}）→ 不計 Recall／MRR")
            continue
        scored.append((recall_at(ranked, relevant, CANDIDATES), reciprocal_rank(ranked, relevant), ndcg_at(ranked, relevant, 3)))
        print(f"  {case['case_id']} 合格 {len(allowed):>2} 段｜相關 {' '.join(sorted(relevant)):<9}｜前 4 {' '.join(ranked[:4])}")
    r, m, n = (sum(col) / len(scored) for col in zip(*scored))
    print(f"  可計分 {len(scored)} 題：Recall@{CANDIDATES} {r:.3f}｜MRR {m:.3f}｜nDCG@3 {n:.3f}")
    print("  → 全部滿分：這組題目對詞彙檢索太容易，**量不出重排有沒有用**。")
    print()

    active = {c["chunk_id"] for c in corpus["chunks"]
              if c["document_id"] in {d["document_id"] for d in corpus["documents"] if d["status"] == "active"}}
    print(f"=== 二、改寫題（{len(PARAPHRASES)} 題；只在 {len(active)} 段 active 內檢索）===")
    for k, expanded, title in ((CANDIDATES, False, "k=4，同義詞只用於重排"),
                               (len(active), False, f"k={len(active)}（全部），同義詞只用於重排"),
                               (CANDIDATES, True, "k=4，同義詞也用於查詢擴充（候選階段）")):
        rows = evaluate(index, PARAPHRASES, active, k, expanded)
        print(f"  {title}")
        for qid, split, relevant, full, before, after in rows:
            where = "、".join(str(full.index(c) + 1) if c in full else "未檢出" for c in sorted(relevant))
            b, a = first_hit(before, relevant), first_hit(after, relevant)
            moved = "未進候選，重排無能為力" if b is None else f"候選第 {b} → 重排第 {a}"
            print(f"    {qid} {split:<7} 相關 {' '.join(sorted(relevant)):<9} 完整排序 {where:<7}｜{moved}")
        for split in ("dev", "holdout"):
            cand, mrr_b, mrr_a = summary(rows, split, k)
            print(f"    {split:<7} Recall@{k} {cand:.3f}｜MRR {mrr_b:.3f} → {mrr_a:.3f}")
        print()

    print("=== 三、不加合格過濾會怎樣（Q05：問的是沒有 SOP 的 P010）===")
    q05 = next(c for c in cases["cases"] if c["case_id"] == "Q05")
    print(f"  不過濾：前 3 名 {[cid for _, cid in index.search(q05['query'])[:3]]}")
    print("  檢索永遠會回傳東西；qrels 是空集合，所以這 3 段都不相關。是否拒答要另外判斷（第 23 節）。")


if __name__ == "__main__":
    main()
