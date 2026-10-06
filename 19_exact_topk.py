"""第 19 節：記憶體精確檢索 baseline。

把 16 個 chunk 的向量放進一個矩陣，對每個查詢算出**全部**分數再排序。沒有索引、沒有近似。
需要第 16 節的套件與模型；語料與標註是 MariaDB 課的 sop-v1／qrels-v1（llm-demo 的 data/）。
numpy 在用到時才匯入，讓過濾與指標函式不裝套件也能測試。
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

MODEL_ID = "BAAI/bge-small-zh-v1.5"
REVISION = "7999e1d3359715c523056ef9478215996d62a620"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："  # 模型卡原字串，見第 16 節

# products-v1 的類別與狀態（MariaDB 課 data/seed.sql 的兩欄；與第 22 節相同，測試會核對）
PRODUCTS = {
    "P001": (1, "active"), "P002": (1, "active"), "P003": (1, "inactive"), "P004": (2, "active"),
    "P005": (2, "active"), "P006": (1, "inactive"), "P007": (1, "active"), "P008": (1, "active"),
    "P009": (1, "active"), "P010": (1, "active"), "P011": (3, "active"), "P012": (1, "active"),
}


def load(name):
    here = Path(__file__).resolve().parent
    for base in (here, here.parent / "mariadb", Path.cwd(), Path.cwd().parent / "mariadb"):
        path = base / "data" / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"找不到 data/{name}（llm-demo 的 data/，或 MariaDB 課的教材目錄）")


def load_model():
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    return SentenceTransformer(MODEL_ID, revision=REVISION, device="cpu",
                               trust_remote_code=False, model_kwargs={"use_safetensors": True})


class ExactIndex:
    """全部向量放在一個 (N, d) 矩陣；查詢時一次乘完，再排序。"""

    def __init__(self, ids, vectors):
        import numpy as np

        vectors = np.asarray(vectors, dtype=np.float32)
        if len(ids) != len(vectors) or len(set(ids)) != len(ids):
            raise ValueError("ids 必須與向量一一對應且不重複")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("向量必須先正規化；否則內積不等於 cosine（第 17 節）")
        self.ids, self.vectors = list(ids), vectors

    def search(self, query_vector, k, allowed=None):
        """回傳 [(chunk_id, score)]；分數大到小，同分時 chunk_id 小的在前。"""
        if k < 1:
            raise ValueError("k 至少為 1")
        import numpy as np

        scores = self.vectors @ np.asarray(query_vector, dtype=np.float32)
        rows = [(cid, float(s)) for cid, s in zip(self.ids, scores) if allowed is None or cid in allowed]
        rows.sort(key=lambda row: (-row[1], row[0]))
        return rows[:k]

    def nbytes(self):
        return self.vectors.nbytes


def eligible_chunks(corpus, category_id, products):
    """MariaDB 課的合格規則：文件 active，且連到至少一個該類別的 active 產品。"""
    active_docs = {d["document_id"] for d in corpus["documents"] if d["status"] == "active"}
    linked = {doc for pid, doc in corpus["links"]
              if products.get(pid, (None, None))[0] == category_id and products[pid][1] == "active"}
    return {c["chunk_id"] for c in corpus["chunks"] if c["document_id"] in active_docs & linked}


def brute_force(ids, vectors, query_vector, k):
    """不用矩陣、一筆一筆算；拿來核對 ExactIndex 沒有寫錯。"""
    rows = []
    for cid, vector in zip(ids, vectors):
        rows.append((cid, sum(float(a) * float(b) for a, b in zip(vector, query_vector))))
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows[:k]


def recall_at(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def fingerprint(baseline):
    blob = json.dumps(baseline, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def bench(sizes, dim=512, queries=20, seed=20261006):
    """隨機單位向量量測全量計算的時間與記憶體；時間依機器而定，不寫進講義凍結輸出。"""
    import numpy as np

    rng = np.random.default_rng(seed)
    for n in sizes:
        matrix = rng.standard_normal((n, dim), dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        q = rng.standard_normal((queries, dim), dtype=np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        started = perf_counter()
        for row in q:
            scores = matrix @ row
            np.argpartition(-scores, 10)[:10]
        per_query = (perf_counter() - started) / queries
        print(f"N={n:>9,} 矩陣 {matrix.nbytes / 2**20:8.1f} MiB  每次查詢 {per_query * 1000:8.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--query", action="append", help="另外查自己的問題（不套分類過濾）；可重複")
    parser.add_argument("--save", help="把 baseline 排名寫成 JSON 到這個路徑")
    parser.add_argument("--bench", action="store_true", help="另外量測 1 千到 20 萬筆的時間（依機器而定）")
    args = parser.parse_args()

    corpus, cases = load("sop_corpus.json"), load("search_cases.json")
    model = load_model()
    ids = [c["chunk_id"] for c in corpus["chunks"]]
    vectors = model.encode([c["text"] for c in corpus["chunks"]], normalize_embeddings=True, show_progress_bar=False)
    index = ExactIndex(ids, vectors)
    print("model:", MODEL_ID, "revision:", REVISION)
    print("corpus:", corpus["dataset_version"], "qrels:", cases["annotation_version"],
          "chunks:", len(ids), "矩陣:", index.vectors.shape, index.vectors.dtype, index.nbytes(), "bytes")

    print("\n== 一、精確檢索要先自我核對 ==")
    self_hits = [index.search(v, 1)[0][0] for v in index.vectors]
    print("每段查自己排第一:", self_hits == ids)
    q0 = model.encode([QUERY_PREFIX + cases["cases"][0]["query"]], normalize_embeddings=True, show_progress_bar=False)[0]
    fast, slow = index.search(q0, 16), brute_force(ids, index.vectors, q0, 16)
    print("矩陣與逐筆計算排名相同:", [r[0] for r in fast] == [r[0] for r in slow],
          "最大分數差 < 1e-6:", max(abs(a[1] - b[1]) for a, b in zip(fast, slow)) < 1e-6)

    print("\n== 二、同分時的順序 ==")
    tie = ExactIndex(["C002", "C001", "C003"], [index.vectors[0], index.vectors[0], index.vectors[2]])
    print("兩筆向量完全相同，插入順序 C002, C001 →", [r[0] for r in tie.search(q0, 3)])
    print("沒有第二排序鍵時，同分的順序取決於插入順序；本頁固定以 chunk_id 為第二鍵")

    print(f"\n== 三、qrels-v1 八題（k={args.k}，先過濾合格段落，再排序）==")
    query_vectors = model.encode([QUERY_PREFIX + c["query"] for c in cases["cases"]],
                                 normalize_embeddings=True, show_progress_bar=False)
    baseline = {}
    for case, qv in zip(cases["cases"], query_vectors):
        allowed = eligible_chunks(corpus, case["category_id"], PRODUCTS)
        hits = index.search(qv, args.k, allowed)
        unfiltered = index.search(qv, 1)[0]
        ranked = [cid for cid, _ in hits]
        relevant = case["relevant_chunk_ids"]
        baseline[case["case_id"]] = [[cid, round(s, 4)] for cid, s in hits]
        note = f"recall@{args.k}={recall_at(ranked, relevant, args.k):.2f}" if relevant else "標註無答案"
        print(f"{case['case_id']} 合格 {len(allowed):2} 段 標註 {relevant or '[]'} → "
              f"{[f'{cid} {s:.4f}' for cid, s in hits]}  {note}"
              + ("" if hits else f"（不過濾時第一名是 {unfiltered[0]} {unfiltered[1]:.4f}）"))

    print("\n== 四、無答案題也會回 k 筆 ==")
    answered = [baseline[c["case_id"]][0][1] for c in cases["cases"] if c["relevant_chunk_ids"]]
    unanswered = [(c["case_id"], baseline[c["case_id"]][0][1]) for c in cases["cases"]
                  if not c["relevant_chunk_ids"] and baseline[c["case_id"]]]
    print("有答案題的第一名分數:", sorted(answered))
    print("無答案題的第一名分數:", unanswered)
    print("只有 8 題，不能據此訂門檻（第 17 節：dev 挑的門檻在保留集比全猜還差）")

    print("\n== 五、baseline 紀錄 ==")
    record = {"model": f"{MODEL_ID}@{REVISION}", "dataset_version": corpus["dataset_version"],
              "annotation_version": cases["annotation_version"], "k": args.k, "rankings": baseline}
    print("fingerprint:", fingerprint(record))
    for n in (16, 10_000, 1_000_000):
        print(f"記憶體估算 N={n:>9,}（float32、512 維）: {n * 512 * 4:>13,} bytes ≈ {n * 512 * 4 / 2**20:8.2f} MiB")
    if args.save:
        Path(args.save).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("已寫入:", args.save)
    if args.query:
        print("\n== 自訂查詢（全部 16 段，不套分類過濾）==")
        extra = model.encode([QUERY_PREFIX + q for q in args.query], normalize_embeddings=True, show_progress_bar=False)
        for q, qv in zip(args.query, extra):
            print(q, "→", [f"{cid} {s:.4f}" for cid, s in index.search(qv, args.k)])
    if args.bench:
        print("\n== 量測（依機器而定）==")
        bench([1_000, 10_000, 100_000, 200_000])
