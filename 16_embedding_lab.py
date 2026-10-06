import argparse
from importlib.metadata import version
from time import perf_counter

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "BAAI/bge-small-zh-v1.5"
REVISION = "7999e1d3359715c523056ef9478215996d62a620"
# Exact retrieval instruction from this model's official model card.
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
DOCUMENTS = [
    ("D1", "工人未配戴安全帽，正在進入施工區。", "missing"),
    ("D2", "工人已配戴安全帽，正在進入施工區。", "present"),
    ("D3", "逃生出口被紙箱阻擋，通道無法通行。", "blocked_exit"),
    ("D4", "機台旁地面積水，尚未清理。", "wet_floor"),
    ("D5", "倉庫室內溫度為攝氏二十五度。", "temperature"),
]
QUERIES = [
    "有人沒有戴頭盔就走進工地。",
    "工人有戴頭盔，沒有缺少安全帽。",
    "請查找未被紙箱堵住的出口。",
    "機器附近地板濕滑。",
]


def load_model():
    torch.set_num_threads(2)
    return SentenceTransformer(
        MODEL_ID, revision=REVISION, device="cpu",
        trust_remote_code=False, model_kwargs={"use_safetensors": True},
    )


def search(model, queries, documents=DOCUMENTS):
    document_vectors = model.encode(
        [row[1] for row in documents], batch_size=8,
        normalize_embeddings=True, show_progress_bar=False,
    )
    query_vectors = model.encode(
        [QUERY_PREFIX + query for query in queries], batch_size=8,
        normalize_embeddings=True, show_progress_bar=False,
    )
    return query_vectors @ document_vectors.T, document_vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="append", help="repeat for more queries")
    queries = parser.parse_args().query or QUERIES
    started = perf_counter()
    model = load_model()
    loaded = perf_counter()
    scores, vectors = search(model, queries)
    elapsed = perf_counter() - loaded
    print("model:", MODEL_ID, "revision:", REVISION)
    print("versions:", {name: version(name) for name in ("torch", "transformers", "sentence-transformers", "numpy")})
    print("shape:", vectors.shape, "norms:", np.round(np.linalg.norm(vectors, axis=1), 3))
    print("keyword 頭盔:", [row[0] for row in DOCUMENTS if "頭盔" in row[1]])
    for query, row in zip(queries, scores):
        order = sorted(range(len(DOCUMENTS)), key=lambda i: (-float(row[i]), DOCUMENTS[i][0]))[:3]
        print("query:", query)
        for i in order:
            print(DOCUMENTS[i][0], f"{row[i]:.4f}", DOCUMENTS[i][1])
        print("nearest-label (not a verdict):", DOCUMENTS[order[0]][2])
    print(f"load_seconds={loaded - started:.3f} encode_search_seconds={elapsed:.3f}")
