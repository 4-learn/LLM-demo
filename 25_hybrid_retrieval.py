import math
import re

DOCS = {
    "A": "電池容量 5000 mAh 續航測試",
    "B": "電池續航測試結果",
    "C": "充電器輸出功率 20 W",
}
QUERY = "電池可以撐多久"
# 固定合成向量相似度；只作排名訊號，不代表模型推論
VECTOR = {"A": 0.72, "B": 0.91, "C": 0.10}
# 「電池」的二元字元訊號確實出現在 A、B，C 不含該 n-gram。


def terms(text):
    """中文字元二元 n-gram；ASCII 與數字按連續詞切分。"""
    pieces = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", text.lower())
    output = []
    for piece in pieces:
        if re.fullmatch(r"[\u4e00-\u9fff]+", piece):
            output.extend(piece[i:i + 2] for i in range(len(piece) - 1))
            if len(piece) == 1:
                output.append(piece)
        else:
            output.append(piece)
    return output


def keyword_rank(query, documents):
    q = set(terms(query))
    doc_terms = {doc_id: set(terms(text)) for doc_id, text in documents.items()}
    df = {term: sum(term in found for found in doc_terms.values()) for term in q}
    scores = {doc_id: sum(math.log(1 + len(documents) / (1 + df[t]))
                          for t in q if t in found)
              for doc_id, found in doc_terms.items()}
    return sorted((key for key in scores if scores[key] > 0), key=lambda key: (-scores[key], key))


def fuse(rankings, k=10):
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)
    return sorted(scores, key=lambda key: (-scores[key], key))


def retrieval_queries(original, rewrites):
    if not isinstance(original, str) or not original.strip():
        raise ValueError("original query must be nonempty text")
    if not isinstance(rewrites, (list, tuple)) or any(not isinstance(item, str) for item in rewrites):
        raise ValueError("rewrites must be a list or tuple of strings")
    # 穩定去重，避免同一個改寫被重複送檢索而加重其融合權重。
    return list(dict.fromkeys([original] + [item for item in rewrites if item.strip()]))


if __name__ == "__main__":
    kw = keyword_rank(QUERY, DOCS)
    vec = sorted(VECTOR, key=lambda key: (-VECTOR[key], key))
    print("keyword:", kw)
    print("vector:", vec)
    print("RRF:", fuse([kw, vec]))
    print("queries:", retrieval_queries(QUERY, ["電池續航", ""]))