# 25｜混合檢索與查詢改寫

> **正文改版：2026-10-24｜對應新版 LLM 第 25 節：混合檢索與查詢改寫。**本節用小型合成語料展示中文字元 n-gram 詞項排名與向量排名如何融合，並示範保留原查詢的改寫流程；本節不是 BM25 實作或真實 embedding 評估。

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先備：理解文件、查詢、排名與字典；熟悉向量相似度概念有幫助但不要求套件。50 分鐘：概念 35 分鐘（詞項檢索 10、向量排名 10、融合與改寫 15），Workshop 15 分鐘。

## 一、兩種訊號各有盲點

關鍵字檢索能精確找到料號、專有名詞與錯誤碼，但使用者換了說法就可能漏召回；向量檢索可連結語意相近的表述，卻可能把詞面相似但條件相反的文件排前。混合檢索的目的不是讓其中一方消失，而是保留兩種排名訊號，讓候選更有機會同時涵蓋字面與語意需求。任何排序都只是候選優先序，不等於答案正確。需要判斷效果時，先建立有標註的查詢與相關文件集合，再量 recall@k、MRR 或 nDCG；必須切分調參資料與保留測試資料，不能一邊看測試結果一邊改權重。

## 二、排名融合避免直接相加原始分數

本例不計算 BM25；以下只說明 RRF。倒數排名融合（RRF）只使用名次：每個結果累加 `1 / (k + rank)`，未出現在某路排名的文件不貢獻該路分數。本文以固定合成詞項分數和向量分數產生排名，再融合；這可驗收融合算術，但不是 BM25 公式或真實向量模型的品質證據。

## 三、改寫查詢，原文仍要在場

查詢改寫可將口語描述轉成同義詞或術語，例如「電池撐多久」補充「續航、容量」。但改寫可能刪掉否定、數字、型號或範圍，因此檢索輸入應包含原查詢與改寫候選，並保留來源標記。若改寫結果为空或把「不可以」反成肯定，原查詢仍需參與檢索，並把改寫失敗記錄供評估；不可只搜尋模型改寫後的版本。

## 四、可執行 Demo

下面用標準庫實作簡化詞項分數（中文字元二元 n-gram，非 BM25）與可替換的向量分數列表，並透過 RRF 融合排名。查詢包裝函式會明確保留原查詢。執行：`python3 25_hybrid_retrieval.py`。

<!-- demo: 25_hybrid_retrieval.py -->
```python
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
```

關鍵輸出（本機實測；合成排名 fixtures，向量分數非模型輸出）：

```text
keyword: ['A', 'B']
vector: ['B', 'A', 'C']
RRF: ['A', 'B', 'C']
queries: ['電池可以撐多久', '電池續航']
```

## Workshop

15 分鐘：前 5 分鐘在文件 C 加入「電池續航」並確認 C 取得非零詞項分數；再 5 分鐘將向量分數改成 A=0.95、B=0.91、C=0.10；最後 5 分鐘觀察單路與融合結果。參考判讀：排名可能變動，融合仍依兩路順位累計，並非固定偏袒某檢索器。可執行驗收：執行原程式確認原查詢在 queries 第一位；傳入空改寫後原查詢仍存在；用 `assert fuse([["A", "B"], ["B", "A"]])[0] == "A"` 驗證同分依 ID 規則；確認文件新增關鍵詞後關鍵字排名可見其排序變化。記錄改動前後排名，不以「看起來更好」作為唯一通過條件。

## 教師紀錄與下一步

證據級別：本機實測為教材程式於 2026-12-06 直接執行，環境 Linux、Python 3（標準函式庫，微版本未記錄），無模型 revision，量測方式為 stdout；向量值是合成 fixtures，不是遠端或本機模型推論。詞項計分刻意簡化，沒有 BM25 長度正規化；詞切分對中文並非通用斷詞，RRF 不保證改善召回或答案品質。尚未實班試教，不宣稱學生實測。下一節可連接檢索評估與重排；舊教材連結僅為歷史參照，未聲稱同步。
