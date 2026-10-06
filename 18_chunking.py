"""第 18 節：Chunk 與來源資料模型（只用 Python 標準函式庫）。

語料沿用 MariaDB 課的 `sop-v1`（8 份虛構教學 SOP，每份 2 段），學生在那門課已經入庫過。
這裡把每份文件的兩段接回全文，再用三種方法重切，比較：
  1. 切出來的 chunk 能不能**自己成立**（否定詞、條件有沒有被切走）；
  2. 每個 chunk 帶的來源欄位夠不夠**引用與更新**。

本節不計算任何向量；`embedding_revision` 只是欄位，示範「不同向量空間不能混用」的檢查。
"""

import hashlib
import json
from pathlib import Path

PARAGRAPH_BREAK = "\n\n"
EMBEDDING = "BAAI/bge-small-zh-v1.5@7999e1d3359715c523056ef9478215996d62a620"   # 與第 16 節相同


def load_corpus():
    here = Path(__file__).resolve().parent
    for base in (here, here.parent / "mariadb", Path.cwd(), Path.cwd().parent / "mariadb"):
        path = base / "data" / "sop_corpus.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("找不到 data/sop_corpus.json（在 MariaDB 課的教材目錄）")


def documents(corpus):
    """把 MariaDB 的兩段接回全文，段落之間用空行。"""
    texts = {}
    for chunk in corpus["chunks"]:
        texts.setdefault(chunk["document_id"], []).append(chunk["text"])
    versions = {d["document_id"]: d["source_version"] for d in corpus["documents"]}
    return {doc_id: (versions[doc_id], PARAGRAPH_BREAK.join(parts)) for doc_id, parts in texts.items()}


# ---- 三種切法：都回傳 (start, end) 區間，文字一律由原文切出，保證 offset 可回查 ----
def split_fixed(text, size=40, overlap=0):
    spans, start = [], 0
    while start < len(text):
        spans.append((start, min(start + size, len(text))))
        if start + size >= len(text):
            break
        start += size - overlap
    return spans


def split_on(text, separator):
    spans, start = [], 0
    while start < len(text):
        found = text.find(separator, start)
        end = len(text) if found == -1 else found + len(separator)
        piece = text[start:end]
        if piece.strip():
            # 不把分隔用的空行算進 chunk
            stripped_end = start + len(piece.rstrip("\n"))
            spans.append((start, stripped_end))
        start = end
    return spans


STRATEGIES = {
    "固定 40 字": lambda t: split_fixed(t, 40, 0),
    "固定 40 字＋重疊 10": lambda t: split_fixed(t, 40, 10),
    "段落": lambda t: split_on(t, PARAGRAPH_BREAK),
}


KEY_TERMS = ("2.4 GHz", "5 GHz", "E01", "已配對", "待配對", "已連線", "參考值 25")


def broken_terms(text, chunks):
    """關鍵詞在原文出現、但沒有任何一個 chunk 完整包含它的位置。"""
    broken = []
    for term in KEY_TERMS:
        start = text.find(term)
        while start != -1:
            end = start + len(term)
            if not any(c["start"] <= start and end <= c["end"] for c in chunks):
                broken.append((term, start))
            start = text.find(term, start + 1)
    return sorted(broken, key=lambda item: item[1])


def make_chunks(doc_id, version, text, strategy):
    rows = []
    for index, (start, end) in enumerate(STRATEGIES[strategy](text)):
        body = text[start:end]
        rows.append({
            "chunk_id": f"{doc_id}-v{version}-{index:02d}",
            "document_id": doc_id,
            "source_version": version,
            "start": start,
            "end": end,
            "text": body,
            "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "embedding_revision": EMBEDDING,
        })
    return rows


def cite(docs, chunk):
    """用 offset 回原文核對：版本不對或文字不符就拒絕引用。"""
    version, text = docs[chunk["document_id"]]
    if version != chunk["source_version"]:
        return f"拒絕：chunk 來自 v{chunk['source_version']}，文件現在是 v{version}"
    if text[chunk["start"]:chunk["end"]] != chunk["text"]:
        return "拒絕：offset 對不上原文"
    return "可引用"


def search_space_ok(chunks):
    revisions = {c["embedding_revision"] for c in chunks}
    return len(revisions) == 1, revisions


def main():
    corpus = load_corpus()
    docs = documents(corpus)
    total_chars = sum(len(t) for _, t in docs.values())
    print(f"語料 {corpus['dataset_version']}：{len(docs)} 份文件、原始 {len(corpus['chunks'])} 段、全文共 {total_chars} 字")
    print()

    print("=== 一、三種切法的數量 ===")
    for name in STRATEGIES:
        chunks = [c for doc_id, (v, t) in docs.items() for c in make_chunks(doc_id, v, t, name)]
        stored = sum(len(c["text"]) for c in chunks)
        print(f"  {name:<14} {len(chunks):>3} 個 chunk｜儲存 {stored} 字（原文的 {stored / total_chars:.0%}）")
    print()

    print("=== 二、切壞了什麼（全部 8 份文件）===")
    print(f"  {'切法':<14} {'跨段':>4} {'關鍵詞被切斷':>8}")
    for name in STRATEGIES:
        crossing = broken = 0
        for doc_id, (v, t) in docs.items():
            chunks = make_chunks(doc_id, v, t, name)
            crossing += sum(1 for c in chunks if PARAGRAPH_BREAK in c["text"])
            broken += len(broken_terms(t, chunks))
        print(f"  {name:<14} {crossing:>4} {broken:>8}")
    print()
    print("  D002 用「固定 40 字」切出的 5 個 chunk：")
    for chunk in make_chunks("D002", *docs["D002"], "固定 40 字"):
        print(f"    {chunk['chunk_id'][-2:]} [{chunk['start']:>3},{chunk['end']:>3}) {chunk['text']!r}")
    print("  斷在哪：", "、".join(f"{term}@{pos}" for term, pos in broken_terms(docs["D002"][1], make_chunks("D002", *docs["D002"], "固定 40 字"))))
    print()

    print("=== 三、與 MariaDB 課的 content_hash 對照 ===")
    mariadb = {c["chunk_id"]: hashlib.sha256(c["text"].encode("utf-8")).hexdigest() for c in corpus["chunks"]}
    paragraph = [c for doc_id, (v, t) in docs.items() for c in make_chunks(doc_id, v, t, "段落")]
    same = sum(1 for c in paragraph if c["content_hash"] in mariadb.values())
    print(f"  段落切法 {len(paragraph)} 個 chunk，content_hash 與 MariaDB 課相同的有 {same} 個")
    print()

    print("=== 四、文件改版：D002 開頭加一句之後 ===")
    version, text = docs["D002"]
    old = make_chunks("D002", version, text, "段落")
    new_text = "注意：本頁於第二版更新。" + text
    docs_v2 = dict(docs)
    docs_v2["D002"] = (2, new_text)
    for name in ("固定 40 字", "段落"):
        before = {c["content_hash"] for c in make_chunks("D002", 1, text, name)}
        after = make_chunks("D002", 2, new_text, name)
        reusable = sum(1 for c in after if c["content_hash"] in before)
        print(f"  {name:<10} 改版後 {len(after)} 個 chunk，內容未變可沿用向量的 {reusable} 個")
    print(f"  舊 chunk {old[1]['chunk_id']} 引用檢查：{cite(docs_v2, old[1])}")
    print()

    print("=== 五、混用向量空間 ===")
    mixed = paragraph[:2] + [dict(paragraph[2], embedding_revision="BAAI/bge-small-zh-v1.5@未知")]
    ok, revisions = search_space_ok(mixed)
    print(f"  {'可以' if ok else '拒絕'}一起檢索：出現 {len(revisions)} 種 embedding_revision")


if __name__ == "__main__":
    main()
