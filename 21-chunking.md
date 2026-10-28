# Chunk 與來源資料模型

> **新版正文：20E07-E03-05｜對應新版 LLM 第 19 節：Chunk 與來源資料模型。**
> 你在 MariaDB 課已經把 `sop-v1` 的 16 段文字入庫過——那 16 段是**教師預先切好的**。本節回頭問：如果切法由你決定，怎麼切？每個 chunk 要帶哪些欄位，才能被**引用**、被**更新**、又不會**混用向量空間**？核心主張：**chunk 不是一段文字，是一段文字加上能回到原文的座標。** 只用 Python 標準函式庫，不計算向量。

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先修第 17 節（embedding 模型與 revision）、第 18 節（相似度的限制）；MariaDB 課第 18、22 節（`chunks`、`content_hash`、`source_version`）。

| 時間 | 內容 |
| --- | --- |
| 0–8 分鐘 | 為什麼要切：模型輸入上限、檢索粒度、引用粒度 |
| 8–20 分鐘 | 三種切法：固定長度、重疊、段落；各切壞什麼 |
| 20–30 分鐘 | 來源資料模型：ID、版本、offset、hash、embedding revision |
| 30–35 分鐘 | 跑 Demo：切 8 份文件、改版一份 |
| 35–50 分鐘 | Workshop：調重疊、改版位置、驗證引用 |

驗收能力：交付 **chunk schema 與切分比較**——說明每個欄位的用途、兩種切法各切壞了什麼，並證明改版後舊 chunk 不會被當成現行證據。

> **與 MariaDB 課的分工**（分工見[課綱](https://hackmd.io/SyC69urYGx?type=view)）：本節是 LLM 課獨有。MariaDB 課教的是「欄位怎麼存、交易怎麼保證一致」；本節教的是「切法怎麼影響檢索與引用」。本節沿用該課的語料與欄位名稱，**不重建 schema**。

## 一、為什麼要切

| 理由 | 說明 |
| --- | --- |
| 模型輸入上限 | embedding 模型有最大長度，超過的部分會被截掉（第 17 節） |
| 檢索粒度 | 整份文件一個向量，幾種主題混在一起，相似度會被稀釋 |
| **引用粒度** | 回答要指出「哪一段」支持主張（第 24 節）；整份文件當引用等於沒引用 |

切太大：一個向量混太多主題。切太小：一個 chunk 不足以獨立成立——**否定詞、條件、主詞被切到別的 chunk**。

## 二、三種切法

| 切法 | 怎麼做 | 主要風險 |
| --- | --- | --- |
| 固定長度 | 每 N 字切一刀 | 切在詞中間、句子中間 |
| 固定長度＋重疊 | 每刀往回退 K 字 | 儲存與向量數增加；重疊不保證長句完整 |
| 段落 | 依空行切 | 段落長短不一；依賴原文有好的段落結構 |

沒有一種切法永遠最好。**要比較，就要先定義「切壞」是什麼**——本節用兩個可數的指標：

- **跨段**：一個 chunk 同時包含兩個段落的內容。
- **關鍵詞被切斷**：`2.4 GHz`、`5 GHz`、`E01` 等關鍵詞在原文中出現，但沒有任何一個 chunk 完整包含它。

## 三、來源資料模型

| 欄位 | 用途 | 沒有它會怎樣 |
| --- | --- | --- |
| `chunk_id` | 唯一識別；本節格式 `{document_id}-v{版本}-{序號}` | 無法引用、無法更新 |
| `document_id` | 回到哪份文件 | 只有文字，不知道出處 |
| `source_version` | 來自文件的哪個版本 | 文件改版後，舊 chunk 仍被當成現行證據 |
| `start`、`end` | 在原文中的 offset | 無法回原文核對；無法標示引用位置 |
| `content_hash` | 文字內容的 SHA-256 | 改版後不知道哪些 chunk 內容沒變、哪些向量可以沿用 |
| `embedding_revision` | 向量由哪個模型的哪個 revision 產生 | 換模型後新舊向量混在一起檢索——不同向量空間的距離沒有意義 |

`content_hash` 的算法與 MariaDB 課相同：`SHA-256(UTF-8 文字)`。Demo 第三段會核對兩課算出來的值是否一致。

## 四、可執行 Demo

讀 MariaDB 課的 `data/sop_corpus.json`，把每份文件的兩段用空行接回全文，再用三種方法重切。**所有 chunk 文字一律由 `text[start:end]` 切出**，保證 offset 能回查。

<!-- demo: E07_chunking.py -->
```python
"""第 19 節：Chunk 與來源資料模型（只用 Python 標準函式庫）。

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
EMBEDDING = "BAAI/bge-small-zh-v1.5@7999e1d3359715c523056ef9478215996d62a620"   # 與第 17 節相同


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
```

執行指令（在 [`llm-demo`](https://github.com/4-learn/llm-demo) 目錄下；語料在 `data/sop_corpus.json`，**不需要網路、不需要模型**）：

```bash
python3 E07_chunking.py
```

關鍵輸出（Linux／Python 3.10.15，20E07-E03-05）：

```text
語料 sop-v1：8 份文件、原始 16 段、全文共 1367 字

=== 一、三種切法的數量 ===
  固定 40 字         38 個 chunk｜儲存 1367 字（原文的 100%）
  固定 40 字＋重疊 10   47 個 chunk｜儲存 1757 字（原文的 129%）
  段落              16 個 chunk｜儲存 1351 字（原文的 99%）

=== 二、切壞了什麼（全部 8 份文件）===
  切法               跨段   關鍵詞被切斷
  固定 40 字           8        2
  固定 40 字＋重疊 10     9        0
  段落                0        0

  D002 用「固定 40 字」切出的 5 個 chunk：
    00 [  0, 40) '本段為虛構教學資料，不適用真實設備。教學閘道器 A 與 B 只支援 2.4 GH'
    01 [ 40, 80) 'z 無線網路，不支援 5 GHz。建立模擬連線前，先確認課堂網路名稱對應的頻段，'
    02 [ 80,120) '不要以訊號格數判斷相容性。\n\n本段為虛構教學資料，不適用真實設備。教學閘道器連線'
    03 [120,160) '失敗時，先檢查是否選到 2.4 GHz 網路，再核對模擬密碼。此閘道器不支援 5'
    04 [160,200) ' GHz，改用 5 GHz 不是排除連線失敗的方法；沒有提供其他頻段的設定步驟。'
  斷在哪： 2.4 GHz@34、5 GHz@159

=== 三、與 MariaDB 課的 content_hash 對照 ===
  段落切法 16 個 chunk，content_hash 與 MariaDB 課相同的有 16 個

=== 四、文件改版：D002 開頭加一句之後 ===
  固定 40 字    改版後 6 個 chunk，內容未變可沿用向量的 0 個
  段落         改版後 2 個 chunk，內容未變可沿用向量的 1 個
  舊 chunk D002-v1-01 引用檢查：拒絕：chunk 來自 v1，文件現在是 v2

=== 五、混用向量空間 ===
  拒絕一起檢索：出現 2 種 embedding_revision
```

## 五、結果怎麼讀

### 1. 固定 40 字：D002 的兩個關鍵詞被切斷

看 D002 的 5 個 chunk：

- chunk 00 結尾是 `2.4 GH`，chunk 01 開頭是 `z 無線網路`——**「2.4 GHz」不在任何一個 chunk 裡**。查「2.4 GHz」的檢索，兩個 chunk 都比對不完整。
- chunk 03 結尾是 `此閘道器不支援 5`，chunk 04 開頭是 ` GHz，改用 5 GHz 不是排除連線失敗的方法`。單獨看 chunk 04，它讀起來像在討論「改用 5 GHz」，**「不支援」留在上一個 chunk**。

這正是 MariaDB 課 search_cases 的 Q03／Q04 想測的否定語意。切法本身就可能把否定詞和它否定的對象拆開。

### 2. 重疊 10 字修好了這兩處，代價是 29% 的額外儲存

「固定 40 字＋重疊 10」的關鍵詞切斷數是 0，但 chunk 數從 38 變 47，儲存的字數是原文的 129%——**向量數也多 24%**，檢索成本與儲存成本都跟著增加。

而且重疊只保證**短於重疊長度＋1 的詞**一定完整。重疊 10 字時，11 字以內的詞一定會完整出現在某個 chunk；更長的句子（例如 20 字的「改用 5 GHz 不是排除連線失敗的方法」）**有時完整、有時不完整**，取決於剛好切在哪裡。

### 3. 段落切法：這份語料的最佳解，但有前提

段落切法 16 個 chunk、跨段 0、切斷 0，而且 **content_hash 與 MariaDB 課的 16 段完全相同**——兩課的資料可以互相核對。

但這是因為 `sop-v1` 本來就是教師逐段寫成、每段 77～105 字。真實文件的段落可能是一行標題，也可能是三千字的一大段。段落切法**依賴原文有好的結構**，拿到 PDF 轉出來的文字時要先檢查。

另外注意儲存是原文的 99% 而不是 100%：8 份文件各有 1 處段落空行（2 個字元），共 16 字（1367 − 1351），**不屬於任何 chunk**。這是刻意的，但代表 chunk 的 offset 之間會有空隙。

### 4. 改版：段落切法能沿用一半向量，固定長度一個都不能

D002 開頭加一句 12 字的說明之後：

| 切法 | 改版後 chunk | content_hash 未變 |
| --- | --- | --- |
| 固定 40 字 | 6 | **0** |
| 段落 | 2 | **1** |

固定長度切法的每一刀都**相對於文件開頭**，開頭多 12 字，後面每一刀全部位移，所有 chunk 的內容都變了——**全部向量都要重算**。段落切法只有第一段變了，第二段的 hash 不變，它的向量可以沿用。

但是：**能沿用向量，不代表能沿用舊的 chunk 紀錄**。舊 chunk `D002-v1-01` 的內容雖然與新版第二段相同，它的 `source_version` 是 1，`start` 也已經不對了。引用檢查會拒絕它。正確做法是建立新版 chunk（`D002-v2-01`），用 hash 比對後**複製**向量，而不是讓舊紀錄繼續被檢索到。這與 MariaDB 課第 22 節的版本一致性是同一件事的兩面。

### 5. 混用向量空間

最後一段把其中一個 chunk 的 `embedding_revision` 改掉，檢查立刻拒絕。兩個不同 revision 的向量算出來的 cosine 是一個數字，但**沒有意義**——就像比較兩把不同刻度的尺量出來的長度。檢查要在**檢索之前**做，而不是發現結果怪怪的才回頭查。

## 六、常見誤判

| 誤判 | 為什麼錯 |
| --- | --- |
| 「chunk 就是一段文字」 | 沒有 document_id、版本與 offset，無法引用也無法更新 |
| 「固定長度最簡單，就用它」 | 會切斷關鍵詞與否定詞；文件開頭一改，全部向量重算 |
| 「重疊越多越安全」 | 儲存與向量數線性增加；且只保證短於重疊長度的詞 |
| 「段落切法永遠最好」 | 依賴原文結構；本節語料是刻意寫好的 |
| 「hash 相同就能沿用舊 chunk 紀錄」 | 只能沿用向量；版本與 offset 仍屬於舊版 |
| 「換模型只要重算新文件的向量」 | 新舊向量不能一起檢索；要整批重算後切換 |
| 「之後再補 metadata」 | 入庫時沒記的 revision，事後無法從向量反推 |

## Workshop：切分比較與引用驗證

### 任務

1. 執行 Demo，先預測三種切法在 D002 上各有幾個 chunk，再核對。
2. 新增一種切法「固定 40 字＋重疊 3」，重跑第二段。關鍵詞切斷數變成多少？
3. 把 `"改用 5 GHz 不是排除連線失敗的方法"` 加進 `KEY_TERMS`，用「固定 40 字＋重疊 10」重跑。它有被切斷嗎？**這個結果能推廣到其他長句嗎？**
4. 第四段改成在 D002 **結尾**加一句 `（第二版補充：請勿使用 5 GHz。）`。兩種切法各能沿用幾個向量？跟加在開頭比，為什麼差這麼多？
5. 寫兩個測試：① 未改版的 chunk 引用檢查回傳「可引用」；② 把 chunk 的 `start` 加 1，引用檢查必須拒絕。

### 參考判讀

- 第 1 題：D002 全文 200 字（兩段 93、105 字，加中間 2 字元空行）。固定 40 字 → 5 個；重疊 10 → 每次前進 30 字 → 7 個；段落 → 2 個。
- 第 2 題：切斷數 **1**（從 2 降到 1）。重疊 3 只保證 4 字以內的詞完整；「2.4 GHz」有 7 個字元。
- 第 3 題：**沒有**被切斷——它剛好落在 [150,190) 這個 chunk 裡。但這只是剛好：重疊 10 字只**保證** 11 字以內的詞完整。用程式枚舉 D002 所有**不跨段落**的 12 字連續片段，有 5 個不在任何一個 chunk 裡（若連跨越空行的片段也算則是 6 個）。長句完不完整取決於刀落在哪，**不能從一個例子推廣**。
- 第 4 題：加在結尾時，固定 40 字改版後 6 個 chunk、**沿用 5 個**；段落 2 個 chunk、沿用 1 個。固定長度切法對「開頭改動」非常敏感（每一刀都位移），對「結尾改動」不敏感。段落切法兩種情況都只影響被改的那一段。
- 第 5 題：未改版 → 「可引用」；`start` 加 1 → 「拒絕：offset 對不上原文」。

### 驗收

- 交出 chunk schema，每個欄位寫明用途與「沒有它會怎樣」。
- 至少比較兩種切法，報 chunk 數、儲存比例、跨段數、關鍵詞切斷數，並舉出一個被切壞的具體 chunk。
- 證明改版後的舊 chunk 不會被引用：用測試，不用口頭說明。
- 若沿用 MariaDB 課的資料，content_hash 必須與該課一致；不一致要說明原因。

## 教師紀錄與下一步

- 本節**完全不需要模型、網路或任何 SDK**，也**不計算向量**。`embedding_revision` 只是欄位值，用來示範混用檢查。證據等級：**本機實測（切分與 metadata 規則）**。
- 語料是 MariaDB 課的 `data/sop_corpus.json`，**兩課共用同一份內容**：[llm-demo](https://github.com/4-learn/llm-demo) 裡的檔案由產生器從 MariaDB 課原檔逐字複製，不另行編輯。MariaDB 課若改版語料，重新產生後本節輸出會跟著變；`test_lessons.py` 以 `dataset_version == "sop-v1"` 與 16 個 hash 相符釘住這個依賴。
- 撰寫時的實際紀錄：① 第一版第二段只挑 D002 的含「5 GHz」chunk 印出，篩選條件寫錯，漏印了真正被切斷的 chunk 00；改為對全部 8 份文件**計數**「跨段」與「關鍵詞切斷」，再列出 D002 的全部 chunk。② 改寫時漏了一行變數宣告，第一次執行 `NameError`。③ Workshop 第 3 題原本打算說「重疊 10 能保護長句」，實際枚舉後發現只能保證 11 字以內，改寫為現在的問法。
- 第 20 節接上：把本節段落切法的 16 個 chunk 載入記憶體做精確 top-k；需要第 17 節的模型（下載政策待決）或預先計算的向量 fixtures。
- 20E07-E03-06 更正：原本寫「MariaDB 課第 16 節的 JSONL」，核對 MariaDB 講義後發現第 16 節是 CSV 匯入品質、沒有 JSONL；本節沿用的 `source_version` 來自 MariaDB 第 18、22 節，先修已如此寫明。
