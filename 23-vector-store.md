# MariaDB Vector 持久化銜接

> **正文改版：20E07-E03-06｜對應新版 LLM 第 21 節：MariaDB Vector 持久化銜接。**
> 第 20 節的精確檢索把 16 段向量放在 Python 的記憶體裡，程式一結束就沒了。本節把同一批向量存進 MariaDB 的 `VECTOR(512)` 欄位，**關掉連線、開一個新的程序再查**，排名要與第 20 節逐題相同。這一梯次的學員在 MariaDB 課第 18–20 節已經學過 schema、入庫與向量索引，本節**不重教 SQL**：資料表由教師預先建立，學員用只能讀寫的 `course_app` 帳號，透過一個 Python 介面（`upsert`、`search`）操作。沒有資料庫時，同一個介面改用記憶體 fixtures，**輸出會明講「沒有測到持久化」**。核心主張：**持久化要用「新程序看得到」來證明；換了儲存層，排名要和精確 baseline 對得上。**

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先修第 19 節（chunk）、第 20 節（精確檢索 baseline，fingerprint `b57e402af0a1c19f`）。本梯次已修 MariaDB 課第 10 節（Python 連線、私人設定檔）、第 18–20 節（向量欄位、排序、索引）；**單獨開班時不要求修過 MariaDB**，用 fixtures 模式。

| 時間 | 內容 |
| --- | --- |
| 0–10 分鐘 | 記憶體 vs 資料庫：什麼叫持久化、怎麼證明 |
| 10–25 分鐘 | 預建 schema、限權帳號、Python 介面 |
| 25–35 分鐘 | 跑 Demo：匯入、查詢、新程序再查、錯誤路徑 |
| 35–50 分鐘 | Workshop：重啟 client 後查詢，與第 20 節比對 |

驗收能力：交出**真實持久化紀錄**（新程序查到的筆數與排名），或 **fixtures 契約測試**；兩者都要寫清楚哪些已驗證、哪些待驗證。

## 一、記憶體與資料庫差在哪

| | 第 20 節：記憶體 | 本節：MariaDB |
| --- | --- | --- |
| 資料在哪 | Python 程序的 numpy 矩陣 | 資料庫檔案 |
| 程式結束後 | 沒了，下次要重新 encode | 還在 |
| 誰能讀 | 只有這個程序 | 有權限的任何程序、任何人 |
| 權限 | 沒有 | 帳號決定能不能讀、寫、刪表 |
| 排序 | `vectors @ q` 後排序 | `ORDER BY VEC_DISTANCE_COSINE(...)` |
| 需要核對 | 矩陣與逐筆計算相同 | **與第 20 節的排名相同** |

「持久化」不能用同一個程序寫完馬上讀來證明：那可能只是讀到快取或還在記憶體裡的物件。Demo 第三段**關掉連線，另開一個 Python 程序**，只查不寫。

## 二、教師預建的資料表

```sql
CREATE TABLE llm_chunk_vectors (
  chunk_id VARCHAR(32) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  source_version INT NOT NULL CHECK (source_version > 0),
  document_id VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  status ENUM('active','retired') NOT NULL DEFAULT 'active',
  review ENUM('approved','pending') NOT NULL DEFAULT 'approved',
  content_hash CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  model_id VARCHAR(100) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  model_revision CHAR(40) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  body TEXT NOT NULL,
  embedding VECTOR(512) NOT NULL,
  PRIMARY KEY (chunk_id, source_version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

| 欄位 | 為什麼要有 |
| --- | --- |
| `(chunk_id, source_version)` 主鍵 | 同一段可以有多個版本（第 22 節更新時用）；重複匯入是更新，不是新增 |
| `status`、`review` | 停用與未覆核的段落也存，但查詢時排除（第 22 節） |
| `content_hash` | 文字改了，向量就要重算；比對雜湊才知道 |
| `model_id`、`model_revision` | 不同模型的向量不能混在一起比距離 |
| `body` | 檢索到之後要給生成模型的原文（第 24 節） |
| `embedding VECTOR(512)` | 維度固定；511 維的向量資料庫會拒絕 |

這張表與 MariaDB 課的 `chunk_vectors` **分開**（名字前面加 `llm_`），不動 MariaDB 課的資料。建表用 `course_editor`（有 CREATE），學員用 `course_app`（只有 SELECT、INSERT、UPDATE、DELETE）；學員**不能**建表或刪表。

距離用 cosine：向量已正規化，所以 `1 − 距離` 就是第 20 節的內積分數。本節**不建向量索引**（16 筆，全量計算就是精確結果）；索引與 `EXPLAIN` 在 MariaDB 課第 20 節，近似檢索會讓排名可能與精確 baseline 不同，要另外量 recall。

## 三、可執行 Demo

需要：同資料夾第 20 節的程式（`E07_*.py` 或講義）、`data/sop_corpus.json`、`data/search_cases.json`、`data/21-vectors.json`（示範機用第 17、20 節同一模型與 revision 算好的向量，所以不必載入模型）。

- **MariaDB 模式**：`pip install mariadb==1.1.14`（MariaDB 課第 10 節同一版），以及私人設定檔 `~/mariadb-course-app.json`（`chmod 600`），教師已建好資料表。
- **fixtures 模式**：只要標準函式庫。沒有資料庫時自動切換，輸出第一行會寫原因；也可以加 `--fixtures` 強制。

<!-- demo: E07_vector_store.py -->
```python
"""第 21 節：MariaDB Vector 持久化銜接。

把第 20 節的 16 段向量存進 MariaDB（`VECTOR(512)`），關掉連線、開一個新的程序再查，排名要與第 20 節的
記憶體精確檢索相同。資料表由教師預先建立（本節不重教 SQL）；學員用 `course_app` 帳號，只能讀寫、不能建表刪表。

兩個後端，同一個介面：
  MariaDB  ：需要 `mariadb==1.1.14` 與 `~/mariadb-course-app.json`（MariaDB 課第 10 節的私人設定檔）
  fixtures ：記憶體裡的同介面替身。**它不能證明持久化**，輸出會明講

向量預設讀 `data/21-vectors.json`（示範機 4-learn 用第 17、20 節同一模型與 revision 算好的 float32），
所以不必載入模型；`--encode` 會重新計算並與檔案比對（需要 sentence-transformers 與模型快取）。

    python E07_vector_store.py                 # 有資料庫用資料庫，沒有就用 fixtures（輸出第一行會寫）
    python E07_vector_store.py --fixtures      # 強制 fixtures
    python E07_vector_store.py --encode        # 重新計算向量並比對
"""
import argparse
import base64
import hashlib
import json
import math
import os
import re
import stat
import struct
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent if __file__ != "<stdin>" else Path.cwd()
TABLE = "llm_chunk_vectors"
DIM = 512
K = 3
SCHEMA = f"""CREATE TABLE {TABLE} (
  chunk_id VARCHAR(32) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  source_version INT NOT NULL CHECK (source_version > 0),
  document_id VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  status ENUM('active','retired') NOT NULL DEFAULT 'active',
  review ENUM('approved','pending') NOT NULL DEFAULT 'approved',
  content_hash CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  model_id VARCHAR(100) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  model_revision CHAR(40) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  body TEXT NOT NULL,
  embedding VECTOR({DIM}) NOT NULL,
  PRIMARY KEY (chunk_id, source_version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"""


def own_source():
    """本檔原始碼：直接執行時是 .py；從講義載入時，從講義的 demo 區塊取出。"""
    path = Path(__file__)
    if path.suffix == ".py" and path.exists():
        return path.read_text(encoding="utf-8")
    lecture = HERE / "23-vector-store.md"
    return re.search(r"<!-- demo: E07_[\w.]+ -->\n```python\n(.*?)\n```", lecture.read_text(encoding="utf-8"), re.S).group(1)


def lesson(nn):
    found = [(p, p.read_text(encoding="utf-8")) for p in sorted(HERE.glob(f"{nn}_*.py"))]
    for md in sorted(HERE.glob(f"{nn}-*.md")):
        found += [(md, src) for _, src in re.findall(r"<!-- demo: ([\w.]+) -->\n```python\n(.*?)\n```",
                                                     md.read_text(encoding="utf-8"), re.S)]
    if not found:
        raise FileNotFoundError(f"找不到第 {nn} 節的程式（{nn}_*.py 或 {nn}-*.md）")
    path, source = found[0]
    module = types.ModuleType(f"lesson{nn}")
    module.__file__ = str(path)
    exec(compile(source, path.name, "exec"), module.__dict__)
    return module


L19 = lesson("22")


def load(name):
    return L19.load(name)


def find_data(name):
    for base in (HERE / "data", HERE, Path.cwd() / "data", Path.cwd(), HERE.parent / "mariadb" / "data"):
        if (base / name).exists():
            return base / name
    raise FileNotFoundError(name)


# ---------------------------------------------------------------- 向量檔

def pack(vector):
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")


def unpack(text):
    raw = base64.b64decode(text)
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_vectors(corpus, cases):
    record = json.loads(find_data("21-vectors.json").read_text(encoding="utf-8"))
    if record["model"] != f"{L19.MODEL_ID}@{L19.REVISION}":
        raise ValueError(f"向量檔的模型 {record['model']} 與第 20 節不同")
    for chunk in corpus["chunks"]:
        if record["chunks"][chunk["chunk_id"]]["content_hash"] != text_hash(chunk["text"]):
            raise ValueError(f"{chunk['chunk_id']} 的文字與向量檔不符：語料改過，要重新 --encode")
    chunks = {cid: unpack(v["vector"]) for cid, v in record["chunks"].items()}
    queries = {qid: unpack(v) for qid, v in record["queries"].items()}
    if {c["case_id"] for c in cases["cases"]} != set(queries):
        raise ValueError("向量檔的查詢與 search_cases.json 不符")
    return record, chunks, queries


def encode(corpus, cases):
    model = L19.load_model()
    chunk_vecs = model.encode([c["text"] for c in corpus["chunks"]], normalize_embeddings=True, show_progress_bar=False)
    query_vecs = model.encode([L19.QUERY_PREFIX + c["query"] for c in cases["cases"]],
                              normalize_embeddings=True, show_progress_bar=False)
    return ({c["chunk_id"]: [float(x) for x in v] for c, v in zip(corpus["chunks"], chunk_vecs)},
            {c["case_id"]: [float(x) for x in v] for c, v in zip(cases["cases"], query_vecs)})


# ---------------------------------------------------------------- 兩個後端，同一個介面

def rows_from(corpus, chunks):
    docs = {d["document_id"]: d for d in corpus["documents"]}
    return [{"chunk_id": c["chunk_id"], "source_version": c["source_version"], "document_id": c["document_id"],
             "status": "active" if docs[c["document_id"]]["status"] == "active" else "retired",
             "review": "approved", "content_hash": text_hash(c["text"]), "model_id": L19.MODEL_ID,
             "model_revision": L19.REVISION, "body": c["text"], "embedding": chunks[c["chunk_id"]]}
            for c in corpus["chunks"]]


class FixtureStore:
    """記憶體替身：行為與 MariaStore 相同，但程序結束資料就沒了。"""
    name = "fixtures（記憶體，未測真實持久化）"

    def __init__(self):
        self.rows = {}

    def upsert(self, rows):
        for row in rows:
            if len(row["embedding"]) != DIM:
                raise ValueError(f"向量維度 {len(row['embedding'])}，資料表要 {DIM}")
            self.rows[(row["chunk_id"], row["source_version"])] = dict(row)

    def count(self):
        return len(self.rows)

    def search(self, query, k, allowed=None, revision=L19.REVISION):
        if allowed is not None and not allowed:
            return []
        hits = []
        for row in self.rows.values():
            if row["status"] != "active" or row["review"] != "approved" or row["model_revision"] != revision:
                continue
            if allowed is not None and row["chunk_id"] not in allowed:
                continue
            dot = sum(a * b for a, b in zip(row["embedding"], query))
            norm = math.sqrt(sum(a * a for a in row["embedding"])) * math.sqrt(sum(b * b for b in query))
            hits.append((row["chunk_id"], 1 - dot / norm))
        hits.sort(key=lambda h: (h[1], h[0]))
        return [(cid, 1 - d) for cid, d in hits[:k]]

    def set_status(self, chunk_id, source_version, status):
        self.rows[(chunk_id, source_version)]["status"] = status

    def delete(self, chunk_id, source_version=None):
        doomed = [key for key in self.rows if key[0] == chunk_id and source_version in (None, key[1])]
        for key in doomed:
            del self.rows[key]
        return len(doomed)

    def versions(self, chunk_id):
        return sorted((v, r["status"], r["review"]) for (c, v), r in self.rows.items() if c == chunk_id)

    def drop_table(self):
        raise PermissionError("fixtures 沒有權限模型：無法示範 course_app 不能刪表")

    def close(self):
        pass


def read_config(path=None):
    path = Path(path or os.environ.get("COURSE_DB_CONFIG") or "~/mariadb-course-app.json").expanduser()
    with path.open(encoding="utf-8") as source:
        if stat.S_IMODE(os.fstat(source.fileno()).st_mode) & 0o077:
            raise ValueError(f"{path} 必須是私人檔案（chmod 600）")
        return json.load(source)


class MariaStore:
    name = "MariaDB"

    def __init__(self, config):
        import mariadb

        self.error = mariadb.Error
        self.conn = mariadb.connect(**config, autocommit=True)
        self.cur = self.conn.cursor()
        self.cur.execute("SELECT CURRENT_USER(), DATABASE(), VERSION()")
        self.account, self.database, self.version = self.cur.fetchone()
        self.cur.execute("SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE() "
                         "AND TABLE_NAME = ?", (TABLE,))
        if self.cur.fetchone()[0] != 1:
            raise LookupError(f"資料表 {TABLE} 不存在：請教師用 course_editor 執行講義第三段的 CREATE TABLE")

    def upsert(self, rows):
        sql = (f"INSERT INTO {TABLE} (chunk_id, source_version, document_id, status, review, content_hash, "
               "model_id, model_revision, body, embedding) VALUES (?,?,?,?,?,?,?,?,?, VEC_FromText(?)) "
               "ON DUPLICATE KEY UPDATE document_id=VALUES(document_id), status=VALUES(status), "
               "review=VALUES(review), content_hash=VALUES(content_hash), model_id=VALUES(model_id), "
               "model_revision=VALUES(model_revision), body=VALUES(body), embedding=VALUES(embedding)")
        self.cur.executemany(sql, [(r["chunk_id"], r["source_version"], r["document_id"], r["status"], r["review"],
                                    r["content_hash"], r["model_id"], r["model_revision"], r["body"],
                                    json.dumps(r["embedding"])) for r in rows])

    def count(self):
        self.cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
        return self.cur.fetchone()[0]

    def search(self, query, k, allowed=None, revision=L19.REVISION):
        if allowed is not None and not allowed:
            return []                                  # 空集合不能寫成 IN ()，也不該查
        where, params = "", [json.dumps(query), revision]
        if allowed is not None:
            where = f" AND chunk_id IN ({','.join('?' * len(allowed))})"
            params += sorted(allowed)
        self.cur.execute(
            f"SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, VEC_FromText(?)) AS d FROM {TABLE} "
            f"WHERE status='active' AND review='approved' AND model_revision=?{where} "
            f"ORDER BY d, chunk_id LIMIT ?", params + [k])
        return [(cid, 1 - d) for cid, d in self.cur.fetchall()]

    def set_status(self, chunk_id, source_version, status):
        self.cur.execute(f"UPDATE {TABLE} SET status=? WHERE chunk_id=? AND source_version=?",
                         (status, chunk_id, source_version))

    def delete(self, chunk_id, source_version=None):
        if source_version is None:
            self.cur.execute(f"DELETE FROM {TABLE} WHERE chunk_id=?", (chunk_id,))
        else:
            self.cur.execute(f"DELETE FROM {TABLE} WHERE chunk_id=? AND source_version=?", (chunk_id, source_version))
        return self.cur.rowcount

    def versions(self, chunk_id):
        self.cur.execute(f"SELECT source_version, status, review FROM {TABLE} WHERE chunk_id=? ORDER BY source_version",
                         (chunk_id,))
        return [tuple(row) for row in self.cur.fetchall()]

    def drop_table(self):
        self.cur.execute(f"DROP TABLE {TABLE}")

    def close(self):
        self.cur.close()
        self.conn.close()


def open_store(force_fixtures=False):
    """回傳 (store, 為什麼用這個後端)。"""
    if force_fixtures:
        return FixtureStore(), "指定 --fixtures"
    try:
        return MariaStore(read_config()), "連線成功"
    except ImportError:
        reason = "沒有安裝 mariadb 套件"
    except FileNotFoundError:
        reason = "找不到 ~/mariadb-course-app.json"
    except LookupError as error:
        reason = str(error)
    except Exception as error:                               # 連線、權限、設定檔錯誤
        reason = f"{type(error).__name__}: {str(error)[:80]}"
    return FixtureStore(), f"無法使用 MariaDB（{reason}）"


# ---------------------------------------------------------------- 查詢與比對

def run_queries(store, corpus, cases, queries):
    out = {}
    for case in cases["cases"]:
        allowed = L19.eligible_chunks(corpus, case["category_id"], L19.PRODUCTS)
        out[case["case_id"]] = [[cid, round(score, 4)] for cid, score in store.search(queries[case["case_id"]], K, allowed)]
    return out


def memory_baseline(corpus, cases, chunks, queries):
    """第 20 節的做法（全量內積後排序），用純 Python 算，當作答案。"""
    out = {}
    for case in cases["cases"]:
        allowed = L19.eligible_chunks(corpus, case["category_id"], L19.PRODUCTS)
        q = queries[case["case_id"]]
        rows = sorted(((cid, sum(a * b for a, b in zip(v, q))) for cid, v in chunks.items() if cid in allowed),
                      key=lambda r: (-r[1], r[0]))
        out[case["case_id"]] = [[cid, round(s, 4)] for cid, s in rows[:K]]
    return out


def fingerprint_like_19(corpus, cases, rankings):
    return L19.fingerprint({"model": f"{L19.MODEL_ID}@{L19.REVISION}", "dataset_version": corpus["dataset_version"],
                            "annotation_version": cases["annotation_version"], "k": K, "rankings": rankings})


def frozen_E07_fingerprint():
    lecture = HERE / "22-exact-baseline.md"
    found = re.search(r"fingerprint: (\w+)", lecture.read_text(encoding="utf-8")) if lecture else None
    return found.group(1) if found else None


# ---------------------------------------------------------------- 主程式

def query_only(force_fixtures):
    """給「重啟 client」用：新程序、新連線，不寫入，只查。輸出 JSON。"""
    corpus, cases = L19.load("sop_corpus.json"), L19.load("search_cases.json")
    _, _, queries = load_vectors(corpus, cases)
    store, _ = open_store(force_fixtures)
    print(json.dumps({"count": store.count(), "rankings": run_queries(store, corpus, cases, queries)}))
    store.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", action="store_true")
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--save", action="store_true", help="與 --encode 一起用：寫回 data/21-vectors.json")
    parser.add_argument("--query-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.query_only:
        return query_only(args.fixtures)

    corpus, cases = L19.load("sop_corpus.json"), L19.load("search_cases.json")
    if args.encode:
        chunks, queries = encode(corpus, cases)
        if args.save:
            record = {"model": f"{L19.MODEL_ID}@{L19.REVISION}", "query_prefix": L19.QUERY_PREFIX,
                      "dataset_version": corpus["dataset_version"], "annotation_version": cases["annotation_version"],
                      "format": "base64 little-endian float32，已正規化",
                      "chunks": {cid: {"content_hash": text_hash(next(c["text"] for c in corpus["chunks"]
                                                                       if c["chunk_id"] == cid)), "vector": pack(v)}
                                 for cid, v in chunks.items()},
                      "queries": {qid: pack(v) for qid, v in queries.items()}}
            find_data("21-vectors.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n",
                                                           encoding="utf-8")
            print("已寫入 data/21-vectors.json")
        _, saved_chunks, saved_queries = load_vectors(corpus, cases)
        diff = max(abs(a - b) for cid in chunks for a, b in zip(chunks[cid], saved_chunks[cid]))
        diff = max([diff] + [abs(a - b) for qid in queries for a, b in zip(queries[qid], saved_queries[qid])])
        print(f"重新計算的向量與 data/21-vectors.json 最大差 {diff:.2e}")
        return None

    record, chunks, queries = load_vectors(corpus, cases)
    store, reason = open_store(args.fixtures)
    print(f"後端：{store.name}｜{reason}")
    if isinstance(store, MariaStore):
        print(f"  帳號 {store.account}｜資料庫 {store.database}｜MariaDB {store.version}")
    print(f"向量：data/21-vectors.json｜{record['model']}｜{len(chunks)} 段＋{len(queries)} 題｜{DIM} 維 float32")

    print("\n== 一、匯入（upsert）==")
    rows = rows_from(corpus, chunks)
    for attempt in (1, 2):
        store.upsert(rows)
        print(f"  第 {attempt} 次匯入 {len(rows)} 筆後，資料表共 {store.count()} 筆")
    print("  主鍵是 (chunk_id, source_version)：重複匯入是更新，不會變成 32 筆")
    retired = [r["chunk_id"] for r in rows if r["status"] == "retired"]
    print(f"  文件已停用的段落也存進去、標 retired：{'、'.join(retired)}（查詢時排除，不是不存）")

    print(f"\n== 二、查詢（k={K}，先用第 20 節的合格規則過濾）==")
    here = run_queries(store, corpus, cases, queries)
    baseline = memory_baseline(corpus, cases, chunks, queries)
    for qid, hits in here.items():
        print(f"  {qid} → {[f'{cid} {s:.4f}' for cid, s in hits] or '[]（合格範圍為空，沒有送出查詢）'}")
    same = sum(here[q] == baseline[q] for q in here)
    print(f"  與記憶體精確排名（第 20 節做法）相同：{same}/{len(here)} 題")
    fp, frozen = fingerprint_like_19(corpus, cases, here), frozen_E07_fingerprint()
    print(f"  fingerprint {fp}；第 20 節講義 {frozen}：{'相同' if fp == frozen else '不同'}")

    print("\n== 三、關掉連線，開新程序再查 ==")
    store.close()
    child = subprocess.run([sys.executable, "-", "--query-only"] + (["--fixtures"] if args.fixtures else []),
                           input=own_source(), capture_output=True, text=True, check=True, cwd=HERE)
    after = json.loads(child.stdout)
    if isinstance(store, MariaStore):
        print(f"  新程序看到 {after['count']} 筆；8 題排名與關閉前相同：{'是' if after['rankings'] == here else '否'}")
        print("  這是真實持久化：資料在資料庫裡，不在這個 Python 程序的記憶體")
    else:
        print(f"  新程序看到 {after['count']} 筆（fixtures 只活在原本的程序裡）")
        print("  **本次沒有測到持久化**：報告要寫「fixtures 契約測試，真實持久化待驗證」")

    print("\n== 四、錯誤路徑 ==")
    store, _ = open_store(args.fixtures)
    bad = dict(rows[0], embedding=rows[0]["embedding"][:DIM - 1])
    try:
        store.upsert([bad])
        print("  511 維向量：被接受了？")
    except Exception as error:
        print(f"  511 維向量 → 拒絕（{type(error).__name__}）")
    other = store.search(queries["Q01"], K, None, revision="0" * 40)
    print(f"  用別的 model_revision 查 → {len(other)} 筆（不同模型的向量不能混查）")
    try:
        store.drop_table()
        print("  course_app 刪表：成功了？權限設錯了")
    except PermissionError as error:
        print(f"  刪表 → {error}")
    except Exception as error:
        print(f"  course_app 刪表 → 拒絕（{type(error).__name__}：{str(error).split('(')[0].strip()[:60]}）")
    print(f"  最後資料表共 {store.count()} 筆")
    store.close()


if __name__ == "__main__":
    main()
```

執行（程式與資料也放在 [llm-demo](https://github.com/4-learn/llm-demo)）：

```bash
python E07_vector_store.py
```

關鍵輸出（示範機 4-learn：Ubuntu 24.04／Python 3.12.3／MariaDB 11.8.9／mariadb 1.1.14，`course_app` 帳號，20E07-E03-06）：

```text
後端：MariaDB｜連線成功
  帳號 course_app@localhost｜資料庫 mariadb_workshop_2026｜MariaDB 11.8.9-MariaDB-ubu2404
向量：data/21-vectors.json｜BAAI/bge-small-zh-v1.5@7999e1d3359715c523056ef9478215996d62a620｜16 段＋8 題｜512 維 float32

== 一、匯入（upsert）==
  第 1 次匯入 16 筆後，資料表共 16 筆
  第 2 次匯入 16 筆後，資料表共 16 筆
  主鍵是 (chunk_id, source_version)：重複匯入是更新，不會變成 32 筆
  文件已停用的段落也存進去、標 retired：C005、C006（查詢時排除，不是不存）

== 二、查詢（k=3，先用第 20 節的合格規則過濾）==
  Q01 → ['C001 0.7841', 'C002 0.7253', 'C011 0.7019']
  Q02 → ['C004 0.8230', 'C003 0.7818']
  Q03 → ['C004 0.7453', 'C003 0.7310']
  Q04 → ['C004 0.8317', 'C003 0.7603']
  Q05 → ['C001 0.6412', 'C007 0.5942', 'C011 0.5881']
  Q06 → ['C015 0.7794', 'C007 0.7561', 'C016 0.7533']
  Q07 → ['C014 0.8096', 'C013 0.7049']
  Q08 → []（合格範圍為空，沒有送出查詢）
  與記憶體精確排名（第 20 節做法）相同：8/8 題
  fingerprint b57e402af0a1c19f；第 20 節講義 b57e402af0a1c19f：相同

== 三、關掉連線，開新程序再查 ==
  新程序看到 16 筆；8 題排名與關閉前相同：是
  這是真實持久化：資料在資料庫裡，不在這個 Python 程序的記憶體

== 四、錯誤路徑 ==
  511 維向量 → 拒絕（OperationalError）
  用別的 model_revision 查 → 0 筆（不同模型的向量不能混查）
  course_app 刪表 → 拒絕（OperationalError：DROP command denied to user 'course_app'@'localhost' for tab）
  最後資料表共 16 筆
```

沒有資料庫時（fixtures 模式，備課機 Python 3.10.15 實跑），第一、二段逐字相同，第三、四段變成：

```text
== 三、關掉連線，開新程序再查 ==
  新程序看到 0 筆（fixtures 只活在原本的程序裡）
  **本次沒有測到持久化**：報告要寫「fixtures 契約測試，真實持久化待驗證」

== 四、錯誤路徑 ==
  511 維向量 → 拒絕（ValueError）
  用別的 model_revision 查 → 0 筆（不同模型的向量不能混查）
  刪表 → fixtures 沒有權限模型：無法示範 course_app 不能刪表
  最後資料表共 0 筆
```

## 四、結果怎麼讀

### 1. 匯入兩次還是 16 筆

主鍵是 `(chunk_id, source_version)`，`INSERT ... ON DUPLICATE KEY UPDATE` 讓重複匯入變成更新。如果主鍵只有自動遞增的 id，跑兩次就會有 32 筆，而且每段都會被檢索到兩次。停用文件 D003 的 C005、C006 也存進去、標 `retired`：**不存**和**存了但不查**是不同的決定，後者才能追溯。

### 2. 排名與第 20 節相同

8 題逐題與記憶體精確排名相同，fingerprint 也和第 20 節講義的 `b57e402af0a1c19f` 一樣。分數四捨五入到小數 4 位後相同；MariaDB 用 float32 存，計算順序和 numpy 不同，小數更後面的位數可能有差，但不影響排名。如果排名不同，先懷疑：向量是不是同一個模型與 revision、過濾條件是否相同、同分時的第二排序鍵（`chunk_id`）有沒有寫。

Q08 的合格範圍是空的：程式**沒有送出查詢**。`IN ()` 在 SQL 是語法錯誤；更重要的是，空範圍代表「沒有可以回答的資料」（第 24 節轉人工），不是「查全部」。

### 3. 新程序看得到，才叫持久化

MariaDB 模式下，新程序看到 16 筆、排名相同。fixtures 模式下，新程序看到 0 筆：**fixtures 能驗證介面與排名邏輯，不能驗證持久化**，報告要照寫。

### 4. 錯誤路徑

- 511 維：MariaDB 由 `VECTOR(512)` 拒絕；fixtures 由程式檢查拒絕。兩邊例外型別不同（`OperationalError` vs `ValueError`），所以呼叫端不要只攔某一種資料庫例外。
- 別的 `model_revision`：0 筆。WHERE 條件裡固定帶 revision，換模型時舊向量不會被混查（第 22 節的版本切換）。
- `course_app` 刪表：資料庫拒絕。fixtures **沒有權限模型**，這一項只能在真實資料庫驗證。

## 五、常見誤判

| 誤判 | 實際 |
| --- | --- |
| 「寫進去馬上讀得到，就是持久化」 | 要換一個程序、新的連線再讀 |
| 「fixtures 跑過了，資料庫一定也行」 | fixtures 不能驗證持久化、權限、型別檢查 |
| 「換成資料庫，檢索品質會變好」 | 排名與第 20 節相同；儲存層不改變向量 |
| 「合格範圍為空就查全部」 | 空範圍代表沒有可回答的資料 |
| 「學員帳號給 ALL 比較方便」 | 學員只需要讀寫；建表刪表是教師的事 |

## Workshop：重啟 client 後查詢

### 任務

1. （4 分鐘）跑 Demo，記下第一行的後端與原因。如果是 fixtures，原因是什麼？
2. （4 分鐘）關掉終端機，開一個新的終端機，執行 `python E07_vector_store.py --query-only`。看到幾筆？排名和第 20 節講義相同嗎？
3. （4 分鐘）把 `rows_from` 裡第一段的 `source_version` 改成 2，再匯入一次。資料表變成幾筆？查詢結果有沒有重複？為什麼？（做完把它改回 1，並請教師用 `course_editor` 刪掉多出的那一筆）
4. （3 分鐘）寫下你的證據等級：真實持久化，還是 fixtures 契約測試？待驗證的是哪幾項？

### 參考判讀

- 第 2 題：MariaDB 模式看到 16 筆、排名相同；fixtures 模式是 0 筆、8 題全空。後者要寫「真實持久化待驗證」。
- 第 3 題：17 筆，因為 `(C001, 2)` 是新的主鍵。查詢時 C001 會出現兩次（兩個版本都是 active）。這正是第 22 節要處理的問題：更新時要把舊版本設成 `retired`。
- 第 4 題：fixtures 模式下待驗證的是持久化、權限（刪表）、資料庫的維度檢查。

### 驗收

- 第 2 題的輸出（標證據等級與機器）。
- 第 3 題的筆數與解釋。

## 教師紀錄與下一步

- 證據等級：**遠端實測**（示範機 4-learn 的真實 MariaDB）＋**本機實測**（備課機的 fixtures 模式）。20E07-E03-06 在示範機以 `course_app` 帳號執行：匯入、8 題排名與第 20 節 fingerprint 相同、新程序查到 16 筆、511 維被拒、`course_app` 刪表被拒。備課機沒有 MariaDB，fixtures 模式第一、二段與示範機逐字相同。
- 環境準備（教師）：
  - 示範機 4-learn 上，`llm_chunk_vectors` 由 `course_editor` 在 `mariadb_workshop_2026` 建立（講義第二段的 SQL），只建一次；不加 `IF NOT EXISTS`，已存在就會報錯，不會隱藏結構不符。
  - 示範機的 venv 已安裝 `mariadb==1.1.14`（20E07-E03-06）。
  - 沿用 MariaDB 課的 `course_app` 與 `~/mariadb-course-app.json`，沒有新增帳號；先前提議的 `~/llm-course.cnf` 不需要了。
  - 要清掉重來：`course_editor` 執行 `DROP TABLE llm_chunk_vectors`，再重建。
- 向量檔 `data/21-vectors.json` 用第 20 節的模型與 revision 算出；`--encode` 重新計算，備課機與檔案最大差 0。語料文字一改，`content_hash` 對不上，程式會要求重新計算。
- 撰寫時的實際紀錄：原本打算另建 LLM 專用帳號，需要 sudo；後來查到示範機已有 MariaDB 課的 `course_editor`（可在 workshop 庫建表）與 `course_app`（只能讀寫），改為沿用，不需要 sudo。
- `course_reader` 的密碼與 MariaDB 課的設定檔不符（示範機登入失敗），本節沒有用到它，**尚未修正**。
- 學員的 i5／16 GB 機器、Windows／WSL 上的 MariaDB 連線：**待驗證**。
- 舊教材 [RAG 檢索](https://hackmd.io/@yillkid/BJbBtXSjbe) 儲存段落**尚未改寫**。
- 下一步：第 22 節 metadata 過濾與版本更新，用同一張表。
