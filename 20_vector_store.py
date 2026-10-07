"""第 20 節：MariaDB Vector 持久化銜接。

把第 19 節的 16 段向量存進 MariaDB（`VECTOR(512)`），關掉連線、開一個新的程序再查，排名要與第 19 節的
記憶體精確檢索相同。資料表由教師預先建立（本節不重教 SQL）；學員用 `course_app` 帳號，只能讀寫、不能建表刪表。

兩個後端，同一個介面：
  MariaDB  ：需要 `mariadb==1.1.14` 與 `~/mariadb-course-app.json`（MariaDB 課第 10 節的私人設定檔）
  fixtures ：記憶體裡的同介面替身。**它不能證明持久化**，輸出會明講

向量預設讀 `data/20-vectors.json`（示範機 4-learn 用第 16、19 節同一模型與 revision 算好的 float32），
所以不必載入模型；`--encode` 會重新計算並與檔案比對（需要 sentence-transformers 與模型快取）。

    python 20_vector_store.py                 # 有資料庫用資料庫，沒有就用 fixtures（輸出第一行會寫）
    python 20_vector_store.py --fixtures      # 強制 fixtures
    python 20_vector_store.py --encode        # 重新計算向量並比對
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
    lecture = next(iter(sorted(HERE.glob("20-*.md"))))
    return re.search(r"<!-- demo: 20_[\w.]+ -->\n```python\n(.*?)\n```", lecture.read_text(encoding="utf-8"), re.S).group(1)


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


L19 = lesson("19")


# ---------------------------------------------------------------- 向量檔

def pack(vector):
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")


def unpack(text):
    raw = base64.b64decode(text)
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_vectors(corpus, cases):
    record = json.loads((HERE / "data" / "20-vectors.json").read_text(encoding="utf-8"))
    if record["model"] != f"{L19.MODEL_ID}@{L19.REVISION}":
        raise ValueError(f"向量檔的模型 {record['model']} 與第 19 節不同")
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
    """第 19 節的做法（全量內積後排序），用純 Python 算，當作答案。"""
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


def frozen_19_fingerprint():
    lecture = next(iter(sorted(HERE.glob("19-*.md"))), None)
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
    parser.add_argument("--save", action="store_true", help="與 --encode 一起用：寫回 data/20-vectors.json")
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
            (HERE / "data" / "20-vectors.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n",
                                                           encoding="utf-8")
            print("已寫入 data/20-vectors.json")
        _, saved_chunks, saved_queries = load_vectors(corpus, cases)
        diff = max(abs(a - b) for cid in chunks for a, b in zip(chunks[cid], saved_chunks[cid]))
        diff = max([diff] + [abs(a - b) for qid in queries for a, b in zip(queries[qid], saved_queries[qid])])
        print(f"重新計算的向量與 data/20-vectors.json 最大差 {diff:.2e}")
        return None

    record, chunks, queries = load_vectors(corpus, cases)
    store, reason = open_store(args.fixtures)
    print(f"後端：{store.name}｜{reason}")
    if isinstance(store, MariaStore):
        print(f"  帳號 {store.account}｜資料庫 {store.database}｜MariaDB {store.version}")
    print(f"向量：data/20-vectors.json｜{record['model']}｜{len(chunks)} 段＋{len(queries)} 題｜{DIM} 維 float32")

    print("\n== 一、匯入（upsert）==")
    rows = rows_from(corpus, chunks)
    for attempt in (1, 2):
        store.upsert(rows)
        print(f"  第 {attempt} 次匯入 {len(rows)} 筆後，資料表共 {store.count()} 筆")
    print("  主鍵是 (chunk_id, source_version)：重複匯入是更新，不會變成 32 筆")
    retired = [r["chunk_id"] for r in rows if r["status"] == "retired"]
    print(f"  文件已停用的段落也存進去、標 retired：{'、'.join(retired)}（查詢時排除，不是不存）")

    print(f"\n== 二、查詢（k={K}，先用第 19 節的合格規則過濾）==")
    here = run_queries(store, corpus, cases, queries)
    baseline = memory_baseline(corpus, cases, chunks, queries)
    for qid, hits in here.items():
        print(f"  {qid} → {[f'{cid} {s:.4f}' for cid, s in hits] or '[]（合格範圍為空，沒有送出查詢）'}")
    same = sum(here[q] == baseline[q] for q in here)
    print(f"  與記憶體精確排名（第 19 節做法）相同：{same}/{len(here)} 題")
    fp, frozen = fingerprint_like_19(corpus, cases, here), frozen_19_fingerprint()
    print(f"  fingerprint {fp}；第 19 節講義 {frozen}：{'相同' if fp == frozen else '不同'}")

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
