"""第 21 節：Metadata 過濾與版本更新。

沿用第 20 節的資料表與 Python 介面（upsert／search／set_status／delete／versions），不寫 SQL。
  一、前過濾 vs 後過濾：先取 k 筆再過濾，合格的答案可能被擠掉
  二、更新一段文字：新版本上線、舊版本 retired；對照「只 upsert、不撤舊版」會發生什麼
  三、未覆核的段落（review=pending）：存得進去，但查不到
  四、刪除與版本切換：刪掉一段、換模型 revision 時新舊不混查
  五、還原：把資料表恢復成第 20 節匯入後的樣子（16 筆），可以重跑

向量不重新計算：更新後的段落用 `data/21-update-vectors.json`（示範機用同一模型算好）。
沒有資料庫時用 fixtures（輸出第一行會寫），持久化與權限待驗證。

    python 21_filter_update.py
    python 21_filter_update.py --fixtures
"""
import argparse
import json
import re
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
K = 3
# C004 的更新版（合成）：把「沒有提供其他頻段的設定步驟」改成指向新的設定頁
UPDATED = {
    "chunk_id": "C004", "source_version": 2,
    "text": "本段為虛構教學資料，不適用真實設備。教學閘道器連線失敗時，先檢查是否選到 2.4 GHz 網路，再核對模擬密碼。"
            "第二版起，若課堂網路只有 5 GHz，請改用教師提供的 2.4 GHz 備用熱點，不要嘗試修改閘道器頻段。",
}
# 一個只能看配件與錯誤代碼文件的角色（合成）：用來示範後過濾的問題
ROLE_DOCS = {"D007", "D008"}
# 未覆核的同學補充（合成）：內容看起來很相關，但沒人審過
PENDING = {
    "chunk_id": "S001", "source_version": 1, "document_id": "D002",
    "text": "（同學補充，未覆核）教學閘道器其實可以用 5 GHz，只要在設定頁打開隱藏選項就好。",
}


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


L20 = lesson("20")
L19 = L20.L19


def load_update_vectors():
    record = json.loads((HERE / "data" / "21-update-vectors.json").read_text(encoding="utf-8"))
    if record["model"] != f"{L19.MODEL_ID}@{L19.REVISION}":
        raise ValueError("更新向量的模型與第 19 節不同")
    for item in (UPDATED, PENDING):
        if record["chunks"][item["chunk_id"]]["content_hash"] != L20.text_hash(item["text"]):
            raise ValueError(f"{item['chunk_id']} 的文字改過，要重新 --encode")
    return {cid: L20.unpack(v["vector"]) for cid, v in record["chunks"].items()}


def make_row(base, item, vector, **changes):
    row = dict(base, source_version=item["source_version"], body=item["text"],
               content_hash=L20.text_hash(item["text"]), embedding=vector)
    row.update({k: item[k] for k in ("chunk_id", "document_id") if k in item})
    row.update(changes)
    return row


def post_filter(store, query, k, allowed):
    """錯誤示範：先不過濾取 k 筆，再丟掉不合格的。"""
    return [(cid, s) for cid, s in store.search(query, k) if cid in allowed]


def show(hits):
    return [f"{cid} {s:.4f}" for cid, s in hits] or "[]"


def restore(store, base_rows):
    """恢復成第 20 節匯入後：16 段、版本 1；刪掉本節加的列。"""
    for cid in ("S001",):
        store.delete(cid)
    store.delete("C004", 2)
    store.upsert(base_rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", action="store_true")
    args = parser.parse_args(argv)

    corpus, cases = L19.load("sop_corpus.json"), L19.load("search_cases.json")
    _, chunks, queries = L20.load_vectors(corpus, cases)
    extra = load_update_vectors()
    store, reason = L20.open_store(args.fixtures)
    print(f"後端：{store.name}｜{reason}")
    base_rows = L20.rows_from(corpus, chunks)
    restore(store, base_rows)
    print(f"起點：第 20 節匯入後的 {store.count()} 筆（每次執行都先還原，可以重跑）")
    allowed = {cid: L19.eligible_chunks(corpus, c["category_id"], L19.PRODUCTS)
               for cid, c in ((c["case_id"], c) for c in cases["cases"])}

    print("\n== 一、前過濾 vs 後過濾（k=3）==")
    lost = [qid for qid in queries if store.search(queries[qid], K, allowed[qid]) != post_filter(store, queries[qid], K, allowed[qid])]
    print(f"  qrels 八題：後過濾結果不同的題數 {len(lost)}／8（這八題的前 3 名剛好都合格，測不出問題）")
    role = {c["chunk_id"] for c in corpus["chunks"] if c["document_id"] in ROLE_DOCS}
    for qid in ("Q01", "Q06"):
        pre = store.search(queries[qid], K, role)
        print(f"  {qid}＋只能看 {'、'.join(sorted(ROLE_DOCS))} 的角色｜前過濾 {show(pre)}｜"
              f"後過濾 {show(post_filter(store, queries[qid], K, role))}｜後過濾取 16 筆再濾 {show(post_filter(store, queries[qid], 16, role)[:K])}")
    print("  後過濾先取 3 筆，3 筆都不合格就變成空的；把候選取大可以補救，但要取多少，事先不知道")
    print("\n== 二、更新 C004（版本 1 → 2）==")
    by_id = {r["chunk_id"]: r for r in base_rows}
    new_row = make_row(by_id["C004"], UPDATED, extra["C004"])
    print("  錯誤做法：只 upsert 新版本，不撤舊版本")
    store.upsert([new_row])
    print(f"    C004 的版本：{store.versions('C004')}")
    hits = store.search(queries["Q02"], K, allowed["Q02"] | {"C004"})
    print(f"    Q02 → {show(hits)}  ← C004 出現 {sum(c == 'C004' for c, _ in hits)} 次（新舊兩版都在）")
    print("  正確做法：新版本上線後，把舊版本設成 retired")
    store.set_status("C004", 1, "retired")
    print(f"    C004 的版本：{store.versions('C004')}")
    hits = store.search(queries["Q02"], K, allowed["Q02"])
    print(f"    Q02 → {show(hits)}  ← C004 只出現 1 次，分數換成新版本的")
    print(f"    Q04 → {show(store.search(queries['Q04'], K, allowed['Q04']))}")
    print(f"  舊版本還在資料表裡（可追溯），只是查不到；資料表共 {store.count()} 筆")

    print("\n== 三、未覆核的段落 ==")
    pending = make_row(by_id["C003"], PENDING, extra["S001"], review="pending")
    store.upsert([pending])
    q03 = allowed["Q03"] | {"S001"}
    print(f"  S001（review=pending）已存入；它與 Q03 的 cosine 分數是 "
          f"{sum(a * b for a, b in zip(extra['S001'], queries['Q03'])):.4f}（比 C003、C004 都高）")
    print(f"  Q03 合格範圍加入 S001 → {show(store.search(queries['Q03'], K, q03))}（review 條件擋掉了）")
    print("  權限（allowed）與覆核（review）是兩件事：這裡刻意讓權限放行 S001，只有 review 擋住它")

    print("\n== 四、刪除與換模型 ==")
    removed = store.delete("S001")
    print(f"  刪除 S001：{removed} 筆；之後 Q03 → {show(store.search(queries['Q03'], K, q03))}")
    other = store.search(queries["Q01"], K, allowed["Q01"], revision="0" * 40)
    print(f"  用新模型的 revision 查（新向量還沒建）→ {len(other)} 筆：不會退回去用舊模型的向量")
    print("  換模型的順序：建好全部新向量 → 驗證排名 → 把查詢的 revision 切過去 → 刪舊向量；不要邊建邊查")

    print("\n== 五、還原 ==")
    restore(store, base_rows)
    print(f"  資料表共 {store.count()} 筆；C004 的版本：{store.versions('C004')}")
    after = L20.run_queries(store, corpus, cases, queries)
    print(f"  8 題排名與第 20 節相同：{'是' if after == L20.memory_baseline(corpus, cases, chunks, queries) else '否'}")
    store.close()


if __name__ == "__main__":
    main()
