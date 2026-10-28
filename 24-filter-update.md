# Metadata 過濾與版本更新

> **正文改版：20E07-E03-06｜對應新版 LLM 第 22 節：Metadata 過濾與版本更新。**
> 第 21 節把向量存進 MariaDB，排名與精確 baseline 相同。資料會變：文件改版、有人補了沒審過的內容、模型要換。本節用同一張表、同一組 Python 介面（`upsert`、`search`、`set_status`、`delete`、`versions`），不寫 SQL，做五件事：比較**前過濾與後過濾**、**更新一段文字**（新版上線、舊版撤下）、擋住**未覆核**的段落、**刪除**、**換模型時不混查**。最後把資料表還原，可以重跑。核心主張：**過濾要在排序之前；更新要撤掉舊版本；權限與覆核是兩件事；每一個情境都要有更新前後的查詢測試。**

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先修第 16 節（metadata）、第 19 節（chunk 版本）、第 21 節（持久化介面）。SQL 的過濾寫法見 MariaDB 課第 21 節，版本一致性與交易見第 22 節；本節聚焦「過濾怎麼改變檢索結果」與怎麼測。

| 時間 | 內容 |
| --- | --- |
| 0–10 分鐘 | 前過濾 vs 後過濾；權限 vs 覆核 |
| 10–25 分鐘 | 更新、刪除、重建 embedding、版本切換 |
| 25–35 分鐘 | 跑 Demo |
| 35–50 分鐘 | Workshop：更新一段文字，寫更新前後的查詢測試 |

驗收能力：交出**更新前後的查詢測試**，證明舊版本不殘留在結果中；fixtures 模式要列出真實服務待驗證的項目。

## 一、前過濾與後過濾

| | 前過濾 | 後過濾 |
| --- | --- | --- |
| 做法 | 先限定合格範圍，再排序取 k 筆 | 先排序取 k 筆，再丟掉不合格的 |
| 結果筆數 | 合格段落夠多就一定有 k 筆 | 可能少於 k 筆，甚至 0 筆 |
| 好處 | 結果正確 | 可以直接用向量索引（ANN）先取候選 |
| 風險 | 合格範圍很小時，近似索引可能找不到（要量 recall） | 合格答案被不合格的高分段落擠掉 |

第 20、21 節都是前過濾（`allowed` 傳進 `search`，SQL 裡是 `WHERE chunk_id IN (...)`）。後過濾常出現在「向量資料庫只回傳 top-k，權限在應用程式裡檢查」的架構。

## 二、權限、覆核、版本是三個不同的欄位

| 條件 | 問的是 | 本課的欄位 | 誰決定 |
| --- | --- | --- | --- |
| 權限 | 這個使用者／這題能不能看這份文件 | `allowed`（第 20 節的類別規則、第 24 節的合格範圍） | 系統規則 |
| 覆核 | 這段內容有沒有被審過、能不能當依據 | `review`（`approved`／`pending`） | 內容負責人 |
| 版本 | 這是不是現行版本 | `status`（`active`／`retired`）、`source_version` | 更新流程 |
| 模型 | 這個向量是哪個模型算的 | `model_revision` | 重建流程 |

一個段落要四個條件都過才能被檢索到。少了任何一個，都可能讓「看起來相關、其實不該用」的內容進到第 24 節的上下文。

## 三、可執行 Demo

需要：同資料夾第 20、21 節的程式（`NN_*.py` 或講義）、`data/sop_corpus.json`、`data/search_cases.json`、`data/21-vectors.json`、`data/22-update-vectors.json`（C004 第二版與 S001 的向量，示範機用同一模型算好）。MariaDB 模式的條件與第 21 節相同；沒有資料庫時用 fixtures。

程式每次執行**先還原**成第 21 節匯入後的 16 筆，結束前再還原一次，所以可以重跑，也不會留下本節的測試資料。

<!-- demo: E08_filter_update.py -->
```python
"""第 22 節：Metadata 過濾與版本更新。

沿用第 21 節的資料表與 Python 介面（upsert／search／set_status／delete／versions），不寫 SQL。
  一、前過濾 vs 後過濾：先取 k 筆再過濾，合格的答案可能被擠掉
  二、更新一段文字：新版本上線、舊版本 retired；對照「只 upsert、不撤舊版」會發生什麼
  三、未覆核的段落（review=pending）：存得進去，但查不到
  四、刪除與版本切換：刪掉一段、換模型 revision 時新舊不混查
  五、還原：把資料表恢復成第 21 節匯入後的樣子（16 筆），可以重跑

向量不重新計算：更新後的段落用 `data/22-update-vectors.json`（示範機用同一模型算好）。
沒有資料庫時用 fixtures（輸出第一行會寫），持久化與權限待驗證。

    python E08_filter_update.py
    python E08_filter_update.py --fixtures
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


L20 = lesson("23")
L19 = lesson("22")


def load_update_vectors():
    record = json.loads((HERE / "data" / "22-update-vectors.json").read_text(encoding="utf-8"))
    if record["model"] != f"{L19.MODEL_ID}@{L19.REVISION}":
        raise ValueError("更新向量的模型與第 20 節不同")
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
    """恢復成第 21 節匯入後：16 段、版本 1；刪掉本節加的列。"""
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
    print(f"起點：第 21 節匯入後的 {store.count()} 筆（每次執行都先還原，可以重跑）")
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
    print(f"  8 題排名與第 21 節相同：{'是' if after == L20.memory_baseline(corpus, cases, chunks, queries) else '否'}")
    store.close()


if __name__ == "__main__":
    main()
```

執行（程式與資料也放在 [llm-demo](https://github.com/4-learn/llm-demo)）：

```bash
python E08_filter_update.py
```

關鍵輸出（示範機 4-learn：Ubuntu 24.04／Python 3.12.3／MariaDB 11.8.9／mariadb 1.1.14，`course_app` 帳號，20E07-E03-06；連跑兩次輸出相同。備課機 fixtures 模式除第一行外逐字相同）：

```text
後端：MariaDB｜連線成功
起點：第 21 節匯入後的 16 筆（每次執行都先還原，可以重跑）

== 一、前過濾 vs 後過濾（k=3）==
  qrels 八題：後過濾結果不同的題數 0／8（這八題的前 3 名剛好都合格，測不出問題）
  Q01＋只能看 D007、D008 的角色｜前過濾 ['C015 0.6947', 'C016 0.6628', 'C013 0.6393']｜後過濾 []｜後過濾取 16 筆再濾 ['C015 0.6947', 'C016 0.6628', 'C013 0.6393']
  Q06＋只能看 D007、D008 的角色｜前過濾 ['C015 0.7794', 'C016 0.7533', 'C013 0.6797']｜後過濾 ['C015 0.7794', 'C016 0.7533']｜後過濾取 16 筆再濾 ['C015 0.7794', 'C016 0.7533', 'C013 0.6797']
  後過濾先取 3 筆，3 筆都不合格就變成空的；把候選取大可以補救，但要取多少，事先不知道

== 二、更新 C004（版本 1 → 2）==
  錯誤做法：只 upsert 新版本，不撤舊版本
    C004 的版本：[(1, 'active', 'approved'), (2, 'active', 'approved')]
    Q02 → ['C004 0.8244', 'C004 0.8230', 'C003 0.7818']  ← C004 出現 2 次（新舊兩版都在）
  正確做法：新版本上線後，把舊版本設成 retired
    C004 的版本：[(1, 'retired', 'approved'), (2, 'active', 'approved')]
    Q02 → ['C004 0.8244', 'C003 0.7818']  ← C004 只出現 1 次，分數換成新版本的
    Q04 → ['C004 0.7657', 'C003 0.7603']
  舊版本還在資料表裡（可追溯），只是查不到；資料表共 17 筆

== 三、未覆核的段落 ==
  S001（review=pending）已存入；它與 Q03 的 cosine 分數是 0.7180（比 C003、C004 都高）
  Q03 合格範圍加入 S001 → ['C003 0.7310', 'C004 0.7129']（review 條件擋掉了）
  權限（allowed）與覆核（review）是兩件事：這裡刻意讓權限放行 S001，只有 review 擋住它

== 四、刪除與換模型 ==
  刪除 S001：1 筆；之後 Q03 → ['C003 0.7310', 'C004 0.7129']
  用新模型的 revision 查（新向量還沒建）→ 0 筆：不會退回去用舊模型的向量
  換模型的順序：建好全部新向量 → 驗證排名 → 把查詢的 revision 切過去 → 刪舊向量；不要邊建邊查

== 五、還原 ==
  資料表共 16 筆；C004 的版本：[(1, 'active', 'approved')]
  8 題排名與第 21 節相同：是
```

## 四、結果怎麼讀

### 1. 八題測不出後過濾的問題

qrels 的八題，前 3 名剛好都是合格段落，後過濾與前過濾結果相同。**測試集沒有涵蓋，不代表沒有問題。**換一個只能看 D007、D008（配件與錯誤代碼）的角色：

- Q01（感測器重設）：不過濾的前 3 名都不屬於這兩份文件，後過濾變成**空的**；前過濾正確回傳 3 筆（雖然不相關，但那是第 24 節「查無答案」要處理的事）。
- Q06：後過濾只剩 2 筆，C013 被擠掉。
- 把候選取大到 16 筆再過濾就對了，但真實資料有幾萬筆時，事先不知道要取多少。

所以 Workshop 要你**自己加一個會讓後過濾失敗的測試案例**，不要只用現成的八題。

### 2. 更新：只 upsert 會留下兩個版本

C004 第二版用新的主鍵 `(C004, 2)` 存入。只 upsert 時，兩個版本都是 `active`，Q02 的結果裡 C004 出現**兩次**，而且舊版還排在 C003 前面。把舊版設成 `retired` 之後，Q02 只剩新版。舊版還在資料表裡（17 筆），可以追溯「當時是用哪一版回答的」。

另一種做法是直接覆寫（同一個主鍵更新文字與向量）：簡單，但無法追溯，也無法在新版有問題時退回。**選哪一種要寫進設計，不要讓它變成意外。**

更新要在同一個交易裡完成（新版寫入、舊版撤下），否則中間有一段時間兩版都在或兩版都不在；交易的寫法見 MariaDB 課第 22 節。本節的 `course_app` 連線用 autocommit，**沒有示範交易**。

### 3. 未覆核：分數最高，但不能用

S001 是一段沒審過的同學補充，內容是錯的（「可以用 5 GHz」），但它和 Q03 的 cosine 分數 0.7180，比 C004 的 0.7129 還高。這裡刻意讓權限放行它，只靠 `review='pending'` 擋住。**權限說「你可以看這份文件」，覆核說「這段內容可以當依據」**，兩者要分開管。

### 4. 刪除與換模型

刪除後再查，S001 不會出現。用新模型的 revision 查，結果是 0 筆：**查不到比查到舊模型的向量好**，因為不同模型的向量距離沒有意義（第 17 節）。換模型的順序：全部新向量建好 → 用第 20 節的方法驗證排名 → 把查詢的 revision 切過去 → 刪除舊向量。

## 五、常見誤判

| 誤判 | 實際 |
| --- | --- |
| 「先取 k 筆再過濾，結果一樣」 | 測試集剛好沒涵蓋；換個角色就變空的 |
| 「upsert 就是更新」 | 新版本換了主鍵，舊版本還是 active |
| 「有權限看，就可以用來回答」 | 覆核是另一個條件 |
| 「換模型時，新舊向量一起查比較保險」 | 不同模型的距離不能比 |
| 「刪掉舊版本最乾淨」 | 無法追溯；用 retired 保留 |

## Workshop：更新一段文字

### 任務

1. （3 分鐘）跑 Demo，記下後端（MariaDB 或 fixtures）。
2. （6 分鐘）寫一個會讓後過濾失敗的案例：選一題、一組 `allowed`，讓前過濾有 3 筆、後過濾少於 3 筆。用 `store.search` 與 `post_filter` 證明。
3. （4 分鐘）在 `main` 的第二段後面加一個斷言：Q02 的結果裡每個 `chunk_id` 只出現一次。把 `set_status("C004", 1, "retired")` 那一行註解掉，確認斷言會失敗。
4. （2 分鐘）寫下：fixtures 模式下，哪幾項待驗證？

### 參考判讀

- 第 2 題：例如 Q02 配上只能看 D007、D008 的角色（不過濾的前 3 名是 C004、C003、C015，只有 C015 合格）。
- 第 3 題：斷言 `len(ids) == len(set(ids))`；拿掉 retired 後 C004 出現兩次，斷言失敗。這就是「更新前後的查詢測試」。
- 第 4 題：持久化、`course_app` 的權限、資料庫的型別與 ENUM 檢查、交易（本節兩種模式都沒示範）。

### 驗收

- 第 2、3 題的程式與輸出（標證據等級與機器）。

## 教師紀錄與下一步

- 證據等級：**遠端實測**（示範機 4-learn 的真實 MariaDB，`course_app` 帳號）＋**本機實測**（備課機 fixtures 模式）。20E07-E03-06 在示範機連跑兩次輸出相同，結束時資料表還原為 16 筆；fixtures 模式除第一行外逐字相同。
- C004 第二版與 S001 是**合成文字**，不是 MariaDB 課的語料；它們只在本節執行期間存在，程式結束前刪除。向量在 `data/22-update-vectors.json`，用第 20 節同一模型與 revision 算出。
- **沒有示範交易**：新版寫入與舊版撤下是兩次 autocommit 的操作。真實系統要放在同一個交易裡（MariaDB 課第 22 節）。
- 後過濾的案例用一個合成的「只能看 D007、D008」角色；qrels 八題本身測不出差異，講義照實寫。
- 舊教材 [RAG 檢索](https://hackmd.io/@yillkid/BJbBtXSjbe) **尚未改寫**。
- 下一步：第 23 節 reranking 與檢索評估。
