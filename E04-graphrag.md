# 知識圖譜與 GraphRAG

> **正文改版：2026-10-24｜對應新版 LLM 選修 E04：知識圖譜與 GraphRAG。**
> 本節以可查核的圖關係回答跨文件問題，並在沒有足夠證據時拒答；本節不是圖資料庫、向量資料庫或大型模型的效能比較。

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先修建議：第 11 節引用核對、第 22 節檢索基準、第 27 節 RAG；沒修過也可用本節合成例子入門。

| 時間 | 內容 |
|---|---|
| 0–10 分鐘 | 實體、關係、方向與來源證據 |
| 10–20 分鐘 | 路徑檢索、關係支撐、未知判定 |
| 20–35 分鐘 | 執行離線 Demo 並檢視路徑 |
| 35–50 分鐘 | Workshop：查詢改寫、驗收與錯誤分析 |

完成後，學生能交出一條逐邊可回原文的圖路徑，指出斷鏈或否定證據，並讓沒有支持的問題得到「查無答案」，而非補出看似合理的關係。

## 一、圖上的邊不是事實本身

GraphRAG 把文字中的實體與關係表示為節點、帶方向的邊，適合查詢「誰負責哪個設備、設備位於哪個區域」等多跳問題。每一條邊都應保留來源文件、原句、版本與抽取狀態；缺少其中一項便不能把邊當作已核實的事實。抽取模型可能把「未指派給甲」誤抽成「指派給甲」，因此否定、時間範圍、同名實體都需要人工或規則核對。

## 二、路徑證據必須逐段成立

查詢「哪位人員負責東區泵浦？」需要人員→負責→泵浦，以及泵浦→位於→東區兩段證據。只因圖中兩個節點相近、共享標籤，不能推導責任關係。Demo 只沿著明確、已核實的邊走；同時列印每段原句，答案附路徑，而不是只顯示一個名字。若任一段缺失，完整答案即不成立。

## 三、查無答案是有效結果

Demo 內故意沒有「林員負責東區泵浦」這條關係。即使圖裡有林員，也不能把他配給查詢。真實系統還要考量檢索召回不足：查不到不等於現實不存在，所以措辭應是「目前這份證據集查無支持」，並列明查詢範圍及下一步（找來源、補文件或請人覆核）。低信心不應以模型常識補洞。

## 四、可執行 Demo

以下程式僅用 Python 標準函式庫，使用內嵌合成三元組，無模型、資料庫或網路。路徑中的每條邊都攜帶原句和核實狀態；錯誤輸入不會偷偷生成邊。

<!-- demo: E04_graphrag.py -->
```python
"""離線示範：沿已核實、帶原文證據的圖邊回答兩跳問題。"""
from collections import defaultdict, deque

EDGES = [
    ("林員", "負責", "北區泵浦", "維護紀錄 R1：林員負責北區泵浦。", True),
    ("北區泵浦", "位於", "北區", "設備清冊 A2：北區泵浦位於北區。", True),
    ("陳員", "負責", "東區風機", "維護紀錄 R2：陳員負責東區風機。", True),
    ("東區風機", "位於", "東區", "設備清冊 A3：東區風機位於東區。", True),
    ("林員", "負責", "東區風機", "草稿 D9：林員可能負責東區風機。", False),
]


def deduplicate_edges(edges):
    """以來源文件與三元組去重，不把相同來源副本當成獨立佐證。"""
    seen = set()
    unique = []
    for edge in edges:
        if not isinstance(edge, tuple) or len(edge) != 5:
            raise ValueError("每條邊必須是五欄 tuple")
        a, rel, b, quote, verified = edge
        if not all(isinstance(x, str) and x.strip() for x in (a, rel, b, quote)):
            raise ValueError("邊的實體、關係與來源原句必須是非空字串")
        if type(verified) is not bool:
            raise ValueError("verified 必須是 bool")
        # Source identity is the leading record token (e.g. R1, A2, D9).
        source = quote.split("：", 1)[0]
        key = (source, a, rel, b)
        if key not in seen:
            seen.add(key)
            unique.append(edge)
    return unique


def supported_paths(start, relation1, relation2, target, edges=EDGES):
    """回傳符合兩段關係、每邊有原文且已核實的路徑。"""
    if not all(isinstance(x, str) and x for x in (start, relation1, relation2, target)):
        raise ValueError("查詢欄位必須是非空字串")
    graph = defaultdict(list)
    for edge in deduplicate_edges(edges):
        if not isinstance(edge, tuple) or len(edge) != 5:
            raise ValueError("每條邊必須是五欄 tuple")
        a, rel, b, quote, verified = edge
        if not all(isinstance(x, str) and x.strip() for x in (a, rel, b, quote)):
            raise ValueError("邊的實體、關係與來源原句必須是非空字串")
        if type(verified) is not bool:
            raise ValueError("verified 必須是 bool")
        if verified:
            graph[a].append((rel, b, quote))
    found = []
    for rel, middle, quote1 in graph[start]:
        if rel != relation1:
            continue
        for rel2, end, quote2 in graph[middle]:
            if rel2 == relation2 and end == target:
                found.append((middle, quote1, quote2))
    return found


def discover_answers(start, relation1, relation2, edges=EDGES):
    """未知 target 的兩跳查詢：枚舉已出現候選終點並回傳逐路徑證據。"""
    if not all(isinstance(x, str) and x.strip() for x in (start, relation1, relation2)):
        raise ValueError("查詢欄位必須是非空字串")
    validated = deduplicate_edges(edges)
    targets = sorted({end for a, rel, middle, _q, verified in validated
                      if a == start and rel == relation1 and verified
                      for _a2, rel2, end, _q2, verified2 in validated
                      if _a2 == middle and rel2 == relation2 and verified2})
    return [(target, supported_paths(start, relation1, relation2, target, validated))
            for target in targets]


def answer(start, relation1, relation2, target):
    paths = supported_paths(start, relation1, relation2, target)
    if not paths:
        return f"查無答案：目前證據集沒有支持「{start}→{relation1}→{relation2}→{target}」的已核實路徑。"
    if len(paths) > 1:
        return "有歧義：找到多條已核實路徑，請人工比較；不得任取第一條。\n" + "\n".join(
            f"候選：{start}→{middle}→{target}｜證據1：{q1}｜證據2：{q2}"
            for middle, q1, q2 in paths)
    middle, q1, q2 = paths[0]
    return f"有證據：{start}→{middle}→{target}\n  證據1：{q1}\n  證據2：{q2}"


def main():
    print(answer("林員", "負責", "位於", "北區"))
    print(answer("林員", "負責", "位於", "東區"))
    print("陳員可查答案：", discover_answers("陳員", "負責", "位於"))


if __name__ == "__main__":
    main()
```

執行：`python3 E04_graphrag.py`。關鍵輸出（本次本機實測，完整輸出）：

```text
有證據：林員→北區泵浦→北區
  證據1：維護紀錄 R1：林員負責北區泵浦。
  證據2：設備清冊 A2：北區泵浦位於北區。
查無答案：目前證據集沒有支持「林員→負責→位於→東區」的已核實路徑。
陳員可查答案： [('東區', [('東區風機', '維護紀錄 R2：陳員負責東區風機。', '設備清冊 A3：東區風機位於東區。')])]
```

## Workshop：補強多跳查詢

**任務（15 分鐘）**：3 分鐘執行 Demo；5 分鐘呼叫 `discover_answers("陳員", "負責", "位於")`，指出它只在已知節點中列舉目標與路徑；4 分鐘加入同一來源的重複邊，確認路徑證據不重複；3 分鐘加入一筆未核實邊，確認不得成為答案。

**參考判讀**：陳員的兩條核實邊可構成東區風機→東區路徑；重複證據不代表獨立來源，需按文件 ID 去重。草稿邊不應影響輸出。若查詢模板無法表示未知設備名稱，應先列出候選、對每個候選逐邊驗證，不猜答案。

**驗收**：提交修改後的 `supported_paths` 測試，至少包含有效兩跳、缺一段、只有未核實邊、空欄位四種；輸出須展示每條被採納邊的原句與「查無」分支。教師可用 `python3` 執行並確認四種結果符合上述判讀。

## 教師紀錄與下一步

- 證據級別：**本機實測**（2026-10-25；Linux x86_64，Python 3.12；僅標準函式庫；模型 revision 不適用；以 `python3 E04_graphrag.py` 執行）。輸出是此合成圖的確定性結果，不代表 GraphRAG 模型或產品效能。
- 限制：未測圖抽取品質、向量檢索、圖資料庫、實際文件或使用者研究；資料集封閉，拒答不證明外部世界沒有事實。多筆路徑會回傳歧義提示並列出路徑原文，但不做實體消歧、來源獨立性判定或可信度校準。
- 尚未實班試教；50 分鐘配時尚未由真人驗證。教師紀錄證據只支持程式功能與範例輸出，不可外推為學生學習成效。
- 下一步：回到第 27 節 RAG 引用與第 22 節檢索評測，為每條圖邊加入來源版本、時間有效性及獨立測試集。
- 舊教材連結：無；本篇為新增選修，沒有宣稱舊教材已同步。
