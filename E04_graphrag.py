"""離線示範：沿已核實、帶原文證據的圖邊回答兩跳問題。"""
from collections import defaultdict, deque

EDGES = [
    ("林員", "負責", "北區泵浦", "R1", "維護紀錄 R1：林員負責北區泵浦。", True),
    ("北區泵浦", "位於", "北區", "A2", "設備清冊 A2：北區泵浦位於北區。", True),
    ("陳員", "負責", "東區風機", "R2", "維護紀錄 R2：陳員負責東區風機。", True),
    ("東區風機", "位於", "東區", "A3", "設備清冊 A3：東區風機位於東區。", True),
    ("林員", "負責", "東區風機", "D9", "草稿 D9：林員可能負責東區風機。", False),
]


def deduplicate_edges(edges):
    """按明確來源 ID 與三元組去重；狀態衝突時保守採未核實。"""
    merged = {}
    for edge in edges:
        if not isinstance(edge, tuple) or len(edge) != 6:
            raise ValueError("每條邊必須是六欄 tuple")
        a, rel, b, source_id, quote, verified = edge
        if not all(isinstance(x, str) and x.strip() for x in (a, rel, b, source_id, quote)):
            raise ValueError("邊的實體、關係、來源 ID 與原句必須是非空字串")
        if type(verified) is not bool:
            raise ValueError("verified 必須是 bool")
        key = (source_id, a, rel, b)
        if key not in merged:
            merged[key] = (a, rel, b, source_id, quote, verified)
        else:
            previous = merged[key]
            # Duplicate records must never become trusted because of input order:
            # any unverified copy makes the merged claim unverified.
            quotes = " ⟂ ".join(sorted(set(previous[4].split(" ⟂ ")) | {quote}))
            merged[key] = (a, rel, b, source_id, quotes, previous[5] and verified)
    return [merged[key] for key in sorted(merged)]


def supported_paths(start, relation1, relation2, target, edges=EDGES):
    """回傳符合兩段關係、每邊有原文且已核實的路徑。"""
    if not all(isinstance(x, str) and x for x in (start, relation1, relation2, target)):
        raise ValueError("查詢欄位必須是非空字串")
    graph = defaultdict(list)
    for edge in deduplicate_edges(edges):
        if not isinstance(edge, tuple) or len(edge) != 6:
            raise ValueError("每條邊必須是六欄 tuple")
        a, rel, b, source_id, quote, verified = edge
        if not all(isinstance(x, str) and x.strip() for x in (a, rel, b, source_id, quote)):
            raise ValueError("邊的實體、關係、來源 ID 與原句必須是非空字串")
        if type(verified) is not bool:
            raise ValueError("verified 必須是 bool")
        if verified:
            graph[a].append((rel, b, source_id, quote))
    found = []
    for rel, middle, _source1, quote1 in graph[start]:
        if rel != relation1:
            continue
        for rel2, end, _source2, quote2 in graph[middle]:
            if rel2 == relation2 and end == target:
                found.append((middle, quote1, quote2))
    return found


def discover_answers(start, relation1, relation2, edges=EDGES):
    """未知 target 的兩跳查詢：枚舉已出現候選終點並回傳逐路徑證據。"""
    if not all(isinstance(x, str) and x.strip() for x in (start, relation1, relation2)):
        raise ValueError("查詢欄位必須是非空字串")
    validated = deduplicate_edges(edges)
    targets = sorted({end for a, rel, middle, _sid, _q, verified in validated
                      if a == start and rel == relation1 and verified
                      for _a2, rel2, end, _sid2, _q2, verified2 in validated
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


def test_duplicate_verification_merge_is_order_independent():
    unverified = ("林員", "負責", "南區泵浦", "R9", "維護紀錄 R9：未核實。", False)
    verified = ("林員", "負責", "南區泵浦", "R9", "維護紀錄 R9：核實副本。", True)
    first_order = deduplicate_edges([unverified, verified])
    reverse_order = deduplicate_edges([verified, unverified])
    assert first_order == reverse_order
    assert len(first_order) == 1 and first_order[0][5] is False
    assert "未核實" in first_order[0][4] and "核實副本" in first_order[0][4]
    assert not supported_paths("林員", "負責", "位於", "南區", [
        first_order[0],
        ("南區泵浦", "位於", "南區", "A9", "設備清冊 A9：位於南區。", True),
    ])


def main():
    test_duplicate_verification_merge_is_order_independent()
    print("重複邊核實狀態合併：順序獨立，衝突時保守未核實")
    print(answer("林員", "負責", "位於", "北區"))
    print(answer("林員", "負責", "位於", "東區"))
    print("陳員可查答案：", discover_answers("陳員", "負責", "位於"))


if __name__ == "__main__":
    main()