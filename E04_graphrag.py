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