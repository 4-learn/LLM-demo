import re

PAGES = [
    (1, ["維修紀錄", "工單編號：WO-2048", "日期：2026-04-02"]),
    (2, ["處理結果：已完成"]),
]


def extract_ticket(pages):
    if not isinstance(pages, (list, tuple)):
        raise ValueError("pages 必須是頁面序列")
    candidates = []
    for page in pages:
        if not isinstance(page, (list, tuple)) or len(page) != 2:
            raise ValueError("每頁必須是 (page_no, lines)")
        page_no, lines = page
        if type(page_no) is not int or page_no < 1:
            raise ValueError("page_no 必須是正整數")
        if not isinstance(lines, (list, tuple)) or not all(isinstance(line, str) for line in lines):
            raise ValueError("lines 必須是字串序列")
        for line in lines:
            match = re.fullmatch(r"工單編號：\s*([A-Z]{2}-\d{4})", line)
            if match:
                candidates.append({"value": match.group(1), "page": page_no, "evidence": line})
    if not candidates:
        return {"status": "missing", "value": None, "evidence": []}
    if len(candidates) != 1:
        return {"status": "ambiguous", "value": None, "evidence": candidates}
    return {"status": "ok", **candidates[0]}


if __name__ == "__main__":
    print(extract_ticket(PAGES))
    print(extract_ticket([(1, ["維修紀錄", "日期：2026-04-02"])]))