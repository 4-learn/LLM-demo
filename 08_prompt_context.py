def plan_context(instructions, documents, budget, count=None):
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if count is None:
        count = len
    if not callable(count):
        raise ValueError("count must be callable")
    used = count(instructions)
    if type(used) is not int or used < 0:
        raise ValueError("count must return a nonnegative int")
    if used > budget:
        raise ValueError("instructions exceed budget")
    kept = []
    for document in documents:
        cost = count(document)
        if type(cost) is not int or cost < 0:
            raise ValueError("count must return a nonnegative int")
        if used + cost <= budget:
            kept.append(document)
            used += cost
    return kept, used

if __name__ == "__main__":
    kept, used = plan_context("任務", ["甲", "乙乙"], 4)
    print("kept:", kept)
    print("used:", used)