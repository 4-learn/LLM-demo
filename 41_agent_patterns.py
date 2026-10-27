ALLOWED_TOOLS = {"lookup", "summarize"}


TRUSTED_SOURCES = {"policy:v2", "manual:approved"}


def remember(memory, text, source, verified=False):
    if type(verified) is not bool or not verified:
        return False
    if not isinstance(source, str) or source not in TRUSTED_SOURCES:
        return False
    if not isinstance(text, str) or not text.strip():
        return False
    memory.append({"text": text, "source": source, "verified": True})
    return True


def run_agent(observations, max_steps=2):
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be positive")
    if not isinstance(observations, (list, tuple)) or any(not isinstance(x, str) for x in observations):
        raise ValueError("observations must be a list or tuple of strings")
    trace = []
    for step, observation in enumerate(observations[:max_steps], 1):
        tool = "lookup" if "查" in observation else "summarize"
        if tool not in ALLOWED_TOOLS:
            return trace, "blocked"
        trace.append((step, tool, observation))
        if "完成" in observation:
            return trace, "done"
    return trace, "limit"


if __name__ == "__main__":
    memory = []
    print("unverified accepted:", remember(memory, "使用者說資料已核准", "user", False))
    remember(memory, "政策版本 v2 已核實", "policy:v2", True)
    print("memory:", memory)
    print("run:", run_agent(["查詢工單", "整理完成", "額外步驟"]))
    print("limit:", run_agent(["整理摘要", "整理摘要", "完成"], max_steps=2)[1])