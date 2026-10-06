import math

TOKENS = ("建議", "覆核", "立即", "<EOS>")
EOS = 3
LOGITS = {
    (): (2.0, 1.0, 0.0, -1.0),
    (0,): (-1.0, 3.0, 0.0, -2.0),
    (0, 1): (-2.0, -1.0, -3.0, 4.0),
    (2,): (-1.0, 3.0, 0.0, -2.0),
    (2, 1): (-2.0, -1.0, -3.0, 4.0),
}


def softmax(logits, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not logits or not all(math.isfinite(x) for x in logits):
        raise ValueError("logits must be finite and non-empty")
    largest = max(logits)
    weights = [math.exp((x - largest) / temperature) for x in logits]
    total = sum(weights)
    return [x / total for x in weights]


def generate(table, max_new_tokens=5):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    history = []
    trace = []
    for _ in range(max_new_tokens):
        logits = table.get(tuple(history))
        if logits is None:
            raise ValueError("this toy table has no logits for that context")
        if len(logits) != len(TOKENS):
            raise ValueError("one logit per token is required")
        probabilities = softmax(logits)
        chosen = max(range(len(TOKENS)), key=lambda i: logits[i])
        trace.append((tuple(history), probabilities, chosen))
        if chosen == EOS:
            return history, trace, "eos"
        history.append(chosen)
    return history, trace, "length"


if __name__ == "__main__":
    ids, trace, reason = generate(LOGITS)
    for step, (context, probabilities, chosen) in enumerate(trace, 1):
        values = [round(x, 3) for x in probabilities]
        print(f"step={step} context={context} p={values} choose={TOKENS[chosen]}")
    print("output:", "".join(TOKENS[i] for i in ids), "stop:", reason)
    print("short run stop:", generate(LOGITS, max_new_tokens=1)[2])
    for temperature in (0.5, 1.0, 2.0):
        print("T=", temperature, "p=", [round(x, 3) for x in softmax(LOGITS[()], temperature)])
