import math

def attention(scores, values):
    if not isinstance(scores, (list, tuple)) or not isinstance(values, (list, tuple)):
        raise ValueError("scores and values must be lists or tuples")
    if len(scores) != len(values) or not scores:
        raise ValueError("scores and values must have equal nonzero length")
    if any(type(x) not in (int, float) for x in scores + values):
        raise ValueError("scores and values must be numbers")
    # 轉 float 後檢查，避免超大整數造成 math.isfinite OverflowError。
    try:
        finite = all(math.isfinite(float(x)) for x in scores + values)
    except (OverflowError, TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError("inputs must be finite")
    peak = max(scores)
    exps = [math.exp(x - peak) for x in scores]
    total = sum(exps)
    weights = [x / total for x in exps]
    return weights, sum(w * v for w, v in zip(weights, values))

if __name__ == "__main__":
    weights, result = attention([0.0, 1.0], [10.0, 20.0])
    print("weights:", [round(x, 3) for x in weights])
    print("weighted value:", round(result, 3))