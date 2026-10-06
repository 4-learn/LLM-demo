"""第 04 節：解碼設定對照實驗（只用 Python 標準函式庫）。"""

import math
import random
from collections import Counter

TOKENS = ("missing", "present", "unknown", "<EOS>")
EOS = 3
MAX_NEW_TOKENS = 4
SEED = 20261005
TRIALS = 200

FIRST_LOGITS = (0.2, 2.6, 0.9, -1.0)
FOLLOWUP_LOGITS = {
    0: (0.0, -1.0, -1.0, 3.0),
    1: (-1.0, 0.0, -1.0, 2.0),
    2: (-1.0, -1.0, 0.0, 2.0),
}
# 依證據，第一步的正確答案是 token 0（missing）；分數最高的 token 1（present）是錯的。
CORRECT = 0


def step_logits(history):
    """玩具規則：第一步用固定分數，之後只看最後一個 token。"""
    if not history:
        return FIRST_LOGITS
    if history[-1] not in FOLLOWUP_LOGITS:
        raise ValueError("this toy table only continues from token 0, 1 or 2")
    return FOLLOWUP_LOGITS[history[-1]]


def softmax(logits, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not logits or not all(math.isfinite(x) for x in logits):
        raise ValueError("logits must be finite and non-empty")
    largest = max(logits)
    weights = [math.exp((x - largest) / temperature) for x in logits]
    total = sum(weights)
    return [w / total for w in weights]


def _keep_and_renormalize(probabilities, keep):
    total = sum(probabilities[index] for index in keep)
    if total <= 0:
        raise ValueError("kept probabilities sum to zero")
    return [
        probabilities[index] / total if index in keep else 0.0
        for index in range(len(probabilities))
    ]


def top_k(probabilities, k):
    if k < 1:
        raise ValueError("k must be positive")
    order = sorted(range(len(probabilities)), key=lambda i: -probabilities[i])
    return _keep_and_renormalize(probabilities, set(order[:k]))


def top_p(probabilities, p):
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    order = sorted(range(len(probabilities)), key=lambda i: -probabilities[i])
    keep, cumulative = set(), 0.0
    for index in order:
        keep.add(index)
        cumulative += probabilities[index]
        if cumulative >= p:
            break
    return _keep_and_renormalize(probabilities, keep)


def sample_inverse_cdf(probabilities, rng):
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold < cumulative:
            return index
    return len(probabilities) - 1


def sample_gumbel_max(probabilities, rng):
    best_score, chosen = None, None
    for index, probability in enumerate(probabilities):
        if probability <= 0:
            continue
        score = math.log(probability) - math.log(-math.log(rng.random()))
        if best_score is None or score > best_score:
            best_score, chosen = score, index
    return chosen


def decode_step(logits, *, greedy=False, temperature=1.0, k=None, p=None, rng=None,
                sampler=sample_inverse_cdf):
    probabilities = softmax(logits, temperature)
    if k is not None:
        probabilities = top_k(probabilities, k)
    if p is not None:
        probabilities = top_p(probabilities, p)
    if greedy:
        return max(range(len(probabilities)), key=lambda i: probabilities[i])
    return sampler(probabilities, rng)


def generate(*, rng, max_new_tokens=MAX_NEW_TOKENS, **setting):
    history = []
    for _ in range(max_new_tokens):
        chosen = decode_step(step_logits(history), rng=rng, **setting)
        if chosen == EOS:
            return tuple(history), "eos"
        history.append(chosen)
    return tuple(history), "length"


def run(setting, trials, seed):
    rng = random.Random(seed)
    correct, outputs, reasons = 0, Counter(), Counter()
    for _ in range(trials):
        ids, reason = generate(rng=rng, **setting)
        outputs[ids] += 1
        reasons[reason] += 1
        correct += ids[:1] == (CORRECT,)
    return correct, outputs, reasons


if __name__ == "__main__":
    print("=== 第一步的分布（同一組 logits，只改 temperature）===")
    for temperature in (0.2, 1.0, 2.0):
        values = [round(x, 4) for x in softmax(step_logits([]), temperature)]
        print(f"T={temperature:<4} p={values}")

    print()
    base = softmax(step_logits([]))
    for label, truncated in (("top_k(k=2)", top_k(base, 2)), ("top_p(p=0.90)", top_p(base, 0.9))):
        keep = [i for i, x in enumerate(truncated) if x > 0]
        values = [round(x, 4) for x in truncated]
        print(f"{label:<14} keep={keep} p={values}  token0 機率={truncated[CORRECT]}")

    print()
    print(f"=== {TRIALS} 次解碼（seed={SEED}）===")
    settings = [
        ("greedy", {"greedy": True}),
        ("T=0.2", {"temperature": 0.2}),
        ("T=1.0", {"temperature": 1.0}),
        ("T=2.0", {"temperature": 2.0}),
        ("top_k=2, T=1.0", {"temperature": 1.0, "k": 2}),
        ("top_p=0.90, T=1.0", {"temperature": 1.0, "p": 0.9}),
    ]
    for label, setting in settings:
        correct, outputs, reasons = run(setting, TRIALS, SEED)
        reason_text = ",".join(f"{k}={v}" for k, v in sorted(reasons.items()))
        print(f"{label:<18} 正確 {correct:>3}/{TRIALS} = {correct / TRIALS:6.1%}  "
              f"不重複輸出={len(outputs):<2} 終止={reason_text}")

    print()
    print("=== 重現性：同一 seed、不同實作 ===")
    for label, sampler in (("inverse-CDF", sample_inverse_cdf), ("Gumbel-max", sample_gumbel_max)):
        rng = random.Random(SEED)
        first = [generate(rng=rng, temperature=2.0, sampler=sampler)[0] for _ in range(8)]
        print(f"{label:<12} 前 8 次={first}")
    for label in ("inverse-CDF 重跑", "inverse-CDF 重跑"):
        rng = random.Random(SEED)
        first = [generate(rng=rng, temperature=2.0, sampler=sample_inverse_cdf)[0] for _ in range(8)]
        print(f"{label:<12} 前 8 次={first}")
