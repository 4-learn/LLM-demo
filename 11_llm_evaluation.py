def binary_metrics(truth, predicted):
    if len(truth) != len(predicted) or not truth:
        raise ValueError("nonempty inputs must have equal length")
    if any(type(x) is not int or x not in (0, 1) for x in truth + predicted):
        raise ValueError("labels must be integer 0 or 1")
    tp = sum(t == 1 and p == 1 for t, p in zip(truth, predicted))
    fp = sum(t == 0 and p == 1 for t, p in zip(truth, predicted))
    fn = sum(t == 1 and p == 0 for t, p in zip(truth, predicted))
    return tp, fp, fn, tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0

if __name__ == "__main__":
    print(binary_metrics([1, 1, 0], [1, 0, 1]))