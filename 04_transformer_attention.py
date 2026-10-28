scores = [1, 2]
values = [10, 20]
total = scores[0] + scores[1]
weights = [scores[0] / total, scores[1] / total]
result = weights[0] * values[0] + weights[1] * values[1]
print(weights, result)
