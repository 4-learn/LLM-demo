scores = [1, 2]
values = [10, 20]
weights = [score / sum(scores) for score in scores]
result = sum(w * value for w, value in zip(weights, values))
print(weights, result)
