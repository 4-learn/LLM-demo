scores = [1, 2]
values = [10, 20]
total = scores[0] + scores[1]
first_weight = scores[0] / total
second_weight = scores[1] / total
result = first_weight * values[0] + second_weight * values[1]
print("權重：", first_weight, second_weight)
print("結果：", result)
