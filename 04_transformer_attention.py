def average(weights, values):
    total = 0
    for i in range(len(weights)):
        total += weights[i] * values[i]
    return total

weights = [0.25, 0.75]
values = [10, 20]
print(average(weights, values))