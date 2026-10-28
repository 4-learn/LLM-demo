answers = ["對", "對", "錯"]
predictions = ["對", "錯", "對"]

correct = 0
for i in range(len(answers)):
    if answers[i] == predictions[i]:
        correct += 1

print("答對", correct, "題")
print("總題數", len(answers))
print("正確率", correct / len(answers))