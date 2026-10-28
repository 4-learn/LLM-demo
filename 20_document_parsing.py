lines = ["維修紀錄", "工單編號：WO-2048", "問題：燈不亮"]
number = None

for line in lines:
    if line.startswith("工單編號："):
        number = line.replace("工單編號：", "")

if number:
    print("找到：", number)
else:
    print("找不到工單編號")