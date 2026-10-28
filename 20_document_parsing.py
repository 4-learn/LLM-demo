lines = ["維修紀錄", "工單編號：WO-2048", "問題：燈不亮"]
number = ""

for line in lines:
    if line[:5] == "工單編號：":
        number = line[5:]

if number != "":
    print("找到：", number)
else:
    print("找不到工單編號")