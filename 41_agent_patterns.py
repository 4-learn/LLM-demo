weather = "未知"
step_limit = 3

for step in range(step_limit):
    if weather == "未知":
        next_step = "查詢天氣"
    else:
        next_step = "整理並回答"
    print("第", step + 1, "步：模擬模型選擇：", next_step)
    if next_step == "查詢天氣":
        weather = "晴天"
    else:
        break
