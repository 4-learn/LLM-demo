task = "用一句話摘要"
data = "今天下雨，公園活動改到室內。"
rule = "只根據資料回答。"
prompt = task + "\n" + rule + "\n資料：" + data
print(prompt)
