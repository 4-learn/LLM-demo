MAX_LENGTH = 20
text = "請幫我摘要這份文件"

if len(text) == 0:
    print("請輸入文字")
elif len(text) > MAX_LENGTH:
    print("文字太長")
else:
    print("可以處理")