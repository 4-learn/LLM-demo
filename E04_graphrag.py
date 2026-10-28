responsible_for = {"林員": "北區泵浦"}
located_in = {"北區泵浦": "北區"}
person = "林員"

if person in responsible_for:
    equipment = responsible_for[person]
    if equipment in located_in:
        print("有資料支持：", located_in[equipment])
    else:
        print("資料不足：找不到設備位置")
else:
    print("資料不足：找不到負責設備")
