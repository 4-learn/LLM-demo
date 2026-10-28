documents = {
    "A": "電池容量 5000 mAh",
    "B": "電池續航測試",
    "C": "充電器輸出功率",
}
query = "電池"
for name in documents:
    if query in documents[name]:
        print(name, documents[name])
