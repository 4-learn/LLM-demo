# 25｜用關鍵字找文件

> 本課只學一件事：用一個查詢，在小型文件清單中找出含有關鍵字的文件。

[回課程書目](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 找到相關文件

如果要找電池資料，可以先搜尋「電池」。這種方法簡單直接，但如果文件只寫「續航」，就可能找不到。搜尋結果是候選資料，不一定就是答案。

## Demo：關鍵字搜尋

<!-- demo: 25_hybrid_retrieval.py -->
```python
documents = {
    "A": "電池容量 5000 mAh",
    "B": "電池續航測試",
    "C": "充電器輸出功率",
}
query = "電池"

for name in documents:
    if query in documents[name]:
        print(name, documents[name])
```

## 動手試試

把 `query` 改成「充電器」，再執行；再試「續航」。比較哪些文件出現、哪些沒有。

**延伸名詞**：向量搜尋可找意思相近的文字；混合檢索會合併多種搜尋方式。本課不實作這些方法。
