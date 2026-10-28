# E06｜先檢查使用者輸入

> 本課只學一件事：在處理文字前先檢查它是不是空白或太長。

[回課程書目](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 為什麼要限制輸入

如果服務收到很長的文字，處理時間和成本可能增加。先設一個簡單長度上限，可以拒絕太長的輸入。真實服務還需要更多檢查，本課只練習這一個規則。

## Demo：檢查文字長度

<!-- demo: E06_productization.py -->
```python
MAX_LENGTH = 20
text = "請幫我摘要這份文件"

if len(text) == 0:
    print("請輸入文字")
elif len(text) > MAX_LENGTH:
    print("文字太長")
else:
    print("可以處理")
```

## 動手試試

把 `MAX_LENGTH` 改成 `5`，再執行。也可以把 `text` 改成空字串，觀察結果。

**延伸名詞**：Docker、Gradio、Streamlit 是部署與介面工具，不是本課先備要求；本 Demo 沒有啟動服務。
