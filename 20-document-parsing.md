# 20｜從文字找出工單編號

> 本課只學一件事：根據明確標籤，從文字中找到欄位值；找不到就說找不到。

[回課程書目](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 看一張工單

文字：「工單編號：WO-2048。問題：燈不亮。」我們可以找到工單編號 `WO-2048`。如果文字沒有「工單編號」這個標籤，就不能猜一個出來。

## Demo：逐行尋找標籤

<!-- demo: 20_document_parsing.py -->
```python
lines = ["維修紀錄", "工單編號：WO-2048", "問題：燈不亮"]
number = None

for line in lines:
    if line.startswith("工單編號："):
        number = line.replace("工單編號：", "")

if number:
    print("找到：", number)
else:
    print("找不到工單編號")
```

## 動手試試

把 `lines` 裡的工單編號改成 `WO-3001`，再把標籤整行刪掉各執行一次。結果應分別顯示新編號與「找不到」。

**延伸內容**：PDF 文字抽取、掃描件 OCR、表格與多頁文件是後續課題，本課沒有實作它們。
