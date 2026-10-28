# Day 21｜Dify 工作流：模板轉換與資料物件

> **定位**：理解工作流中的結構化資料轉換，使用 Python 標準函式庫模擬將資料物件轉為使用者可讀文字；Dify 介面步驟須依授課時版本核對。

## 學習目標
- 區分結構化物件與呈現文字。
- 從輸入資料取出欄位並產生安全、可預期的摘要。
- 說明模板不應被當作資料驗證或權限檢查。

## 概念
工作流常接收 JSON（鍵值格式的文字資料）欄位，再以模板或程式碼節點轉成下一步所需格式。物件（object）是由欄位和值組成的資料結構；轉換前應檢查欄位存在與型別，不要假設模型或外部節點一定回傳正確資料。Dify 介面行為需依當前版本確認。

## 實作：檢查資料再格式化

```python
record = {"name": "林同學", "status": "待確認", "amount": 180}

def render_summary(item):
    if not isinstance(item.get("name"), str):
        raise ValueError("name 必須是文字")
    if not isinstance(item.get("amount"), int):
        raise ValueError("amount 必須是整數")
    return f"{item['name']}：{item['status']}，金額 {item['amount']} 元"

print(render_summary(record))
```

預期輸出：`林同學：待確認，金額 180 元`。

## 練習
移除 `amount` 或改成文字，觀察驗證如何阻止錯誤資料進入後續流程。再思考若使用者提供 `name`，如何避免將其直接當 HTML 或指令解讀。

## 自我檢查
1. 模板能否保證輸入資料正確？
2. 為何格式轉換和權限判斷應分開？

## 原文與版本註記
[原文 Day 21：Dify 工作流 (4)：用模板轉換和代碼執行來操作 object](https://ithelp.ithome.com.tw/articles/10364027?sc=rss.iron)

原文操作反映 2024 年 Dify 介面與節點版本。介面、節點能力及語法依授課時官方文件核對；本例為標準 Python 示意，不宣稱 Dify 當前 UI 相同。
