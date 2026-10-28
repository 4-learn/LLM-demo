# Day 16｜自訂工具與 FastAPI：明確定義輸入和回應

> **主題：** Agent 的自訂工具需要一份清楚的參數契約；API 端仍須驗證輸入、回傳可預期的結果，並限制副作用。

Agent 是能依觀察選擇下一步的受限流程。API（應用程式介面）讓程式依規格交換請求與回應；FastAPI 是 Python 網頁 API 框架。本章只用 Python 函式示範契約，不啟動 Web 服務。Schema（資料結構規格）會描述工具參數的欄位、型別及限制。

## 動手做：定義一個安全的純函式工具

```python
def lookup_order(order_id, orders):
    if not isinstance(order_id, str) or not order_id.startswith("ORD-"):
        return {"ok": False, "error": "invalid_order_id"}
    if order_id not in orders:
        return {"ok": False, "error": "not_found"}
    return {"ok": True, "status": orders[order_id]}

orders = {"ORD-1001": "配送中"}
print(lookup_order("ORD-1001", orders))
print(lookup_order("bad", orders))
print(lookup_order("ORD-9999", orders))
```

預期輸出：

```text
{'ok': True, 'status': '配送中'}
{'ok': False, 'error': 'invalid_order_id'}
{'ok': False, 'error': 'not_found'}
```

這個本地函式演示工具契約，不是可部署的 FastAPI server。實際服務還要加認證、授權、輸入長度限制、錯誤處理與日誌遮罩。若工具會改資料或觸發付款等副作用，不能只因模型要求就執行，必須由應用程式核對使用者權限並取得必要確認。

## 小練習

加入 `ORD-1002`，並測試空字串、錯誤格式及不存在的訂單。分別說明格式驗證和使用者授權是什麼不同問題。

## 原文與版本

原文：[Day 16｜Dify 入門 (2)：讓 Agent 使用自定義工具吧！(內含一咪咪 FastAPI)](https://ithelp.ithome.com.tw/articles/10360896?sc=rss.iron)

FastAPI / Dify 的安裝方式、API schema 與介面可能更新。本例不依賴第三方套件；若延伸實作，核對當前官方文件，勿將憑證寫入程式碼或公開日誌。
