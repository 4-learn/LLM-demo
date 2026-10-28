# Day 23｜Dify 工作流：變量聚合與分支輸出

> **定位**：理解多條流程分支如何把各自結果匯整成一致的輸出；用純 Python 模擬欄位聚合與缺值處理。

## 學習目標
- 說明變量聚合器解決的流程整合問題。
- 將多個分支結果整理為一致的資料結構。
- 對缺欄位採取明確處理，而非默默輸出錯誤結果。

## 概念
當不同分支可能產生不同變數，後續節點需要一個穩定輸入。變量聚合是把多條分支的輸出整理到共同欄位；fallback（後備值或路徑）用於沒有有效輸出時的處理。聚合應明確定義欄位，不應把空值或錯誤誤認為有效答案。

## 實作：合併分支結果

```python
def aggregate(route_a=None, route_b=None):
    candidates = [x for x in (route_a, route_b) if isinstance(x, str) and x.strip()]
    if not candidates:
        return {"status": "no_result", "answer": None}
    return {"status": "ok", "answer": candidates[0]}

print(aggregate(route_a="由分支 A 找到的摘要"))
print(aggregate())
```

預期第一筆狀態為 `ok`，第二筆為 `no_result`。若分支有衝突，應明訂優先順序或交由人工檢查，而非任意挑選。

## 練習
讓兩個分支同時回傳不同內容，設計一個「衝突」狀態；再測試其中一個分支回傳空字串。

## 自我檢查
1. 變量聚合如何讓後續節點使用穩定欄位？
2. 多個來源彼此矛盾時，為什麼不應靜默覆蓋？

## 原文與版本註記
[原文 Day 23：Dify 工作流 (6)：用變量聚合器來簡化流程！](https://ithelp.ithome.com.tw/articles/10364634?sc=rss.iron)

Dify 節點行為與變量選取畫面可能已更新，授課前依官方文件核對；本例為概念模型，不是特定版本的 Dify API。
