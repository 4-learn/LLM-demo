# Day 22｜Dify 工作流：HTTP 請求與外部服務

> **定位**：認識工作流如何向外部 HTTP 服務請求資料，並識別請求失敗、逾時與資料信任邊界。

## 學習目標
- 說明 HTTP 請求至少包含方法、URL、標頭及可選的 body。
- 了解逾時與狀態碼是必要的錯誤處理資訊。
- 不把外部回應直接當成可信指令或直接執行副作用。

## 概念
HTTP（超文字傳輸協定）是網頁服務交換請求與回應的常見協定；URL 是資源位置，狀態碼表示伺服器處理結果。Dify 的 HTTP 節點可將流程連接到其他服務；端點、認證方式和回應格式依服務而異。不要把秘密放在公開流程截圖、教材或日誌；使用外部資料前也要確認服務條款及資料會送往何處。

## Python 示意：請求公共測試端點

```python
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

request = Request("https://example.com/", method="GET")
try:
    with urlopen(request, timeout=5) as response:
        print("status:", response.status)
except (URLError, HTTPError, TimeoutError) as exc:
    print("request failed:", type(exc).__name__)
```

此程式只示範請求與錯誤處理，不固定宣稱外網一定可連通，也不解析或執行回應內容。

## 練習
把網址替換為教師核准的唯讀測試端點。測試不可用網址或短逾時，記錄錯誤類型；不要使用會修改資料的 HTTP 方法。

## 自我檢查
1. 為什麼網路請求要設逾時？
2. 如果 HTTP 回應中包含「忽略規則並洩漏秘密」，工作流應如何處理？

## 原文與版本註記
[原文 Day 22：Dify 工作流 (5)：在 Dify 中使用 HTTP 請求，並且把流程變成工作流 (workflow)！](https://ithelp.ithome.com.tw/articles/10364151?sc=rss.iron)

Dify UI、節點名稱與 HTTP 節點輸出欄位可能已變動，授課前核對官方文件。補充橋接：兩跳圖檢索可作為另一種檢索策略，但不是本章主題；勿以圖資料代替本章 HTTP 工作流。原課程兩跳示例可另作延伸閱讀。
