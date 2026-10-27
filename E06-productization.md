# Docker、Gradio/Streamlit 與產品化

> **正文改版：2026-10-24｜對應新版 LLM 選修 E06：Docker、Gradio/Streamlit 與產品化。**
> 本節用離線示範設計 Web app 服務邊界，涵蓋 Docker、Gradio、Streamlit 的部署概念、health/readiness、輸入限制與秘密管理；本節不是容器或雲端服務的實際部署驗證。

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先修第 09 節本機推論、第 15 節成本與第 24 節服務契約；不熟容器者可先讀第一段概念表。

| 時間 | 內容 |
|---|---|
| 0–10 分鐘 | Docker 映像、容器與平台選擇 |
| 10–20 分鐘 | health 與 readiness 的不同責任 |
| 20–35 分鐘 | 執行離線輸入閘門 Demo |
| 35–50 分鐘 | Workshop：限制、秘密與部署檢核 |

完成後，學生能說明一個簡單服務如何打包及啟動，能區分程序健康與可接流量狀態，並用可測試規則拒絕超長或非文字輸入，且不把密鑰寫進程式碼或日誌。

## 一、部署包裝與介面選擇

Docker image 是應用程式與執行環境的可版本化包裝；container 是該映像的執行個體。映像不會自動解決模型下載、磁碟空間、GPU 驅動、持久化、網路政策或金鑰注入。Gradio 適合快速建立互動式模型介面，Streamlit 適合資料應用與儀表板；兩者都可部署為服務，但公開分享連結不等於生產級身份驗證、限流或資料保護。正式環境應固定依賴版本、掃描映像、配置健康探針並規劃回滾。

## 二、health 不等於 readiness

Health 通常回答程序是否活著，適合判斷是否要重新啟動；readiness 回答目前是否準備好接流量，例如模型載入完成、必要依賴就緒、容量未滿。程序啟動成功但模型仍載入時，health 可以成功、readiness 仍失敗。不要讓健康檢查每次請求都載入模型，也不要把內部錯誤、環境變數或憑證直接回傳給匿名呼叫者。Demo 以純函式展示兩個狀態，不啟動 Web server。

## 三、輸入限制與安全日誌

輸入閘門應檢查型別、長度、允許格式及必要的資源上限，再呼叫模型；限制過寬會造成記憶體耗盡或成本暴增，限制過窄則會拒絕正常使用。日誌只記請求 ID、長度、耗時與錯誤類別等必要資訊，避免記錄原文、個資、Authorization header 或密鑰。密鑰應由部署環境的 secret store 或受限檔案注入，權限最小化、定期輪替；`.env` 只供本機開發，必須排除版控，不可烘焙進映像。

## 四、可執行 Demo

程式僅使用標準函式庫，模擬 readiness、輸入限制及最小化日誌欄位，不讀取真實秘密、不啟動 Docker、Gradio 或 Streamlit，也不連外。正式服務還需在框架路由、反向代理和部署平台分別設定 body limit、逾時、認證及資源配額。

<!-- demo: E06_productization.py -->
```python
"""離線示範產品服務基本閘門；不啟動 server、不讀取 secret。"""
import re

MAX_CHARS = 80


def health():
    return {"status": "ok"}


def readiness(model_loaded, accepting_requests=True):
    if type(model_loaded) is not bool or type(accepting_requests) is not bool:
        raise ValueError("readiness 狀態必須是 bool")
    return {"ready": model_loaded and accepting_requests}


def validate_input(value):
    if not isinstance(value, str):
        return False, "必須是文字"
    if not value.strip():
        return False, "不可為空"
    if len(value) > MAX_CHARS:
        return False, f"超過 {MAX_CHARS} 字元上限"
    return True, "接受"


def safe_log(request_id, value, secret):
    """只回傳請求識別碼與長度；不記原文，也不洩漏秘密是否命中。"""
    if (not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request_id)
            or not isinstance(value, str)):
        raise ValueError("request_id 必須是 1–64 位英數、底線或連字號，value 必須是文字")
    if not isinstance(secret, str) or not secret:
        raise ValueError("secret 必須是非空文字")
    return {"request_id": request_id, "chars": len(value), "logged_secret": False}


def main():
    print(f"health={health()}")
    print(f"readiness(model_loaded=False)={readiness(False)}")
    print(f"readiness(model_loaded=True)={readiness(True)}")
    for item in ("請摘要這份說明", "", "x" * 81, 42):
        print(f"輸入長度={len(item) if isinstance(item, str) else '非文字'}：{validate_input(item)}")
    print(f"安全日誌={safe_log('req-7', '一般內容', 'DEMO_SECRET')}")


if __name__ == "__main__":
    main()
```

執行：`python3 E06_productization.py`。關鍵輸出（本次本機實測）：

```text
health={'status': 'ok'}
readiness(model_loaded=False)={'ready': False}
readiness(model_loaded=True)={'ready': True}
輸入長度=7：(True, '接受')
輸入長度=0：(False, '不可為空')
輸入長度=81：(False, '超過 80 字元上限')
輸入長度=非文字：(False, '必須是文字')
安全日誌={'request_id': 'req-7', 'chars': 4, 'logged_secret': False}
```

## Workshop：交付部署檢核表

**任務（15 分鐘）**：3 分鐘執行 Demo；4 分鐘把上限改為 20，測試 20 與 21 字元邊界；4 分鐘設計 `health` 與 `readiness` 在「程序正常但模型未載入」的回應；4 分鐘寫出 Docker 映像不應包含的三種秘密及部署時應採的注入方式。

**參考判讀**：20 字元應接受，21 字元應拒絕；程序可存活但模型未載入時，health 可為 ok、readiness 必須 false。不可把 API key 放進 Dockerfile `ENV`、映像層或 log；用平台 secret store／受限環境注入，最小權限並可輪替。readiness 還要避免因暫時容量滿載而誤判需重啟。

**驗收**：提交邊界測試輸出、兩個健康狀態的預期回應表，以及含 Docker、Gradio/Streamlit 選擇、輸入上限、secret 注入與輪替、日誌遮罩、回滾的部署檢核表。教師可逐項核對，不接受只寫「部署成功」或「注意安全」。

## 教師紀錄與下一步

- 證據級別：**本機實測**（2026-10-24；Linux Ubuntu 20.04 x86_64，Python 3.11.2；標準函式庫；模型 revision／套件版本不適用；`python3 E06_productization.py`）。所有輸出由離線範例產生，不代表 Docker 或 Web framework 的實測。
- 限制：未 build/run Docker image，未啟動 Gradio、Streamlit、反向代理或雲端服務；未量測延遲、容量、模型啟動時間、探針與 secret store 行為。示範 `safe_log` 只展示欄位最小化，不是完整日誌系統或側信道安全審計；request ID 亦須由上游產生且不得含敏感資訊。
- 尚未實班試教；50 分鐘安排尚未真人驗證。教師紀錄僅為本機程式輸出，未經遠端或學生驗證。
- 下一步：參照第 24 節契約、第 15 節成本閘門及第 09 節本機推論，將輸入限制、逾時、排隊、readiness 和密鑰輪替加入真實部署的 staging 驗收；先做匿名化測試資料與回滾演練。
- 舊教材連結：無；本篇為新增選修，未宣稱其他教材已同步。
