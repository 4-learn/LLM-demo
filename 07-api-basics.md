# 07｜用 API 請模型回答

> **定位**：用可檢查的 HTTP API 請求／回應格式，理解程式如何準備訊息並讀取模型服務的答覆；本節以離線 fixture 重播，不實際建立網路連線，也不需要帳號或金鑰。真實呼叫僅作概念延伸，不是本節的必做前提。

## 學習目標

完成本節後，你應該能夠：
- 說明 API request/response、HTTPS、JSON、模型名稱與 API key 各代表什麼
- 從一筆合成事件建立 chat completion 請求，辨認輸入、輸出、usage 與錯誤
- 在不洩漏憑證、不送真實個資、不把模型當成法規或工安判斷者的前提下，安全地描述真實 API 呼叫的條件

[回 51 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 範例程式碼

- [demo](https://github.com/4-learn/llm-demo)：`E02_api_basics.py`
- 必做離線 fixture；真實 NVIDIA Build API 呼叫為選做，需自行取得 API key 並連網。
- 本節不代申請或收集金鑰；真實呼叫只送合成事件，可能受額度、模型可用性與服務政策限制。
- NVIDIA 官方參考：[LLM APIs](https://docs.api.nvidia.com/nim/reference/llm-apis)、[Llama 3.1 8B API](https://docs.api.nvidia.com/nim/reference/meta-llama-3_1-8b-infer)；執行前請確認模型仍可用且帳戶有權限。

---

## 情境

上一節你依授權、記憶體和上下文挑出候選模型。現在主管問：「如果模型不在工地電腦，而是在服務商那邊，我的程式怎麼請它整理一筆事件？」

本節先用合成事件走過一次完整的請求與回應。你會看到 API 是程式和服務之間的介面；它不是模型本身，也不保證回答正確。

---

本節提供兩種練習路徑：必做的離線 fixture，以及選做的 NVIDIA Build 真實 API 呼叫。真實呼叫前先由教師確認外部服務使用符合課程安排；只送下方合成資料，不放入真實工地資料、個資或機密。NVIDIA 的模型清單、端點、額度與資料政策會變，請依當時官方文件核對，不保證免費或一定可用。NVIDIA 官方參考：[LLM APIs](https://docs.api.nvidia.com/nim/reference/llm-apis)、[Llama 3.1 8B API](https://docs.api.nvidia.com/nim/reference/meta-llama-3_1-8b-infer)。

## 一、API 呼叫是一次網路請求

API（Application Programming Interface）是程式依照約定向服務提出請求、再讀回應的介面。常見的文字模型 API 使用 HTTPS 傳送 JSON：

```text
你的程式 -- HTTPS + JSON request --> 服務商的 API
你的程式 <-- HTTP status + JSON response -- 服務商的 API
```

一次 chat completion 請求通常會包含：

| 欄位 | 用途 | 注意事項 |
| --- | --- | --- |
| `model` | 指定服務商提供的模型 ID | 名稱與可用性依帳號、端點和日期而變 |
| `messages` | system/user 等對話內容 | 內容會離開本機送到服務商；不要放真實工地資料或個資 |
| `Authorization` | 以 API key 證明呼叫者身分 | 類似可計費的密碼；不可貼在講義、對話、截圖或程式碼 |
| `base_url` | API 服務的網路位置 | 不同供應商端點可能不同；只使用教師核准的 HTTPS 端點 |

「OpenAI 相容」通常表示某些請求格式相似，**不代表**所有欄位、模型、用量計算、資料保存政策或價格都相同。收到 HTTP 200 也只代表請求得到成功回應，不表示模型內容正確。

### 金鑰安全規則

- 本課預設離線 fixture；**不要為了完成課堂作業申請或貼出自己的 API key**。
- 若教師安排真實呼叫，只在自己受控的本機終端機設定環境變數，絕不將 key 寫進程式、命令列參數、筆記本、Git、截圖或聊天訊息。
- 只送講義提供的合成資料；真實呼叫會把訊息傳到外部服務，可能產生費用，也受供應商資料政策與限額影響。
- 公用電腦課後清除憑證；若曾意外外洩，立即撤銷並更換。

---

## 二、先看請求和回應長什麼樣

下方是教學用結構示意，不是實際呼叫：

```json
{
  "model": "provider-specific-model-id",
  "messages": [
    {"role": "system", "content": "請只根據提供的合成事件摘要回答。"},
    {"role": "user", "content": "事件：疑似未戴安全帽。請用一句話列出待人工確認事項。"}
  ],
  "max_tokens": 80
}
```

伺服器回應可能包括 `choices` 中的文字、`finish_reason` 和 `usage`。不同供應商欄位可能不同；不能假設每個服務都回傳同一份資料。

- `finish_reason` 表示服務回報的停止原因；其值依供應商而異。
- `usage` 是服務回報的用量資訊；可能缺少、估算方式不同，不能直接等同帳單。
- `logprobs` 或 `top_logprobs` 若有支援，通常也只回傳部分候選資訊；不能宣稱那就是完整詞彙表分布。
- 模型文字可能錯誤、拒答、空白或不符合格式；JSON 可解析不代表內容為真。

---

## 三、離線 Demo：同一個介面，兩種傳輸

本節 Demo 預設使用固定 fixture，明確標記不是本機模型推論、不是遠端 API 實測。程式將請求整理和回應解析分開，讓你先練習資料流而不必註冊服務；另外提供明確啟動、需自行設定金鑰的選做 NVIDIA Build 真實傳輸路徑。

<!-- demo: E02_api_basics.py -->
```python
"""第 06 節：API request/response 的入門示範。

預設僅重播固定 fixture，無網路、無金鑰、無第三方套件。
只有明確加上 --live 並設定 NVIDIA_API_KEY 才會送出真實請求；不構成工安建議。
"""
import json
import os
import sys
# openai SDK is imported lazily for the explicit --live path only.

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"

SYNTHETIC_MESSAGES = [
    {"role": "system", "content": "只根據合成事件內容，列出待人工確認事項。"},
    {"role": "user", "content": "合成事件：疑似一人未戴安全帽，正在進入施工區。"},
]
FIXTURE = {
    "model": "fixture-model-not-a-real-provider",
    "choices": [{"message": {"role": "assistant", "content": "請由現場人員確認影像與個人防護具狀況。"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 42, "completion_tokens": 14, "total_tokens": 56},
}


def make_request(messages=SYNTHETIC_MESSAGES, model="fixture-model-not-a-real-provider"):
    """建立可序列化的請求內容；不含憑證。"""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 必須是非空清單")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model 必須是非空字串")
    return {"model": model, "messages": messages, "max_tokens": 80}


def call_nvidia(request, api_key):
    """選做真實傳輸；只接受由呼叫端提供的金鑰，不列印或保存。"""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("選做 live 需安裝 openai SDK；fixture 不需要") from None
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key, timeout=45, max_retries=0)
    try:
        result = client.chat.completions.create(
            model=request["model"],
            messages=request["messages"],
            max_tokens=request["max_tokens"],
        )
    except Exception as exc:
        raise RuntimeError(f"API 呼叫失敗（{type(exc).__name__}）；不自動重試") from None
    if not result.choices:
        raise RuntimeError("response 缺少 choices")
    choice = result.choices[0]
    message = choice.message
    if message is None or not isinstance(message.content, str):
        raise RuntimeError("response 缺少文字 content")
    return {
        "choices": [{"message": {"role": "assistant", "content": message.content},
                     "finish_reason": choice.finish_reason}],
        "usage": result.usage.model_dump() if result.usage else None,
    }


def parse_response(response):
    """安全讀取最小欄位；缺欄位或型別錯誤時明確失敗。"""
    if not isinstance(response, dict):
        raise ValueError("response 必須是物件")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("response 缺少 choices")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("response 缺少文字 content")
    return {
        "text": message["content"],
        "finish_reason": choices[0].get("finish_reason"),
        "usage": response.get("usage"),
    }


def main():
    live = "--live" in sys.argv
    request = make_request(model=NVIDIA_MODEL if live else "fixture-model-not-a-real-provider")
    if live:
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise SystemExit("選做真實呼叫需在本機設定 NVIDIA_API_KEY；不要將 key 寫入程式或命令列")
        print("模式：NVIDIA Build 真實 API 呼叫；僅送講義合成事件")
        response = call_nvidia(request, api_key)
    else:
        print("模式：離線 fixture（不是模型或 API 實測）")
        print("Request JSON：")
        print(json.dumps(request, ensure_ascii=False, indent=2))
        response = FIXTURE
    result = parse_response(response)
    print("\n回答：", result["text"])
    print("停止原因：", result["finish_reason"])
    print("服務回報用量（若有提供；不等於帳單）：", result["usage"])


if __name__ == "__main__":
    main()
```

執行：

```bash
python3 E02_api_basics.py
```

預設模式不會連網。只有看到明確標示的 fixture 回應，不能把它報告成「我呼叫了模型」。

---

## 選做：NVIDIA Build 真實呼叫

請先確認教師允許使用外部服務，再依 NVIDIA 官方網站流程註冊並建立 API key。服務是否開放、模型名稱、端點、免費額度與資料政策會變，請上課當天查官方文件；不保證免費或每個帳號都有相同權限。這次呼叫會把合成訊息送至外部服務。

在 Linux/macOS 終端機輸入以下指令，互動式輸入 key（不回顯、不放進命令列參數），再執行：

```bash
read -s -p "NVIDIA API key: " NVIDIA_API_KEY; echo
export NVIDIA_API_KEY
python3 E02_api_basics.py --live
unset NVIDIA_API_KEY
```

請勿在分享終端機、截圖或作業中顯示 key。PowerShell 請用作業系統的環境變數 UI 或安全憑證管理方式設定 `$env:NVIDIA_API_KEY`；不要將 key 寫在命令文字或貼進聊天。程式不會列印 key；課後清除環境變數。若 key 曾外洩，立即從 NVIDIA 帳戶撤銷並重新建立。

若遇到 401/403、429、模型不存在或逾時，停止並查閱官方文件，不要自動重試，也不要貼出包含 Authorization header 的原始錯誤。作業只交遮蔽過的結果和狀況描述，不交憑證或完整 headers。

## 四、失敗也要看得懂

真實服務可能回傳 HTTP 錯誤、逾時或無法解析的內容。常見狀況包括：

| 狀況 | 可能原因 | 安全處理 |
| --- | --- | --- |
| 401 / 403 | 憑證錯誤、無權限或模型未開通 | 不要把 key 貼給同學；由持有人在本機檢查設定 |
| 429 | 超出速率或額度限制 | 停止呼叫，先查限制；不要自動重試造成更多請求 |
| 5xx / timeout | 服務暫時失敗或網路逾時 | 先確認是否已處理；逾時時服務可能已收到並計費 |
| 回應格式不符 | 服務商差異、錯誤頁或模型輸出 | 先驗證結構；不可假設一定有 `choices[0].message.content` |

除非有明確的重試策略和費用保護，本課練習採取**失敗即停、不自動重試**。錯誤訊息不要輸出 Authorization header、API key 或整份含敏感資料的請求。

---

## Workshop

### 題目（15 分鐘）

1. 執行離線 Demo，圈出 request 中的 `model`、`messages`、`max_tokens`，說明各欄位的作用。
2. 讀取 fixture response，找出回答文字、停止原因和 usage；在作業中明確標示它是 fixture。
3. 將 `FIXTURE` 改成缺少 `choices` 的物件，再執行測試，觀察程式如何拒絕不完整回應。
4. 選做：經教師同意後，依 NVIDIA 官方流程在本機設定 key，執行 `python3 E02_api_basics.py --live`；只送合成資料，繳交時不得包含 key 或 headers。
5. 寫下呼叫前必須確認的三件事：金鑰保護、資料是否可外傳、額度/費用與模型可用性。

### 驗收

- `python3 E02_api_basics.py` 可執行且標明離線 fixture；預設不連網。
- `--live` 未設定 `NVIDIA_API_KEY` 時安全拒絕，不發出網路請求；設定後才使用 NVIDIA Build HTTPS endpoint。
- `parse_response({})` 會以 `ValueError` 拒絕缺少的回應欄位。
- 作業不包含任何 API key，也不聲稱 fixture 是真實模型輸出。
- 能說明 API 回答不可代替現場人員、專業人士或主管的正式判斷。

---

## NVIDIA Build 官方資訊（上課前核對）

- [NVIDIA NIM LLM API 參考](https://docs.api.nvidia.com/nim/reference/llm-apis)
- [Meta Llama 3.1 8B Instruct API 參考](https://docs.api.nvidia.com/nim/reference/meta-llama-3_1-8b-infer)
- [NVIDIA Build 模型頁](https://build.nvidia.com/meta/llama-3.1-8b-instruct)

本教材示例使用 OpenAI Python SDK 的 `client.chat.completions.create`，設定 `base_url="https://integrate.api.nvidia.com/v1"` 和模型 ID `meta/llama-3.1-8b-instruct`。這些資訊以編寫時的官方文件為準；服務可能下架、限制帳號或更新端點，教師授課前須實際確認可用性與費用政策。未確認前，離線 fixture 仍是完整可執行的必做路徑。

---

## 教師紀錄與下一步

證據等級：**fixtures**。

- 本 Demo 僅使用 Python 標準函式庫與固定 fixture；未呼叫任何供應商 API，未測量費用、延遲或模型品質。
- 真實模型名稱、端點、價格、免費額度、支援欄位與資料保存政策會變；若未在上課當天重新核對，不得寫成保證。
- 本節不要求學生申請憑證，不示範把真實工地內容送往外部服務。
- 下一節比較本機推論：本節模型在服務商端執行；下一節模型權重載入本機。兩者都不能取代安全驗證與人工責任。
