# 語音 ASR／TTS 與語音代理

> **正文改版：2026-10-24｜對應新版 LLM 選修 E05：語音 ASR／TTS 與語音代理。**
> 本節以文字轉錄重現 ASR→代理決策→TTS 的安全邊界，並在執行不可逆操作前停下；本節不是麥克風、語音辨識器或合成語音的品質測試。

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先修第 10 節 unknown、第 13 節結構化輸出及第 33 節人工確認；可獨立修讀，示範以合成轉錄替代音訊。

| 時間 | 內容 |
|---|---|
| 0–10 分鐘 | ASR、代理理解、TTS 的資料流及錯誤傳遞 |
| 10–20 分鐘 | 轉錄不確定性、重問與確認 |
| 20–35 分鐘 | 執行離線 Demo，追蹤工具副作用 |
| 35–50 分鐘 | Workshop：危險指令的拒絕與確認設計 |

完成後，學生能區分「辨識文字」與「使用者意圖」，能把低信心或含糊轉錄導向重問，並證明高風險工具在確認前零副作用。

## 一、語音管線各段都會出錯

ASR 把聲音轉成文字，代理依文字與上下文決定回覆或呼叫工具，TTS 再把文字讀出。錯誤可在每一段放大：噪音把「不要關閉」辨成「關閉」，代理若把轉錄當成確定命令，工具可能立刻切斷設備。TTS 說得流暢不表示辨識正確，也不等於使用者已聽懂或授權。

## 二、轉錄文字要保留不確定性

實務上要保留候選詞、時間戳、語言、信心分數與原始音訊政策，而不是只存一條「乾淨」文字。信心分數是辨識器的估計，不是機率保證；不同口音或環境下可能失準。低分、否定詞、數量、姓名、設備代號等關鍵槽位不清楚時，代理應重問並用文字複述請求，不能自行補字。日誌也要限制存取與保存期限，音訊可能含個資。

## 三、不可逆副作用前設人工閘門

查詢天氣或念出答案可重試；開鎖、付款、刪除、關閉安全設備則可能不可逆。把工具按風險分類，讓低風險讀取可自動執行，高風險操作必須先顯示清楚的操作摘要，再由可信介面確認；含糊或低信心輸入直接停止。正式系統須將確認綁定使用者、請求與操作，具期限且以原子方式消耗，並持久化冪等狀態。以下 Demo 的 tuple 只是可任意偽造的教學輸入，記憶體集合也只模擬單一程序重播；兩者都不是安全控制或真正的一次性確認。

## 四、可執行 Demo

本程式只用標準函式庫，輸入是教師準備的合成文字轉錄，不錄音、不連線、不呼叫 ASR/TTS。它模擬低信心時（無論讀取或操作意圖）先重問、高信心讀取、信心不足的危險操作、以及確認後的危險操作。需注意：這個簡化例子不實作語者驗證或正式身分授權。低信心閘門在意圖分流前執行。

<!-- demo: E05_voice_agents.py -->
```python
"""離線語音代理安全示範：文字轉錄、信心閘門、確認與副作用計數。"""


import math


def handle(transcript, confidence, confirmed, request_id, confirmation, effects, completed):
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("轉錄不可為空")
    if (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
            or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError("confidence 必須是有限數且介於 0 與 1")
    if type(confirmed) is not bool:
        raise ValueError("confirmed 必須是 bool")
    if (not isinstance(request_id, str) or not request_id.strip()
            or len(request_id) > 128 or any(ord(ch) < 32 or ord(ch) == 127 for ch in request_id)):
        raise ValueError("request_id 必須為 1–128 字元且不含控制字元")
    if not isinstance(effects, list) or not isinstance(completed, set):
        raise ValueError("effects/completed 必須是本機示範用 list/set")
    if confidence < 0.85:
        return "請重說：辨識信心不足，未執行任何操作。"
    if confirmation is not None and confirmation != (request_id, "關閉泵浦"):
        return "確認無效：request_id 或操作不符；未執行。"
    if request_id in completed:
        return "已處理：重播請求不重複執行。"
    if transcript == "查詢泵浦狀態":
        return "TTS 模擬：泵浦狀態正常（唯讀）"
    if transcript in ("關閉泵浦", "不要關閉泵浦"):
        if transcript == "不要關閉泵浦":
            return "已理解為禁止關閉；未執行操作。"
        if not confirmed or confirmation != (request_id, "關閉泵浦"):
            return "待確認：請取得綁定本 request_id 與操作的一次性確認；尚未執行。"
        completed.add(request_id)
        effects.append("關閉泵浦")
        return "已執行一次：關閉泵浦。"
    return "不支援的請求；未執行任何操作。"


def main():
    effects, completed = [], set()
    request_id = "req-close-1"
    cases = [
        ("查詢泵浦狀態", 0.40, False, "req-read-low", None),
        ("查詢泵浦狀態", 0.99, False, "req-read", None),
        ("關閉泵浦", 0.62, True, request_id, (request_id, "關閉泵浦")),
        ("關閉泵浦", 0.97, False, request_id, None),
        ("關閉泵浦", 0.97, True, request_id, (request_id, "關閉泵浦")),
        ("關閉泵浦", 0.97, True, request_id, (request_id, "關閉泵浦")),
        ("不要關閉泵浦", 0.97, True, "req-no", None),
    ]
    for text, score, confirmed, rid, token in cases:
        result = handle(text, score, confirmed, rid, token, effects, completed)
        print(f"ASR文字={text}｜信心={score:.2f}｜request_id={rid}｜確認={confirmed} → {result}")
    print(f"副作用紀錄：{effects}")


if __name__ == "__main__":
    effects, completed = [], set()
    assert handle("查詢泵浦狀態", 0.40, False, "req-read-low", None, effects, completed) == "請重說：辨識信心不足，未執行任何操作。"
    assert effects == [] and completed == set()
    assert handle("查詢泵浦狀態", 0.99, False, "req-read", None, effects, completed) == "TTS 模擬：泵浦狀態正常（唯讀）"
    assert effects == [] and completed == set()
    main()
```

執行：`python3 E05_voice_agents.py`。關鍵輸出（本次本機實測）：

```text
ASR文字=查詢泵浦狀態｜信心=0.40｜request_id=req-read-low｜確認=False → 請重說：辨識信心不足，未執行任何操作。
ASR文字=查詢泵浦狀態｜信心=0.99｜request_id=req-read｜確認=False → TTS 模擬：泵浦狀態正常（唯讀）
ASR文字=關閉泵浦｜信心=0.62｜request_id=req-close-1｜確認=True → 請重說：辨識信心不足，未執行任何操作。
ASR文字=關閉泵浦｜信心=0.97｜request_id=req-close-1｜確認=False → 待確認：請取得綁定本 request_id 與操作的一次性確認；尚未執行。
ASR文字=關閉泵浦｜信心=0.97｜request_id=req-close-1｜確認=True → 已執行一次：關閉泵浦。
ASR文字=關閉泵浦｜信心=0.97｜request_id=req-close-1｜確認=True → 已處理：重播請求不重複執行。
ASR文字=不要關閉泵浦｜信心=0.97｜request_id=req-no｜確認=True → 已理解為禁止關閉；未執行操作。
副作用紀錄：['關閉泵浦']
```

## Workshop：防止轉錄誤操作

**任務（15 分鐘）**：3 分鐘執行 Demo；4 分鐘把 `confidence` 門檻改為 0.9，再以 0.88 輸入驗證重問；4 分鐘新增「刪除記錄」為高風險操作，要求確認；4 分鐘測試空字串、信心值 `1.2`、布林值 `True` 三種錯誤輸入。

**參考判讀**：門檻降低或提高都須依不同噪音與口音資料評估，不可拿單一值宣稱普遍安全。否定詞一旦不穩，須重問而非只靠字串規則；確認訊息需說明對象與後果。非法輸入要丟出明確錯誤，不能默認為同意。

**驗收**：提交新增案例與副作用紀錄；任何低信心、未確認或否定請求的副作用計數為 0；同一 request_id 的有效確認只能執行一次，重播不得再次觸發。另測 NaN、正負無限、非 bool 確認與錯誤 request_id 綁定皆不執行操作。列出每個案例輸入、輸出、計數器值。此為記憶體內示範，非分散式冪等或真正一次性 token 方案。

## 教師紀錄與下一步

- 證據級別：**本機實測**（2026-10-24；Linux Ubuntu 20.04 x86_64，Python 3.11.2；標準函式庫；模型 revision／套件版本不適用；`python3 E05_voice_agents.py`）。轉錄、信心與輸出是程式內固定合成資料，不是 ASR 或 TTS 的實際量測。
- 限制：未接麥克風、ASR、TTS、語者驗證、真實設備或網路；門檻 0.85 僅示範，不是安全標準；布林語句範例不等於通用語意理解。正式設備還需授權、重播防護、冪等鍵、稽核與急停。
- 尚未實班試教；50 分鐘安排未由真人驗證。教師紀錄只能支持程式控制流程，不能宣稱辨識準確率或學生學習成效。
- 下一步：接第 33 節人工確認與第 27 節流程追蹤，將確認、拒絕、逾時、重播均納入副作用測試；真實語音測試需取得明確同意並制定資料保存規範。
- 舊教材連結：無；本篇為新增選修，未宣稱其他教材已同步。
