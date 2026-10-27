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