"""第 06 節：API request/response 的入門示範。

預設僅重播固定 fixture，無網路、無金鑰、無第三方套件。
只有明確加上 --live 並設定 NVIDIA_API_KEY 才會送出真實請求；不構成工安建議。
"""
import json
import os
import sys
import urllib.error
import urllib.request

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
    body = json.dumps(request).encode("utf-8")
    req = urllib.request.Request(
        NVIDIA_BASE_URL + "/chat/completions", data=body, method="POST",
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"API 回傳 HTTP {exc.code}；不要貼出含憑證的請求或錯誤內容") from None
    except urllib.error.URLError:
        raise RuntimeError("連線失敗或逾時；不要自動重試，服務可能已收到請求") from None


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
