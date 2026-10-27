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