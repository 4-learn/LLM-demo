"""第 06 節：本地推論載入與 CPU smoke test。

每一步都可能失敗，而且失敗要能說出原因：快取缺檔、記憶體不夠、模板不合、逾時。
需要第 16 節的套件（另加 transformers 的生成模型）與已下載的 Qwen2.5-0.5B-Instruct 權重。
標準輸出只印可重現的內容；載入時間、延遲與記憶體每次不同，印到標準錯誤並寫進紀錄檔。
"""
import argparse
import json
import os
import platform
import resource
import sys
from pathlib import Path
from time import perf_counter

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"   # 與第 02、05 節相同
PROMPT = "用一句話說明：工人進入工地前要做什麼安全檢查？"
MAX_NEW_TOKENS = 40
MAX_SECONDS = 120          # 單次生成的上限；超過就停，記為 timeout
OVERHEAD = 1.5             # 載入時的暫存、框架本身；第 05 節只算權重，實際一定更多


class SmokeFailure(Exception):
    """帶分類的失敗：stage 是哪一步、kind 是哪一類原因。"""

    def __init__(self, stage, kind, detail):
        super().__init__(f"[{stage}] {kind}: {detail}")
        self.stage, self.kind, self.detail = stage, kind, detail


def available_mib():
    """Linux 讀 /proc/meminfo 的 MemAvailable；其他系統回 None（要自己填）。"""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def peak_rss_mib():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak // (1024 * 1024) if sys.platform == "darwin" else peak // 1024   # macOS 單位是 bytes


def locate(model_id, revision):
    """只找本機快取，絕不連網。缺檔時回報是哪一個 revision 找不到。"""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(snapshot_download(model_id, revision=revision, local_files_only=True))
    except LocalEntryNotFoundError as error:
        raise SmokeFailure("locate", "missing_cache", f"{model_id}@{revision[:7]} 不在本機快取") from error


def weight_mib(path):
    return sum(f.stat().st_size for f in path.glob("*.safetensors")) / 2**20


def memory_gate(need_mib, have_mib):
    """載入前先判斷；不夠就不要載入（載到一半被 OOM killer 殺掉，連紀錄都留不下）。"""
    if have_mib is None:
        raise SmokeFailure("memory", "unknown_ram", "讀不到可用記憶體；請用 --ram-mib 手動填")
    if need_mib > have_mib:
        raise SmokeFailure("memory", "insufficient_ram", f"預估需要 {need_mib:.0f} MiB，可用 {have_mib} MiB")
    return True


def check_template(tokenizer):
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise SmokeFailure("template", "no_chat_template", "這個 tokenizer 沒有對話模板；不能用 chat 格式")
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": "測試"}],
                                             tokenize=False, add_generation_prompt=True)
    if "測試" not in rendered:
        raise SmokeFailure("template", "template_drops_content", "模板輸出不含使用者內容")
    return rendered


def generate(model, tokenizer, text, use_template, max_new_tokens=MAX_NEW_TOKENS, max_time=MAX_SECONDS):
    import torch

    if use_template:
        text = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                             tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    started = perf_counter()
    with torch.no_grad():
        # 明寫所有解碼參數：模型自帶的 generation_config 預設 do_sample=True、repetition_penalty=1.1
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, max_time=max_time,
                                do_sample=False, temperature=None, top_p=None, top_k=None,
                                repetition_penalty=1.0)
    seconds = perf_counter() - started
    new = output[0][inputs["input_ids"].shape[1]:]
    if seconds >= max_time:
        raise SmokeFailure("generate", "timeout", f"{max_time} 秒內只產生 {len(new)} 個 token")
    return {"prompt_tokens": int(inputs["input_ids"].shape[1]), "new_tokens": int(len(new)),
            "text": tokenizer.decode(new, skip_special_tokens=True), "seconds": seconds}


def measure(record, key, value, unit):
    record["measurements"][key] = value
    print(f"[量測] {key}: {value}{unit}", file=sys.stderr)


def run(ram_mib=None, record_path=None):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    record = {"model": f"{MODEL_ID}@{REVISION}", "python": platform.python_version(),
              "torch": torch.__version__, "transformers": transformers.__version__,
              "machine": platform.machine(), "system": platform.system(),
              "threads": torch.get_num_threads(), "measurements": {}, "result": None}
    print("model:", MODEL_ID, "revision:", REVISION)

    print("\n== 一、只從本機快取找權重 ==")
    try:
        locate(MODEL_ID, "0" * 40)
    except SmokeFailure as failure:
        print("故意用不存在的 revision →", failure.kind, "｜", failure.detail)
    path = locate(MODEL_ID, REVISION)
    print("快取路徑最後一層 == revision:", path.name == REVISION)
    print("權重檔:", sorted(f.name for f in path.glob("*.safetensors")), f"{weight_mib(path):.1f} MiB")

    print("\n== 二、載入前先算記憶體 ==")
    need = weight_mib(path) * 2 * OVERHEAD   # 檔案是 bf16；載入成 float32 會變兩倍
    print(f"預估需要: 權重 {weight_mib(path):.1f} MiB × 2（bf16→float32）× {OVERHEAD} = {need:.0f} MiB")
    try:
        memory_gate(need, 1024)
    except SmokeFailure as failure:
        print("假設只有 1024 MiB →", failure.kind, "｜", failure.detail)
    have = ram_mib if ram_mib is not None else available_mib()
    record["measurements"]["available_mib_before_load"] = have
    try:
        memory_gate(need, have)
    except SmokeFailure as failure:
        print("本機 →", failure.kind, "（停止，不載入）")
        record["result"] = {"ok": False, "stage": failure.stage, "kind": failure.kind, "detail": failure.detail}
        return record
    print("本機記憶體足夠: True")

    print("\n== 三、載入 tokenizer 與模板 ==")
    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(path)   # 傳路徑，不傳 ID：見講義「離線載入的坑」
    rendered = check_template(tokenizer)
    print("模板檢查通過；模板把一句「測試」包成:")
    print(rendered)

    print("== 四、載入權重 ==")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    model.eval()
    measure(record, "load_seconds", round(perf_counter() - started, 2), " 秒")
    measure(record, "peak_rss_mib_after_load", peak_rss_mib(), " MiB")
    print("dtype:", model.dtype, "參數量:", f"{sum(p.numel() for p in model.parameters()):,}")
    defaults = model.generation_config
    print("模型自帶的生成預設:", {k: getattr(defaults, k) for k in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty")})

    print("\n== 五、單次短生成（greedy，最多", MAX_NEW_TOKENS, "token）==")
    for use_template in (True, False):
        out = generate(model, tokenizer, PROMPT, use_template)
        label = "有模板" if use_template else "沒模板"
        measure(record, f"generate_seconds_{label}", round(out["seconds"], 2), " 秒")
        measure(record, f"tokens_per_second_{label}", round(out["new_tokens"] / out["seconds"], 1), "")
        print(f"[{label}] 輸入 {out['prompt_tokens']} token，產生 {out['new_tokens']} token")
        print(out["text"])
        print("---")
        if use_template:
            record["sample_output"] = out["text"]
    measure(record, "peak_rss_mib", peak_rss_mib(), " MiB")
    record["result"] = {"ok": True}
    print("smoke test: 通過")
    if record_path:
        Path(record_path).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("紀錄已寫入", Path(record_path).name, file=sys.stderr)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ram-mib", type=int, help="手動指定可用記憶體（讀不到或想模擬小機器時）")
    parser.add_argument("--record", help="把硬體、版本與量測寫成 JSON")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    result = run(args.ram_mib, args.record)
    if not result["result"]["ok"]:
        if args.record:
            Path(args.record).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        sys.exit(2)
