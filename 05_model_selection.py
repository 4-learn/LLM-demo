"""第 05 節：模型選型與授權檢核（只用 Python 標準函式庫、不下載權重）。

資料是 `data/model-candidates.json`：教師於 2026-10-05 從 Hugging Face
公開 API 抓的真實快照，含授權、gated 狀態、官方參數量與權重檔大小。
授權與 gated 狀態會變，引用前必須重新查核。
"""

import json
from pathlib import Path

RAM_BUDGET_MB = 6144          # 這台教室機器願意給模型多少記憶體
NEEDED_CONTEXT = 8192         # 任務需要吃多長的上下文
AUTO_APPROVED = {"mit", "apache-2.0"}
DTYPE_ALIASES = {
    "float32": "float32", "fp32": "float32", "f32": "float32",
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float16": "float16", "fp16": "float16", "f16": "float16",
}
BYTES_PER_DTYPE = {"float32": 4, "bfloat16": 2, "float16": 2}


def load_candidates():
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        path = base / "data" / "model-candidates.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("找不到 data/model-candidates.json")


def normalize_dtype(raw):
    return None if raw is None else DTYPE_ALIASES.get(str(raw).strip().lower())


def is_encoder_only(config):
    """BertModel 這種純編碼器沒有輸出層；Qwen2ForCausalLM 這種有。"""
    return not any("For" in name for name in config.get("architectures", []))


def estimate_parameters(config, naive=False):
    """由 config 估參數量。naive=True 是「所有模型都當 BERT」的天真公式。"""
    h, layers, heads = config["hidden_size"], config["num_hidden_layers"], config["num_attention_heads"]
    head_dim = h // heads
    if head_dim * heads != h:
        raise ValueError("hidden_size 必須能被 num_attention_heads 整除")
    if naive:
        mlp_factor, kv_heads, tied = 2, heads, False
        pos, pooler = config["max_position_embeddings"] * h, h * h
    else:
        bert = config["model_type"] == "bert"
        mlp_factor = 2 if bert else 3            # BERT 2 個矩陣；SwiGLU 3 個
        kv_heads = config.get("num_key_value_heads", heads)
        tied = config.get("tie_word_embeddings", False)
        pos = config["max_position_embeddings"] * h if bert else 0   # RoPE 無學習式位置嵌入
        pooler = h * h if bert else 0
    embed = config["vocab_size"] * h
    lm_head = 0 if (tied or is_encoder_only(config)) else config["vocab_size"] * h
    kv_dim = kv_heads * head_dim
    attn = 2 * h * h + 2 * h * kv_dim
    mlp = mlp_factor * h * config["intermediate_size"]
    return embed + lm_head + layers * (attn + mlp + 2 * h) + pos + pooler


def dtype_of(entry):
    for raw in entry["official_dtypes"]:
        if normalize_dtype(raw):
            return normalize_dtype(raw)
    return normalize_dtype((entry["config"] or {}).get("torch_dtype"))


def weight_mb(entry):
    """官方參數量 × dtype 位元組數。這是初篩用的估算，不是實測。"""
    dtype = dtype_of(entry)
    if entry["official_parameter_count"] is None or dtype is None:
        return None
    return entry["official_parameter_count"] * BYTES_PER_DTYPE[dtype] / 1024 ** 2


def actual_weight_mb(entry):
    """官方 safetensors 檔案的實際位元組數。"""
    total = entry["weight_file_bytes"]
    return total / 1024 ** 2 if total else None


def kv_cache(entry, seq_len):
    """回傳 (MB 或 None, 原因)。KV cache = 2(K,V) × 層 × kv_heads × head_dim × 長度 × 位元組。

    只有自迴歸解碼器需要 KV cache：它一次只算一個 token，得把前面算過的
    K、V 留著。BERT 這種編碼器一次看完整個序列，沒有 KV cache 可算——
    對它套這個公式會得到一個不存在、也不該拿去比大小的數字。
    """
    config = entry["config"]
    if config is None:
        return None, "config 不可讀"
    if is_encoder_only(config):
        return None, "編碼器，一次看完整個序列，沒有 KV cache"
    dtype = normalize_dtype(config.get("torch_dtype"))
    if dtype is None:
        return None, f"dtype {config.get('torch_dtype')} 未知"
    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", heads)
    head_dim = config["hidden_size"] // heads
    mb = 2 * config["num_hidden_layers"] * kv_heads * head_dim * seq_len * BYTES_PER_DTYPE[dtype] / 1024 ** 2
    return mb, None


def review(entry):
    """回傳 (判決, 理由)。所有理由都收集，不短路——選型要看全貌。"""
    reject, check = [], []
    license_id, config = entry["license"], entry["config"]
    if license_id is None:
        reject.append("授權未知，不得直接核准")
    elif license_id not in AUTO_APPROVED:
        check.append(f"授權 '{license_id}' 不在自動核准清單，需人工確認條款")
    if entry["gated"]:
        check.append(f"gated={entry['gated']}，需先接受條款並登入才能下載")
    if config is None:
        check.append(f"config.json 回 HTTP {entry['config_status']}，架構未知、KV cache 算不出來")
    elif config.get("has_auto_map"):
        check.append("模型倉庫帶自訂程式碼（auto_map），載入時可能執行它")
    mb = weight_mb(entry)
    if mb is None:
        check.append("無法估算權重記憶體")
    elif mb > RAM_BUDGET_MB:
        reject.append(f"權重 {mb:.0f} MB 超過預算 {RAM_BUDGET_MB} MB")
    if config and config.get("max_position_embeddings", 0) < NEEDED_CONTEXT:
        reject.append(f"上下文只有 {config['max_position_embeddings']} tokens，不足 {NEEDED_CONTEXT}")
    return ("淘汰" if reject else "待人工複核" if check else "核准"), reject + check


def main():
    data = load_candidates()
    print(f"資料 {data['kind']}，抓取於 {data['fetched_at']}")
    print(f"情境：機器給模型 {RAM_BUDGET_MB} MB；任務需要 {NEEDED_CONTEXT} tokens 上下文")
    print()

    print("=== 1. 由 config 估參數量，對照官方公布值 ===")
    for entry in data["candidates"]:
        config, official = entry["config"], entry["official_parameter_count"]
        if config is None:
            print(f"{entry['id']:<33} config HTTP {entry['config_status']}，只能用官方值 {official:,}")
            continue
        naive, aware = estimate_parameters(config, True), estimate_parameters(config)
        print(f"{entry['id']:<33} 天真 {naive:>13,} ({naive / official - 1:+6.1%})"
              f"  看架構 {aware:>13,} ({aware / official - 1:+7.3%})")
    print()

    print("=== 2. 權重記憶體：估算 vs 實際 safetensors 檔案 ===")
    for entry in data["candidates"]:
        est, actual = weight_mb(entry), actual_weight_mb(entry)
        print(f"{entry['id']:<33} 估算 {est:8.1f} MB   實際 {actual:9.1f} MB   {(est / actual - 1):+6.2%}")
    print()

    print(f"=== 3. KV cache（{NEEDED_CONTEXT} tokens，單一序列）===")
    for entry in data["candidates"]:
        kv, why = kv_cache(entry, NEEDED_CONTEXT)
        if kv is None:
            print(f"{entry['id']:<33} 不適用／算不出來：{why}")
            continue
        config = entry["config"]
        heads = config["num_attention_heads"]
        kv_heads = config.get("num_key_value_heads", heads)
        note = (f"  ← GQA：{heads} 個 query head 只用 {kv_heads} 個 KV head"
                if kv_heads != heads else "")
        print(f"{entry['id']:<33} {kv:8.1f} MB{note}")
        if kv_heads != heads:
            same = dict(entry, config=dict(config, num_key_value_heads=heads))
            print(f"{'  同一個模型，假設它沒用 GQA':<33} {kv_cache(same, NEEDED_CONTEXT)[0]:8.1f} MB")
    print()

    print("=== 4. 選型檢核：授權 → gated → 記憶體 → 上下文 ===")
    for entry in data["candidates"]:
        verdict, reasons = review(entry)
        print(f"[{verdict}] {entry['id']}（授權 {entry['license']}）")
        for reason in reasons:
            print(f"        - {reason}")


if __name__ == "__main__":
    main()
