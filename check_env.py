#!/usr/bin/env python3
"""LLM 49 節環境檢核（只用 Python 標準函式庫）。

設計原則：**本檔不 import 任何第三方套件**，所以在 `pip install` 之前就能跑。
它不會修改任何東西、不會安裝套件、預設也不會連線（要連線需自己加 `--network`）。

用法：

    python3 check_env.py              # 最小檢核：Python 版本、標準函式庫、磁碟
    python3 check_env.py --full       # 完整檢核：加上 venv、第 16 節套件版本、模型快取
    python3 check_env.py --full --json
    python3 check_env.py --network    # 額外測試 huggingface.co 是否可達

離開碼：0 = 沒有 FAIL；1 = 有 FAIL；2 = 參數錯誤。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import sys
from importlib import metadata
from pathlib import Path

MIN_PYTHON = (3, 10)
# 釘選的 torch 2.8.0、numpy 2.2.6 在 PyPI 只有 cp39～cp313 的 wheel（2026-10-06 查 PyPI）。
# 3.14 起 pip 會改裝別的 torch 版本、numpy 從原始碼編譯，裝出來就不是講義的版本。
MAX_WHEEL_PYTHON = (3, 13)
RECOMMENDED_PYTHON = (3, 10)

# 依講義 `16-embedding.md` 的安裝指令釘選。版本不同不代表壞掉，
# 但同一份程式在不同版本上不保證逐字相同的輸出與數值。
PINNED_PACKAGES = {
    "torch": "2.8.0",
    "sentence-transformers": "5.1.2",
    "transformers": "4.57.3",
    "numpy": "2.2.6",
    "langgraph": "1.2.12",   # 第 26 節起；不需模型
    "langgraph-checkpoint-sqlite": "3.1.1",   # 第 29 節
    "mcp": "2.3.0",   # 第 32 節起；2.x 把 FastMCP 改名為 MCPServer
}

MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"

# (名稱, 模型 ID, revision, 用到的節次)。第 06 節的生成模型與第 02、05 節同一 revision。
MODELS = (
    ("bge-small-zh", MODEL_ID, MODEL_REVISION, "16、17、19"),
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775", "06"),
)

# 2026-10-06 備課機實測：venv 安裝後 1.5 GB、Qwen 快取 954 MB、bge 93 MB，合計約 2.6 GB；
# 加上 pip 下載暫存，保守抓 4 GB。（舊值 3 GB 是只算 torch wheel 下載大小時的估計，太少。）
MIN_FREE_GB = 4.0

# 第 06 節備課機實測峰值 3216–3218 MiB（Qwen2.5-0.5B 以 float32 載入）。
SMOKE_RAM_MIB = 3300

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


class Result:
    def __init__(self, name: str, status: str, detail: str, hint: str = "") -> None:
        self.name = name
        self.status = status
        self.detail = detail
        self.hint = hint

    def as_dict(self) -> dict:
        data = {"check": self.name, "status": self.status, "detail": self.detail}
        if self.hint:
            data["hint"] = self.hint
        return data


def parse_version(text: str) -> tuple:
    """把版本字串轉成可比較的整數 tuple；遇到非數字就停在該處。"""
    parts = []
    for chunk in str(text).split("."):
        digits = ""
        for character in chunk:
            if character.isdigit():
                digits += character
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def package_version(name: str):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def nearest_existing(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def hf_cache_dir() -> Path:
    """回傳 Hugging Face hub 快取目錄，邏輯與 huggingface_hub 一致。"""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]) / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_snapshot_dir(model_id: str = MODEL_ID, revision: str = MODEL_REVISION) -> Path:
    folder = "models--" + model_id.replace("/", "--")
    return hf_cache_dir() / folder / "snapshots" / revision


def check_python() -> Result:
    current = sys.version_info[:2]
    text = f"{sys.version.split()[0]}（{sys.executable}）"
    if current < MIN_PYTHON:
        return Result(
            "Python 版本", FAIL, text,
            f"需要 {MIN_PYTHON[0]}.{MIN_PYTHON[1]} 以上。本課範例使用 f-string、"
            "`match` 以外的現代語法與 `importlib.metadata`。",
        )
    if current > MAX_WHEEL_PYTHON:
        return Result(
            "Python 版本", WARN, text,
            f"第 01～05 節可以上；但釘選的 torch 2.8.0、numpy 2.2.6 沒有 {current[0]}.{current[1]} 的安裝檔，"
            f"pip 會改裝別的版本。要上第 06、16 節以後，請改用 {RECOMMENDED_PYTHON[0]}.{RECOMMENDED_PYTHON[1]}"
            f"（最高 {MAX_WHEEL_PYTHON[0]}.{MAX_WHEEL_PYTHON[1]}）重建 venv，見 INSTALL-BASELINE。",
        )
    if current != RECOMMENDED_PYTHON:
        return Result(
            "Python 版本", WARN, text,
            f"教材是在 {RECOMMENDED_PYTHON[0]}.{RECOMMENDED_PYTHON[1]} 上實測。"
            "其他版本可執行，但數值與輸出順序不保證逐字相同。",
        )
    return Result("Python 版本", PASS, text)


def check_stdlib() -> Result:
    """本課第 01～04 節只依賴標準函式庫，先確認關鍵模組真的能載入。"""
    import importlib

    required = ("math", "random", "json", "statistics", "collections", "pathlib")
    broken = []
    for name in required:
        try:
            importlib.import_module(name)
        except Exception as error:  # pragma: no cover - 只有壞掉的環境會走到
            broken.append(f"{name}（{type(error).__name__}）")
    if broken:
        return Result(
            "標準函式庫", FAIL, "無法載入：" + "、".join(broken),
            "第 01～04 節不需要任何第三方套件，這裡失敗代表 Python 安裝不完整。",
        )
    # 自行編譯的 Python（pyenv）若編譯時找不到 OpenSSL／SQLite，這些模組會缺：
    # 缺 ssl → pip 無法下載；缺 blake2b → hashlib 載入時印出一長串 ERROR；缺 sqlite3 → 第 29 節失敗。
    # 2026-10-06 教師 M4 Pro 的 pyenv 3.11.9 實際出現 blake2b／blake2s 缺失。
    missing = [label for label, probe in compiled_module_probes() if not probe()]
    if missing:
        return Result(
            "標準函式庫", FAIL, "這個 Python 編譯不完整，缺：" + "、".join(missing),
            "通常是 pyenv 編譯時找不到 OpenSSL（或之後 Homebrew 升級了 OpenSSL）。"
            "換一個完整的 Python 重建 venv，見 INSTALL-BASELINE「Python 本身壞掉」。",
        )
    return Result(
        "標準函式庫", PASS, "第 01～04 節所需模組皆可載入；ssl、hashlib、sqlite3 完整",
    )


def compiled_module_probes():
    """(說明, 檢查函式)。檢查函式回 True 代表可用；任何例外都視為不可用。"""
    def safe(check):
        def run():
            try:
                return bool(check())
            except Exception:
                return False
        return run

    def has_ssl():
        import ssl
        return ssl.OPENSSL_VERSION

    def has_blake2():
        import hashlib
        return hashlib.blake2b(b"x").hexdigest() and hashlib.sha256(b"x").hexdigest()

    def has_sqlite():
        import sqlite3
        return sqlite3.connect(":memory:").execute("select 1").fetchone()

    return (("ssl（pip 下載需要）", safe(has_ssl)),
            ("hashlib blake2／sha256", safe(has_blake2)),
            ("sqlite3（第 29 節）", safe(has_sqlite)))


def check_venv() -> Result:
    inside = sys.prefix != sys.base_prefix
    text = f"prefix={sys.prefix}"
    if inside:
        return Result("虛擬環境", PASS, f"已在虛擬環境內（{text}）")
    return Result(
        "虛擬環境", FAIL, f"不在虛擬環境內（{text}）",
        "第 16 節會安裝 torch 等大型套件，必須在 venv 內："
        "`python3 -m venv .venv && source .venv/bin/activate`。",
    )


def check_disk() -> Result:
    target = nearest_existing(hf_cache_dir())
    try:
        usage = shutil.disk_usage(target)
    except OSError as error:  # pragma: no cover
        return Result("磁碟空間", WARN, f"無法查詢 {target}（{error}）")
    free_gb = usage.free / 1024 ** 3
    detail = f"{free_gb:.1f} GB 可用（{target}）；第 06、16 節建議至少 {MIN_FREE_GB:.0f} GB"
    if free_gb < MIN_FREE_GB:
        return Result(
            "磁碟空間", FAIL, detail,
            "torch 的 CPU wheel 與模型快取都放在這個分割區。空間不足時先清快取或改用其他分割區。",
        )
    return Result("磁碟空間", PASS, detail)


def check_packages() -> list:
    results = []
    for name, expected in sorted(PINNED_PACKAGES.items()):
        actual = package_version(name)
        if actual is None:
            results.append(Result(
                f"套件 {name}", FAIL, f"未安裝（講義釘選 {expected}）",
                f"在 venv 內執行 `python -m pip install \"{name}=={expected}\"`；"
                "缺套件時第 16 節無法載入模型。",
            ))
        elif actual.split("+")[0] == expected:
            # `+cpu` 是 PEP 440 的本地版本標記：照講義從 PyTorch CPU index 安裝就會得到 2.8.0+cpu，
            # 它就是釘選的 2.8.0，不能報警告（2026-10-06 實裝時發現的誤報）。
            results.append(Result(f"套件 {name}", PASS, actual))
        else:
            results.append(Result(
                f"套件 {name}", WARN, f"已安裝 {actual}，講義釘選 {expected}",
                "可以執行，但輸出的數值與向量不保證與教材一致。要對照教材數字就裝回釘選版本。",
            ))
    return results


def check_cpu_threads() -> Result:
    count = os.cpu_count()
    detail = f"os.cpu_count() = {count}"
    if not count or count < 2:
        return Result(
            "CPU 執行緒", WARN, detail,
            "第 16 節用 `torch.set_num_threads(2)`；可用執行緒過少時速度會明顯變慢。",
        )
    return Result("CPU 執行緒", PASS, detail)


def check_model_cache(label: str = "bge-small-zh", model_id: str = MODEL_ID,
                      revision: str = MODEL_REVISION, sections: str = "16、17、19") -> Result:
    """確認快取裡真的有模型，而不是只有 tokenizer。

    只檢查「目錄存在」會誤判：`tokenizers`／`huggingface_hub` 抓一個
    `tokenizer.json` 就會建出同名 snapshot 目錄，但完全沒有權重。
    """
    name = f"模型快取 {label}（第 {sections} 節）"
    snapshot = model_snapshot_dir(model_id, revision)
    if not snapshot.is_dir():
        return Result(
            name, WARN, f"尚未下載（預期位置 {snapshot}）",
            f"第 {sections} 節需要這個模型。若教室禁止連外，請由教師事先以同一 revision 備妥快取；"
            "課堂上才不會整班同時下載。",
        )
    present = {entry.name for entry in snapshot.iterdir()}
    has_weights = any(name in present for name in ("model.safetensors", "pytorch_model.bin"))
    missing = [name for name in ("config.json",) if name not in present]
    if not has_weights:
        missing.append("model.safetensors（或 pytorch_model.bin）")
    if missing:
        return Result(
            name, WARN,
            f"不完整：{snapshot} 只有 {sorted(present) or '空目錄'}，缺 {'、'.join(missing)}",
            f"這通常代表只抓過 tokenizer（第 02 節的教師工具就會這樣）。第 {sections} 節需要完整權重；"
            "離線環境下會直接失敗。",
        )
    return Result(name, PASS, f"已備妥權重（{snapshot}）")


def read_meminfo(path: str = "/proc/meminfo"):
    """回傳 (總量 MiB, 可用 MiB)；非 Linux 或讀不到時回 (None, None)。"""
    total = available = None
    try:
        with open(path, encoding="ascii") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key == "MemTotal":
                    total = int(rest.split()[0]) // 1024
                elif key == "MemAvailable":
                    available = int(rest.split()[0]) // 1024
    except (OSError, ValueError, IndexError):
        return None, None
    return total, available


def parse_vm_stat(text: str):
    """macOS `vm_stat` 輸出 → 可用 MiB（free＋inactive＋speculative＋purgeable 頁）。讀不懂回 None。"""
    import re

    size = re.search(r"page size of (\d+) bytes", text)
    if not size:
        return None
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        found = re.search(rf"^{key}:\s+(\d+)\.", text, re.M)
        if found:
            pages += int(found.group(1))
    return pages * int(size.group(1)) // (1024 * 1024)


def read_memory():
    """Linux 讀 /proc/meminfo；macOS 用 sysctl 與 vm_stat（系統內建指令，唯讀）。"""
    total, available = read_meminfo()
    if available is not None or platform.system() != "Darwin":
        return total, available
    import subprocess

    try:
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                   timeout=5).stdout.strip()) // (1024 * 1024)
        available = parse_vm_stat(subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, None
    return total, available


def check_memory(meminfo=read_memory) -> Result:
    total, available = meminfo()
    if available is None:
        return Result(
            "記憶體", WARN, "讀不到可用記憶體（不是 Linux／macOS，或系統指令無法執行）",
            f"第 06 節載入生成模型實測峰值約 {SMOKE_RAM_MIB} MiB；請自行在系統監視器確認，"
            "上課時先關掉瀏覽器等大程式。",
        )
    detail = f"總量 {total} MiB、目前可用 {available} MiB；第 06 節需約 {SMOKE_RAM_MIB} MiB"
    if available < SMOKE_RAM_MIB:
        return Result(
            "記憶體", WARN, detail,
            "其他節次不受影響；第 06 節的 smoke test 會在載入前停下並寫紀錄（離開碼 2）。"
            "先關掉其他程式再試，仍不足就使用教師提供的備援紀錄。",
        )
    return Result("記憶體", PASS, detail)


def check_platform(system: str = None) -> Result:
    system = system or platform.system()
    if system == "Windows":
        return Result(
            "作業系統", WARN, "Windows",
            "第 06 節程式用到 `resource` 模組，Windows 沒有；請改用 WSL（Ubuntu）執行。其餘節次可在 Windows 執行。",
        )
    return Result("作業系統", PASS, f"{system} {platform.machine()}")


def check_network(timeout: float = 5.0) -> Result:
    try:
        with socket.create_connection(("huggingface.co", 443), timeout=timeout):
            pass
    except OSError as error:
        return Result(
            "對外連線", WARN, f"無法連線 huggingface.co:443（{type(error).__name__}）",
            "若模型快取已備妥，離線仍可上課（`HF_HUB_OFFLINE=1`）。",
        )
    return Result("對外連線", PASS, "huggingface.co:443 可達")


def run_checks(full: bool, network: bool) -> list:
    results = [check_python(), check_stdlib()]
    if full:
        results.append(check_venv())
        results.append(check_cpu_threads())
        results.append(check_platform())
        results.append(check_memory())
        results.extend(check_packages())
        results.extend(check_model_cache(*model) for model in MODELS)
    results.append(check_disk())
    if network:
        results.append(check_network())
    else:
        results.append(Result("對外連線", SKIP, "未測試（加上 --network 才會連線）"))
    return results


def print_report(results: list) -> None:
    width = max(len(item.name) for item in results)
    marks = {PASS: "✅", WARN: "⚠️ ", FAIL: "❌", SKIP: "➖"}
    for item in results:
        print(f"{marks[item.status]} {item.name.ljust(width)}  {item.detail}")
        if item.hint and item.status in (WARN, FAIL):
            print(f"   {' ' * width}  ↳ {item.hint}")
    counts = {status: sum(1 for item in results if item.status == status)
              for status in (PASS, WARN, FAIL, SKIP)}
    print()
    print(f"合計：{counts[PASS]} 通過、{counts[WARN]} 警告、{counts[FAIL]} 失敗、{counts[SKIP]} 略過")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="LLM 49 節環境檢核（唯讀，不安裝任何東西）",
    )
    parser.add_argument("--full", action="store_true",
                        help="加上虛擬環境、package 版本、模型快取檢查")
    parser.add_argument("--network", action="store_true",
                        help="額外測試 huggingface.co:443 是否可達（預設不連線）")
    parser.add_argument("--json", action="store_true", help="以 JSON 輸出，方便教師收集")
    args = parser.parse_args(argv)

    results = run_checks(full=args.full, network=args.network)

    if args.json:
        payload = {
            "profile": "full" if args.full else "minimal",
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "in_venv": sys.prefix != sys.base_prefix,
            "checks": [item.as_dict() for item in results],
            "failed": sum(1 for item in results if item.status == FAIL),
            "warned": sum(1 for item in results if item.status == WARN),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_report(results)

    return 1 if any(item.status == FAIL for item in results) else 0


if __name__ == "__main__":
    sys.exit(main())
