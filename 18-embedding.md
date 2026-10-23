# Embedding 概念

> **正文改版：20E07-E02-14（20E07-E03-06 修訂：程式改名 `E07_embedding_lab.py`、改由 llm-demo 取得；同版本離線重跑分數不變）｜對應新版 LLM 第 17 節：Embedding 原理與中文適配。**
> 本節真正執行一個中文預訓練模型，在 CPU 上把文字轉為 512 維向量，再比較檢索排名。不使用聊天 API，不需要先懂 SQL；它也是後續 MariaDB Vector 與 RAG 的起點。

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

| 時間 | 活動 | 要觀察的事 |
| --- | --- | --- |
| 0–8 分鐘 | 用字不同，關鍵字搜尋失敗 | 「頭盔」不會字面命中「安全帽」 |
| 8–18 分鐘 | 編碼器、向量、維度與模型版本 | 輸出不是回答，而是一串數字 |
| 18–28 分鐘 | 真實模型 encode、正規化與排序 | 用向量找候選內容 |
| 28–35 分鐘 | 否定句、無答案與錯誤近鄰 | 高分不代表條件符合 |
| 35–50 分鐘 | Workshop：新查詢與近鄰分類 | 結果、錯誤分析與模型紀錄 |

先修：Python、串列與基本矩陣形狀概念。曾讀 [Inference 概念](https://hackmd.io/@yillkid/M1IwAffeRhWVk8nMKD1V7Q) 有幫助，但不需先會聊天 API。Embedding 本身就是推論，沒有生成式 LLM 也能做搜尋或近鄰分類。

## 一、先碰到關鍵字的限制

資料裡寫「工人未配戴安全帽」，使用者卻問「有人沒有戴頭盔就走進工地」。用 Python 的 `"頭盔" in text` 或 SQL 的 `LIKE '%頭盔%'` 都找不到，因為它們是在做字面匹配。

我們可以維護同義詞表，但當句子變長、語序改變或資料領域增加，就需要更多規則。Embedding 提供另一種候選搜尋方式：用模型學到的表示比較文字，而不是只比對相同字串。

不要因此丟掉 SQL：型號 `AB-123`、日期、價格上限、權限與有效版本仍適合明確條件查詢。向量搜尋處理語意接近，並不取代精確限制。

## 二、編碼器實際產生什麼？

```text
文字
  ↓ tokenizer 與模型計算
每個 token 的表示
  ↓ pooling（依模型訓練方式聚合）
固定維度的句子／段落向量
```

本節選用 `BAAI/bge-small-zh-v1.5`，輸出 512 維、模型卡標示 MIT 授權，適合做中文入門實驗，不是「最新模型推薦」。固定 revision：`7999e1d3359715c523056ef9478215996d62a620`。在不同任務與繁體中文資料上仍須評估，不能用模型名稱保證品質。

BGE 這個模型使用第一個 token 的表示作為句向量；其他模型可能使用平均 pooling 或其他方法。不能任意換 pooling 又把它當成同一個檢索模型。

三段文字會得到形狀 `(3, 512)`：三筆資料，每筆 512 個數。文字長短不同，輸出維度仍相同；但輸入有長度上限，過長內容可能被截斷，需要後面的切塊策略。

**每一維通常沒有「安全程度」「價格」「是否戴帽」這種人工可直接命名的意義。** 模型版本與前處理共同決定向量空間；即使兩個模型都輸出 512 維，也不能直接混在一起比較。

## 三、相似度是幾何量，不是事實機率

非零向量的 cosine similarity：

```text
cosine(a, b) = dot(a, b) / (norm(a) * norm(b))
```

本節把每筆向量正規化為長度 1，因此內積 `a @ b` 就是 cosine similarity。未正規化時，內積也受向量長度影響，不能直接當成 cosine。

| 數值 | 可以說 | 不可以說 |
| --- | --- | --- |
| 接近 1 | 兩個向量方向接近 | 文字完全相同、事實完全一致 |
| 接近 0 | 兩個向量接近正交 | 自然語言必定毫無關聯 |
| 接近 -1 | 幾何方向接近相反 | 兩句話就是反義句 |

「已戴安全帽」與「未戴安全帽」談同一主題，可能都很接近。模型找到相關主題，不等於已經確認否定條件。分數 0.8 也不是 80% 的答案正確率，門檻必須依自己的資料評估。

## 四、課前環境

以下安裝命令已在 Linux x86_64／Python 3.10.15 測試。CPU 版 PyTorch 不下載 CUDA 執行環境；安裝與模型下載應在課前完成，不佔 15 分鐘 Workshop。其他 CPU 架構或 Python 版本需另驗證 wheel 相容性。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "pip==25.2"
python -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "sentence-transformers==5.1.2" "transformers==4.57.3" "numpy==2.2.6"
```

首次執行會連線下載公開模型，無需 API key。若網路禁止模型下載，由教師事先準備同一 revision 的快取；不要把未授權外部服務當成必要備援。下載後可用 `HF_HUB_OFFLINE=1` 檢查離線執行。

若只能看教師的預計算輸出，請把結果標成「教師紀錄分析」；它不能處理你新輸入的句子，也不能算本機模型實測。

## 五、完整 Demo：五筆文件，四個查詢

`E07_embedding_lab.py` 的資料全是合成通報，沒有真實個資。模型只做編碼與排名，不會回覆自然語言答案。

<!-- demo: E07_embedding_lab.py -->
```python
import argparse
from importlib.metadata import version
from time import perf_counter

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "BAAI/bge-small-zh-v1.5"
REVISION = "7999e1d3359715c523056ef9478215996d62a620"
# Exact retrieval instruction from this model's official model card.
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
DOCUMENTS = [
    ("D1", "工人未配戴安全帽，正在進入施工區。", "missing"),
    ("D2", "工人已配戴安全帽，正在進入施工區。", "present"),
    ("D3", "逃生出口被紙箱阻擋，通道無法通行。", "blocked_exit"),
    ("D4", "機台旁地面積水，尚未清理。", "wet_floor"),
    ("D5", "倉庫室內溫度為攝氏二十五度。", "temperature"),
]
QUERIES = [
    "有人沒有戴頭盔就走進工地。",
    "工人有戴頭盔，沒有缺少安全帽。",
    "請查找未被紙箱堵住的出口。",
    "機器附近地板濕滑。",
]


def load_model():
    torch.set_num_threads(2)
    return SentenceTransformer(
        MODEL_ID, revision=REVISION, device="cpu",
        trust_remote_code=False, model_kwargs={"use_safetensors": True},
    )


def search(model, queries, documents=DOCUMENTS):
    document_vectors = model.encode(
        [row[1] for row in documents], batch_size=8,
        normalize_embeddings=True, show_progress_bar=False,
    )
    query_vectors = model.encode(
        [QUERY_PREFIX + query for query in queries], batch_size=8,
        normalize_embeddings=True, show_progress_bar=False,
    )
    return query_vectors @ document_vectors.T, document_vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="append", help="repeat for more queries")
    queries = parser.parse_args().query or QUERIES
    started = perf_counter()
    model = load_model()
    loaded = perf_counter()
    scores, vectors = search(model, queries)
    elapsed = perf_counter() - loaded
    print("model:", MODEL_ID, "revision:", REVISION)
    print("versions:", {name: version(name) for name in ("torch", "transformers", "sentence-transformers", "numpy")})
    print("shape:", vectors.shape, "norms:", np.round(np.linalg.norm(vectors, axis=1), 3))
    print("keyword 頭盔:", [row[0] for row in DOCUMENTS if "頭盔" in row[1]])
    for query, row in zip(queries, scores):
        order = sorted(range(len(DOCUMENTS)), key=lambda i: (-float(row[i]), DOCUMENTS[i][0]))[:3]
        print("query:", query)
        for i in order:
            print(DOCUMENTS[i][0], f"{row[i]:.4f}", DOCUMENTS[i][1])
        print("nearest-label (not a verdict):", DOCUMENTS[order[0]][2])
    print(f"load_seconds={loaded - started:.3f} encode_search_seconds={elapsed:.3f}")
```

執行原始四個查詢，或加入自己的問題（程式也放在 [llm-demo](https://github.com/4-learn/llm-demo)：`git clone https://github.com/4-learn/llm-demo.git` 後在該目錄、啟用上面的 venv 後執行）：

```bash
python E07_embedding_lab.py
python E07_embedding_lab.py --query "有人忘了戴頭盔" --query "通道完全暢通"
HF_HUB_OFFLINE=1 python E07_embedding_lab.py
```

文件不加指令前綴，查詢使用模型卡指定的前綴。程式保留官方原字串，所以該行是簡體中文；這是模型輸入規格，不是把整份講義轉成簡體。其他模型的 query／document 設定可能不同。

應看到 `shape: (5, 512)`、各列 norm 約 1，且字面搜尋 `頭盔` 得到空清單。排名與分數請以實際執行為準，不要求不同硬體逐位小數完全相同。

### 怎麼讀結果

以下是 20E07-E02-14 使用本文固定模型與程式的實際 CPU 輸出摘要，不是人工指定排名：

| 查詢 | 第一名／分數 | 觀察 |
| --- | --- | --- |
| 沒有戴頭盔就走進工地 | D1／0.6374 | 找到未戴安全帽，比字面搜尋有用 |
| 有戴頭盔，沒有缺少安全帽 | D1／0.6889；D2 為 0.6575 | **否定條件誤判**，正確描述 D2 反而排第二 |
| 未被紙箱堵住的出口 | D3／0.6904 | 第一名描述的是被堵住，不能當符合條件的答案 |
| 機器附近地板濕滑 | D4／0.5098 | 找到地面積水，但分數不是正確機率 |

不能為了讓 Demo 漂亮就刪掉第二、三題。正是這些錯誤，說明「找相關候選」與「驗證需求是否成立」是兩個步驟。

- 第一題要檢查 D1 是否進入候選，與字面搜尋比較；它是同義說法的例子。
- 第二題拿來觀察「已戴」和「未戴」是否混淆，不因分數高就接受第一名。
- 第三題要求「未被堵住的出口」，資料只有「被堵住」的 D3。即使 D3 排第一，也沒有符合條件的答案；這是刻意安排的無答案查詢。
- 第四題可以觀察「地板濕滑」是否找到「地面積水」D4。

`nearest-label` 只是取第一名文件的人工標籤，展示最簡單的近鄰分類。它不是安全判決，也沒有校準拒絕門檻。這讓我們能比較 [第 01 節規則 baseline](https://hackmd.io/@yillkid/JIq0goMRRz6gkLqntlChYw) 與向量方法，但正式比較要使用同一份獨立測試集，而不是拿不同題目分數硬比。

## Workshop：先寫預期，再看模型

35–40 分鐘：執行原始程式，記錄模型 ID／revision、維度、套件版本與真實硬體，確認這次是本機推論還是分析教師結果。

40–46 分鐘：新增三個查詢，分別是同義說法、否定／精確條件、資料中不存在的需求。**先寫出相關文件 ID 或「無符合文件」，才執行搜尋。** 不要為了讓排名好看而修改人工預期值。

46–50 分鐘：填下表，挑一個錯誤說明為什麼需要其他檢查。若三題都成功，也要分析原始第三題的無答案問題。

| 查詢 | 人工預期 ID／無答案 | 向量 Top-3 | 第一名標籤 | 是否符合完整條件 | 下一步 |
| --- | --- | --- | --- | --- | --- |
| 自訂同義句 | 執行前填 | 執行後填 | 不是判決 | 人工核對 | 保留／覆核 |
| 自訂否定句 | 執行前填 | 執行後填 | 不是判決 | 人工核對 | 補條件／覆核 |
| 自訂無答案 | 無符合文件 | 仍可能有三筆 | 不能當答案 | 不符合 | 回報查無符合內容 |

### 參考判讀與驗收

1. 五筆文件應得到 `(5, 512)`，正規化後每筆長度約 1；這是數值檢查，不是語意品質檢查。
2. 字面搜尋與向量搜尋可能各有優勢。產品編號、有效日期與授權範圍應用明確條件，不能只憑相似度。
3. 原始第三題不能因 D3 排第一就說找到答案；文件明確描述堵住，與查詢條件相反。
4. 同一模型的近鄰標籤可以作分類起點，但要用保留資料評估誤判與拒答，不可把任意 0.8 設成通用安全門檻。
5. 提交三筆人工預期、實際排名、錯誤分析及版本紀錄；不要求「每題模型都答對」才能通過這節課。

## 教師實測與下一步

證據等級：**本機實測**（真實 BGE 模型推論；備課機與示範機 4-learn 的完整測試皆實跑）。

本頁的模型與程式已於 20E07-E02-14 在 Linux x86_64／Python 3.10.15、AMD EPYC 7B12 的 CPU 路徑驗證，固定兩個 PyTorch 執行緒。完成真實模型載入、四題搜尋、向量形狀／正規化檢查與離線快取重跑；**不是學生 i5／16 GB 的效能保證，也尚未完成實班試教**。

本次快取已就緒的單次示範，模型載入約 0.281 秒，五筆文件與四題的 encode／矩陣搜尋約 0.108 秒；不含套件匯入、安裝與下載。這只是測試機上的觀察，不是效能承諾。此處不列普遍準確率；五筆文件與四題只供教學，不能把第一次下載的耗時當穩態推論延遲。

後續第 18 節會深入相似度與失敗評測，第 19–20 節處理 chunk／精確檢索，再接 MariaDB 保存與管理向量。RAG 才會把取回的原文交給生成模型回答；**向量搜尋本身不是 RAG 的全部，也不要求先有聊天機器人**。

參考：[官方模型卡與 query instruction](https://huggingface.co/BAAI/bge-small-zh-v1.5)、[固定模型 revision](https://huggingface.co/BAAI/bge-small-zh-v1.5/tree/7999e1d3359715c523056ef9478215996d62a620)。

[舊 Demo](https://github.com/4-learn/huggingface-demo/blob/main/E02_embedding_demo.py) 與 [舊 Workshop 解答](https://github.com/4-learn/huggingface-workshop/blob/main/model_basics/E02_embedding.py) 尚未同步本次模型與實驗，僅供歷史參照；本節以內嵌完整程式與驗收為準。
