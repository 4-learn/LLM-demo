# LLM 課程 Demo 與資料

4-learn LLM 49 節課程講義內嵌程式的可執行副本。**講義是唯一來源**；本 repo 由 swarm 的
`course-materials/llm/tools/build_llm_demo.py` 從講義產生，請不要直接在這裡修改程式。

講義目錄：<https://hackmd.io/@yillkid/HJl8JXTNkl>

## 使用方式

```bash
git clone https://github.com/4-learn/llm-demo.git
cd llm-demo
python3 04_sampling.py
```

需要 Python 3.10 以上。除下表註明的節次外，程式只用標準函式庫，不需要網路、模型或 API key。

上課前先跑環境檢核（唯讀、不安裝、不連線）：`python3 check_env.py`；要上第 16 節再加 `--full`。

| 節 | 講義 | 程式 | 讀取的資料 | 額外需求 |
| --- | --- | --- | --- | --- |
| 01 | 模型在系統中的角色 | `01_role_baseline.py` | — | — |
| 02 | Tokenizer 與上下文預算 | `02_token_budget.py` | — | — |
| 03 | Inference 概念 | `03_inference_lab.py` | — | — |
| 04 | Sampling 與生成實驗 | `04_sampling.py` | — | — |
| 05 | Hugging Face 模型選型與授權 | `05_model_selection.py` | `data/model-candidates.json` | — |
| 06 | 本地推論載入與 CPU Smoke Test | `06_cpu_smoke.py` | — | torch、transformers、Qwen2.5-0.5B 權重約 942 MiB、可用記憶體約 3.3 GB（見第 06 節講義） |
| 07 | 評測集、指標與可重現基準 | `07_evaluation.py` | `data/eval-set.json` | — |
| 08 | 幻覺、提示注入與權限邊界 | `08_boundaries.py` | — | — |
| 12 | Structured Output：語法、Schema 與語義 | `12_structured_output.py` | — | — |
| 13 | 原生 Function Calling 完整工具回合 | `13_tool_round.py` | — | — |
| 16 | Embedding 概念 | `16_embedding_lab.py` | — | torch、sentence-transformers（版本見第 16 節講義） |
| 17 | 相似度、失敗案例與門檻評測 | `17_similarity_eval.py` | — | torch、sentence-transformers（版本見第 16 節講義） |
| 18 | Chunk 與來源資料模型 | `18_chunking.py` | `data/sop_corpus.json` | — |
| 19 | 記憶體精確檢索 Baseline | `19_exact_topk.py` | `data/search_cases.json`、`data/sop_corpus.json` | torch、sentence-transformers（版本見第 16 節講義） |
| 22 | Reranking 與 Retrieval Evaluation | `22_rerank_eval.py` | `data/search_cases.json`、`data/sop_corpus.json` | — |
| 23 | RAG 引用與查無答案 | `23_rag_citations.py` | `data/search_cases.json`、`data/sop_corpus.json` | — |
| 25 | 工具權限與副作用控制 | `25_side_effects.py` | — | — |
| 28 | 重試、退避與冪等 | `28_retry_idempotency.py` | — | — |
| 30 | Trace 觀測與本地免費備援 | `30_local_trace.py` | — | — |
| 41 | SFT 資料與按來源拆分 | `41_sft_split.py` | `data/sop_corpus.json` | — |
| 42 | LoRA 低秩權重與訓練紀錄判讀 | `42_lora_math.py` | `data/model-candidates.json` | — |
| 46 | KV Cache 與 Prefix Cache 的機制與局限 | `46_prefix_cache.py` | — | — |
| 47 | 模型路由、級聯與品質門檻 | `47_routing.py` | `data/eval-set.json` | — |

## 資料來源與限制

- `data/eval-set.json`：教材作者依標籤定義編寫的**合成**巡查通報與教學標註，不是真實通報；gold 由教師覆核。
- `data/model-candidates.json`：Hugging Face 公開 API 的**快照**（日期見檔內），授權與檔案大小會變。
- `data/tokenizer-fixtures.json`：教師在一台機器上的 tokenizer 實測紀錄，不是任何 API 的計費保證。
- `data/sop_corpus.json`、`data/search_cases.json`：與 MariaDB 課（[mariadb-demo](https://github.com/4-learn/mariadb-demo)）同一份合成 SOP 語料與標註，兩課共用。

所有資料都是合成或公開資訊，不含個資。
