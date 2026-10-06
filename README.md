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

**要安裝套件的節次（06、16 起），Python 請用 3.10～3.13（建議 3.10）**：釘選的 torch 2.8.0、numpy 2.2.6 沒有 3.14 的安裝檔，pip 會改裝別的版本。Linux 與 macOS 的安裝指令不同（macOS 不加 `--index-url`），見 [安裝基線](https://hackmd.io/@yillkid/HJl8JXTNkl) 的 INSTALL-BASELINE。

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
| 14 | 多供應商介面、LiteLLM 與 API 成本 | `14_provider_contract.py` | `data/provider-prices-checked.json`、`data/provider-snapshot.json` | — |
| 16 | Embedding 概念 | `16_embedding_lab.py` | — | torch、sentence-transformers（版本見第 16 節講義） |
| 17 | 相似度、失敗案例與門檻評測 | `17_similarity_eval.py` | — | torch、sentence-transformers（版本見第 16 節講義） |
| 18 | Chunk 與來源資料模型 | `18_chunking.py` | `data/sop_corpus.json` | — |
| 19 | 記憶體精確檢索 Baseline | `19_exact_topk.py` | `data/search_cases.json`、`data/sop_corpus.json` | torch、sentence-transformers（版本見第 16 節講義） |
| 22 | Reranking 與 Retrieval Evaluation | `22_rerank_eval.py` | `data/search_cases.json`、`data/sop_corpus.json` | — |
| 23 | RAG 引用與查無答案 | `23_rag_citations.py` | `data/search_cases.json`、`data/sop_corpus.json` | — |
| 25 | 工具權限與副作用控制 | `25_side_effects.py` | — | — |
| 26 | LangGraph State 與 Reducer | `26_state_reducer.py` | — | langgraph==1.2.12（不需模型） |
| 27 | 條件分支與 Unknown 測試 | `27_branch_unknown.py` | — | langgraph==1.2.12（不需模型） |
| 28 | 重試、退避與冪等 | `28_retry_idempotency.py` | — | — |
| 29 | HITL：真正拒絕與恢復 | `29_hitl_resume.py` | — | langgraph==1.2.12、langgraph-checkpoint-sqlite==3.1.1（不需模型） |
| 30 | Trace 觀測與本地免費備援 | `30_local_trace.py` | — | — |
| 31 | 流程里程碑：沿用骨架整合 | `31_integration.py` | `data/search_cases.json`、`data/sop_corpus.json` | langgraph==1.2.12（不需模型；會載入同資料夾的 23、25、28、30 節程式） |
| 32 | MCP 角色與工具參數契約 | `32_mcp_contract.py` | — | mcp==2.3.0（不需模型；會載入同資料夾的 25 節程式） |
| 33 | MCP Client 生命週期與 Async | `33_mcp_lifecycle.py` | — | mcp==2.3.0（不需模型；會開 stdio 子程序，約 10 秒） |
| 41 | SFT 資料與按來源拆分 | `41_sft_split.py` | `data/sop_corpus.json` | — |
| 42 | LoRA 低秩權重與訓練紀錄判讀 | `42_lora_math.py` | `data/model-candidates.json` | — |
| 46 | KV Cache 與 Prefix Cache 的機制與局限 | `46_prefix_cache.py` | — | — |
| 47 | 模型路由、級聯與品質門檻 | `47_routing.py` | `data/eval-set.json` | — |

## 資料來源與限制

- `data/eval-set.json`：教材作者依標籤定義編寫的**合成**巡查通報與教學標註，不是真實通報；gold 由教師覆核。
- `data/model-candidates.json`：Hugging Face 公開 API 的**快照**（日期見檔內），授權與檔案大小會變。
- `data/tokenizer-fixtures.json`：教師在一台機器上的 tokenizer 實測紀錄，不是任何 API 的計費保證。
- `data/provider-snapshot.json`：LiteLLM 套件內建的能力與價目**快照**（離線產生，會過期）；`data/provider-prices-checked.json`：教師手動核對官方價目頁的紀錄（日期與網址見檔內）。
- `data/sop_corpus.json`、`data/search_cases.json`：與 MariaDB 課（[mariadb-demo](https://github.com/4-learn/mariadb-demo)）同一份合成 SOP 語料與標註，兩課共用。

所有資料都是合成或公開資訊，不含個資。
