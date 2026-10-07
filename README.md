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
| 09 | 多模態輸入、證據與未知 | `09_vision_unknown.py` | `data/09-vision-record.json` | 重播只需標準函式庫（有 Pillow 會核對圖片雜湊）；--live 需要 torch、transformers==4.57.3、Pillow 與 SmolVLM-500M-Instruct 模型快取（約 3.3 GB 記憶體） |
| 10 | 可核對的圖像描述 | `10_grounded_caption.py` | `data/10-caption-record.json` | 重播只需標準函式庫（有 Pillow 會核對圖片雜湊）；--live 需求同 09_vision_unknown.py |
| 11 | 文件摘要、來源與頁碼 | `11_doc_citations.py` | `data/11-doc-record.json` | 重播只需標準函式庫（有 Pillow 會核對圖片雜湊）；--live 需求同 09_vision_unknown.py |
| 12 | Structured Output：語法、Schema 與語義 | `12_structured_output.py` | — | — |
| 13 | 原生 Function Calling 完整工具回合 | `13_tool_round.py` | — | — |
| 14 | 多供應商介面、LiteLLM 與 API 成本 | `14_provider_contract.py` | `data/provider-prices-checked.json`、`data/provider-snapshot.json` | — |
| 15 | VLM 打標與 Metadata 品質 | `15_vlm_tagging.py` | `data/15-tagging-record.json` | 需要同資料夾的 09_vision_unknown.py；重播只需標準函式庫；--live 需求同 09_vision_unknown.py |
| 16 | Embedding 概念 | `16_embedding_lab.py` | — | torch、sentence-transformers（版本見第 16 節講義） |
| 17 | 相似度、失敗案例與門檻評測 | `17_similarity_eval.py` | — | torch、sentence-transformers（版本見第 16 節講義） |
| 18 | Chunk 與來源資料模型 | `18_chunking.py` | `data/sop_corpus.json` | — |
| 19 | 記憶體精確檢索 Baseline | `19_exact_topk.py` | `data/search_cases.json`、`data/sop_corpus.json` | torch、sentence-transformers（版本見第 16 節講義） |
| 20 | MariaDB Vector 持久化銜接 | `20_vector_store.py` | `data/20-vectors.json`、`data/search_cases.json`、`data/sop_corpus.json` | fixtures 模式只需標準函式庫；MariaDB 模式需要 mariadb==1.1.14、~/mariadb-course-app.json 與教師預建的 llm_chunk_vectors；載入同資料夾第 19 節程式 |
| 21 | Metadata 過濾與版本更新 | `21_filter_update.py` | `data/21-update-vectors.json`、`data/search_cases.json`、`data/sop_corpus.json` | 同第 20 節（fixtures 或 MariaDB）；載入同資料夾第 19、20 節程式 |
| 22 | Reranking 與 Retrieval Evaluation | `22_rerank_eval.py` | `data/search_cases.json`、`data/sop_corpus.json` | — |
| 23 | RAG 引用與查無答案 | `23_rag_citations.py` | `data/search_cases.json`、`data/sop_corpus.json` | — |
| 24 | Orchestration 與 LangChain LCEL | `24_lcel.py` | `data/search_cases.json`、`data/sop_corpus.json` | langchain-core==1.6.6（隨 langgraph==1.2.12 安裝）；載入同資料夾第 23 節程式；不需模型 |
| 25 | 工具權限與副作用控制 | `25_side_effects.py` | — | — |
| 26 | LangGraph State 與 Reducer | `26_state_reducer.py` | — | langgraph==1.2.12（不需模型） |
| 27 | 條件分支與 Unknown 測試 | `27_branch_unknown.py` | — | langgraph==1.2.12（不需模型） |
| 28 | 重試、退避與冪等 | `28_retry_idempotency.py` | — | — |
| 29 | HITL：真正拒絕與恢復 | `29_hitl_resume.py` | — | langgraph==1.2.12、langgraph-checkpoint-sqlite==3.1.1（不需模型） |
| 30 | Trace 觀測與本地免費備援 | `30_local_trace.py` | — | — |
| 31 | 流程里程碑：沿用骨架整合 | `31_integration.py` | `data/search_cases.json`、`data/sop_corpus.json` | langgraph==1.2.12（不需模型；會載入同資料夾的 23、25、28、30 節程式） |
| 32 | MCP 角色與工具參數契約 | `32_mcp_contract.py` | — | mcp==2.3.0（不需模型；會載入同資料夾的 25 節程式） |
| 33 | MCP Client 生命週期與 Async | `33_mcp_lifecycle.py` | — | mcp==2.3.0（不需模型；會開 stdio 子程序，約 10 秒） |
| 34 | LLM 與 MCP 的完整工具回合 | `34_mcp_tool_round.py` | `data/34-tool-round-fixtures.json` | mcp==2.3.0（預設重播示範機錄製的模型輸出；`--live` 另需第 06 節的 torch、transformers 與 Qwen2.5-0.5B 權重） |
| 35 | 讀真實檔案的 MCP Server | `35_mcp_file_source.py` | — | mcp==2.3.0（不需模型；會開 stdio 子程序並建立符號連結，Windows 請用 WSL） |
| 36 | Streamable HTTP MCP 服務 | `36_mcp_http.py` | — | mcp==2.3.0、fastapi==0.142.2（不需模型；會在 127.0.0.1 啟動 Uvicorn 子程序；載入同資料夾的 35 節程式） |
| 37 | Multi-Agent、固定 Workflow 與 Reducer | `37_multi_agent.py` | — | langgraph==1.2.12（不需模型） |
| 38 | 單代理、固定流程、多代理：品質、延遲與成本比較 | `38_agent_compare.py` | `data/38-agent-traces.json`、`data/eval-set.json` | 重播只需標準函式庫；--live 需要 torch、transformers 與 Qwen2.5-0.5B 快取 |
| 39 | MCP 增量整合里程碑 | `39_mcp_increment.py` | — | langgraph==1.2.12、mcp==2.3.0（不需模型；載入同資料夾第 23、25、28、30、31 節程式；約 22 秒） |
| 40 | Prompt、RAG 與 SFT 的選擇 | `40_prompt_rag_sft.py` | `data/40-prompt-rag-traces.json`、`data/eval-set.json` | 重播只需標準函式庫（載入同資料夾第 38 節程式）；--live 需要 torch、transformers、sentence-transformers 與 Qwen、bge 快取 |
| 41 | SFT 資料與按來源拆分 | `41_sft_split.py` | `data/sop_corpus.json` | — |
| 42 | LoRA 低秩權重與訓練紀錄判讀 | `42_lora_math.py` | `data/model-candidates.json` | — |
| 43 | 微調前後評估與 Adapter 交付 | `43_adapter_eval.py` | `data/40-prompt-rag-traces.json`、`data/43-adapter-record.json`、`data/eval-set.json` | 重播只需標準函式庫（載入同資料夾第 38、40 節程式）；--train 需要 torch、transformers、peft==0.17.1 與 Qwen 快取（示範機約 2 分 41 秒、峰值約 3.9 GB） |
| 44 | Prefill、Decode 與 Token 成本 | `44_prefill_decode.py` | `data/44-latency-record.json`、`data/eval-set.json`、`data/provider-prices-checked.json` | 重播只需標準函式庫；--live 需要 torch、transformers、Qwen 快取與同資料夾第 06 節程式（示範機約 2 分 30 秒） |
| 45 | 量化與 CPU 實驗 | `45_quant_cpu.py` | `data/45-quant-record.json`、`data/eval-set.json` | 重播只需標準函式庫（載入同資料夾第 38、44 節程式）；--live 需要 torch、transformers 與 Qwen 快取（四個子程序，示範機約 6 分鐘） |
| 46 | KV Cache 與 Prefix Cache 的機制與局限 | `46_prefix_cache.py` | — | — |
| 47 | 模型路由、級聯與品質門檻 | `47_routing.py` | `data/eval-set.json` | — |
| 48 | 推測解碼與服務端推論 | `48_speculative.py` | `data/48-speculative-record.json` | 重播只需標準函式庫；--live 需要 torch、transformers、Qwen 快取與同資料夾第 38 節程式（示範機約 1 分 30 秒） |
| 49 | 全課增量驗收與評估報告 | `49_acceptance.py` | `data/43-adapter-record.json`、`data/44-latency-record.json`、`data/45-quant-record.json`、`data/eval-set.json`、`data/provider-prices-checked.json` | 必須在放講義（NN-*.md）的資料夾執行，llm-demo 單獨執行會說明並結束；標準函式庫（重跑第 31、39 節需要 langgraph、mcp；缺時標「未重跑」，可加 --quick）；需要同資料夾所有節次的程式與 data/ |

## 資料來源與限制

- `data/eval-set.json`：教材作者依標籤定義編寫的**合成**巡查通報與教學標註，不是真實通報；gold 由教師覆核。
- `data/model-candidates.json`：Hugging Face 公開 API 的**快照**（日期見檔內），授權與檔案大小會變。
- `data/tokenizer-fixtures.json`：教師在一台機器上的 tokenizer 實測紀錄，不是任何 API 的計費保證。
- `data/provider-snapshot.json`：LiteLLM 套件內建的能力與價目**快照**（離線產生，會過期）；`data/provider-prices-checked.json`：教師手動核對官方價目頁的紀錄（日期與網址見檔內）。
- `data/34-tool-round-fixtures.json`：教師在示範機上跑 Qwen2.5-0.5B 錄下的模型輸出（**fixtures**，環境與版本見檔內），不證明你的機器會產生同樣文字。
- `data/43-adapter-record.json`：教師在示範機 4-learn 上實際訓練 LoRA 的設定、loss、評估輸出與 adapter 雜湊（adapter 檔案本身不附），供第 43 節重播。
- `data/45-quant-record.json`：教師在示範機 4-learn 上量的四種精度（float32、bfloat16、兩種 int8）對照與激活值統計，供第 45 節重播。
- `data/48-speculative-record.json`：教師在示範機 4-learn 上跑的推測解碼（prompt lookup）紀錄，供第 48 節重播。
- `data/44-latency-record.json`：教師在示範機 4-learn 上量的每個 token 時間戳，供第 44 節重播；換機器數字會不同。
- `data/40-prompt-rag-traces.json`：教師在示範機 4-learn 上跑 Qwen2.5-0.5B 與 bge 錄下的輸出與近鄰，供第 40 節重播。
- `data/38-agent-traces.json`：教師在示範機 4-learn 上跑 Qwen2.5-0.5B 錄下的 traces（輸出、token 數、秒數）；**秒數是示範機當時的紀錄，不是你的機器效能**。
- `data/sop_corpus.json`、`data/search_cases.json`：與 MariaDB 課（[mariadb-demo](https://github.com/4-learn/mariadb-demo)）同一份合成 SOP 語料與標註，兩課共用。

所有資料都是合成或公開資訊，不含個資。
