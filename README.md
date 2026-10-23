# LLM 課程 Demo 與資料

4-learn LLM 50 節課程講義內嵌程式的可執行副本。**講義是唯一來源**；本 repo 由 swarm 的
`course-materials/llm/tools/build_llm_demo.py` 從講義產生，請不要直接在這裡修改程式。

講義目錄：<https://hackmd.io/@yillkid/HJl8JXTNkl>

## 使用方式

```bash
git clone https://github.com/4-learn/llm-demo.git
cd llm-demo
python3 04_sampling.py
```

需要 Python 3.10 以上。除下表註明的節次外，程式只用標準函式庫，不需要網路、模型或 API key。

**要安裝套件的節次（07、17 起），Python 請用 3.10～3.13（建議 3.10）**：釘選的 torch 2.8.0、numpy 2.2.6 沒有 3.14 的安裝檔，pip 會改裝別的版本。Linux 與 macOS 的安裝指令不同（macOS 不加 `--index-url`），見 [安裝基線](https://hackmd.io/@yillkid/HJl8JXTNkl) 的 INSTALL-BASELINE。

上課前先跑環境檢核（唯讀、不安裝、不連線）：`python3 check_env.py`；要上第 17 節再加 `--full`。

| 節 | 講義 | 程式 | 讀取的資料 | 額外需求 |
| --- | --- | --- | --- | --- |
| 01 | 不用 AI，能做到幾分？ | `01_role_baseline.py` | — | — |
| 02 | 一筆事件要花多少 token？ | `02_token_budget.py` | — | — |
| 03 | 模型怎麼一個字一個字寫出報告？ | `03_inference_lab.py` | — | — |
| 04 | 報告要穩定，還是要活潑？ | `04_sampling.py` | — | — |
| 05 | 該用哪個模型？ | `05_model_selection.py` | `data/model-candidates.json` | — |
| 06 | 用 API 請模型回答 | `06_api_basics.py` | — | — |
| 07 | 工地電腦跑得動嗎？ | `07_cpu_smoke.py` | — | torch、transformers、Qwen2.5-0.5B 權重約 942 MiB、可用記憶體約 3.3 GB（見第 07 節講義） |
| 08 | AI 到底幾分？ | `08_evaluation.py` | `data/eval-set.json` | — |
| 09 | AI 會不會闖禍？ | `09_boundaries.py` | — | — |
| 10 | 多模態輸入、證據與未知 | `10_vision_unknown.py` | `data/10-vision-record.json` | 重播只需標準函式庫（有 Pillow 會核對圖片雜湊）；--live 需要 torch、transformers==4.57.3、Pillow 與 SmolVLM-500M-Instruct 模型快取（約 3.3 GB 記憶體） |
| 11 | 可核對的圖像描述 | `11_grounded_caption.py` | `data/11-caption-record.json` | 重播只需標準函式庫（有 Pillow 會核對圖片雜湊）；--live 需求同 10_vision_unknown.py |
| 12 | 文件摘要、來源與頁碼 | `12_doc_citations.py` | `data/12-doc-record.json` | 重播只需標準函式庫（有 Pillow 會核對圖片雜湊）；--live 需求同 10_vision_unknown.py |
| 13 | Structured Output：語法、Schema 與語義 | `13_structured_output.py` | — | — |
| 14 | 原生 Function Calling 完整工具回合 | `14_tool_round.py` | — | — |
| 15 | 多供應商介面、LiteLLM 與 API 成本 | `15_provider_contract.py` | `data/provider-prices-checked.json`、`data/provider-snapshot.json` | — |
| 16 | VLM 打標與 Metadata 品質 | `16_vlm_tagging.py` | `data/16-tagging-record.json` | 需要同資料夾的 10_vision_unknown.py；重播只需標準函式庫；--live 需求同 10_vision_unknown.py |
| 17 | Embedding 概念 | `17_embedding_lab.py` | — | torch、sentence-transformers（版本見第 17 節講義） |
| 18 | 相似度、失敗案例與門檻評測 | `18_similarity_eval.py` | — | torch、sentence-transformers（版本見第 17 節講義） |
| 19 | Chunk 與來源資料模型 | `19_chunking.py` | `data/sop_corpus.json` | — |
| 20 | 記憶體精確檢索 Baseline | `20_exact_topk.py` | `data/search_cases.json`、`data/sop_corpus.json` | torch、sentence-transformers（版本見第 17 節講義） |
| 21 | MariaDB Vector 持久化銜接 | `21_vector_store.py` | `data/21-vectors.json`、`data/search_cases.json`、`data/sop_corpus.json` | fixtures 模式只需標準函式庫；MariaDB 模式需要 mariadb==1.1.14、~/mariadb-course-app.json 與教師預建的 llm_chunk_vectors；載入同資料夾第 20 節程式 |
| 22 | Metadata 過濾與版本更新 | `22_filter_update.py` | `data/22-update-vectors.json`、`data/search_cases.json`、`data/sop_corpus.json` | 同第 21 節（fixtures 或 MariaDB）；載入同資料夾第 20、21 節程式 |
| 23 | Reranking 與 Retrieval Evaluation | `23_rerank_eval.py` | `data/search_cases.json`、`data/sop_corpus.json` | — |
| 24 | RAG 引用與查無答案 | `24_rag_citations.py` | `data/search_cases.json`、`data/sop_corpus.json` | — |
| 25 | Orchestration 與 LangChain LCEL | `25_lcel.py` | `data/search_cases.json`、`data/sop_corpus.json` | langchain-core==1.6.6（隨 langgraph==1.2.12 安裝）；載入同資料夾第 24 節程式；不需模型 |
| 26 | 工具權限與副作用控制 | `26_side_effects.py` | — | — |
| 27 | LangGraph State 與 Reducer | `27_state_reducer.py` | — | langgraph==1.2.12（不需模型） |
| 28 | 條件分支與 Unknown 測試 | `28_branch_unknown.py` | — | langgraph==1.2.12（不需模型） |
| 29 | 重試、退避與冪等 | `29_retry_idempotency.py` | — | — |
| 30 | HITL：真正拒絕與恢復 | `30_hitl_resume.py` | — | langgraph==1.2.12、langgraph-checkpoint-sqlite==3.1.1（不需模型） |
| 31 | Trace 觀測與本地免費備援 | `31_local_trace.py` | — | — |
| 32 | 流程里程碑：沿用骨架整合 | `32_integration.py` | `data/search_cases.json`、`data/sop_corpus.json` | langgraph==1.2.12（不需模型；會載入同資料夾的 24、26、29、31 節程式） |
| 33 | MCP 角色與工具參數契約 | `33_mcp_contract.py` | — | mcp==2.3.0（不需模型；會載入同資料夾的 26 節程式） |
| 34 | MCP Client 生命週期與 Async | `34_mcp_lifecycle.py` | — | mcp==2.3.0（不需模型；會開 stdio 子程序，約 10 秒） |
| 35 | LLM 與 MCP 的完整工具回合 | `35_mcp_tool_round.py` | `data/35-tool-round-fixtures.json` | mcp==2.3.0（預設重播示範機錄製的模型輸出；`--live` 另需第 07 節的 torch、transformers 與 Qwen2.5-0.5B 權重） |
| 36 | 讀真實檔案的 MCP Server | `36_mcp_file_source.py` | — | mcp==2.3.0（不需模型；會開 stdio 子程序並建立符號連結，Windows 請用 WSL） |
| 37 | Streamable HTTP MCP 服務 | `37_mcp_http.py` | — | mcp==2.3.0、fastapi==0.142.2（不需模型；會在 127.0.0.1 啟動 Uvicorn 子程序；載入同資料夾的 36 節程式） |
| 38 | Multi-Agent、固定 Workflow 與 Reducer | `38_multi_agent.py` | — | langgraph==1.2.12（不需模型） |
| 39 | 單代理、固定流程、多代理：品質、延遲與成本比較 | `39_agent_compare.py` | `data/39-agent-traces.json`、`data/eval-set.json` | 重播只需標準函式庫；--live 需要 torch、transformers 與 Qwen2.5-0.5B 快取 |
| 40 | MCP 增量整合里程碑 | `40_mcp_increment.py` | — | langgraph==1.2.12、mcp==2.3.0（不需模型；載入同資料夾第 24、26、29、31、32 節程式；約 22 秒） |
| 41 | Prompt、RAG 與 SFT 的選擇 | `41_prompt_rag_sft.py` | `data/41-prompt-rag-traces.json`、`data/eval-set.json` | 重播只需標準函式庫（載入同資料夾第 39 節程式）；--live 需要 torch、transformers、sentence-transformers 與 Qwen、bge 快取 |
| 42 | SFT 資料與按來源拆分 | `42_sft_split.py` | `data/sop_corpus.json` | — |
| 43 | LoRA 低秩權重與訓練紀錄判讀 | `43_lora_math.py` | `data/model-candidates.json` | — |
| 44 | 微調前後評估與 Adapter 交付 | `44_adapter_eval.py` | `data/41-prompt-rag-traces.json`、`data/44-adapter-record.json`、`data/eval-set.json` | 重播只需標準函式庫（載入同資料夾第 39、41 節程式）；--train 需要 torch、transformers、peft==0.17.1 與 Qwen 快取（示範機約 2 分 41 秒、峰值約 3.9 GB） |
| 45 | Prefill、Decode 與 Token 成本 | `45_prefill_decode.py` | `data/45-latency-record.json`、`data/eval-set.json`、`data/provider-prices-checked.json` | 重播只需標準函式庫；--live 需要 torch、transformers、Qwen 快取與同資料夾第 07 節程式（示範機約 2 分 30 秒） |
| 46 | 量化與 CPU 實驗 | `46_quant_cpu.py` | `data/46-quant-record.json`、`data/eval-set.json` | 重播只需標準函式庫（載入同資料夾第 39、45 節程式）；--live 需要 torch、transformers 與 Qwen 快取（四個子程序，示範機約 6 分鐘） |
| 47 | KV Cache 與 Prefix Cache 的機制與局限 | `47_prefix_cache.py` | — | — |
| 48 | 模型路由、級聯與品質門檻 | `48_routing.py` | `data/eval-set.json` | — |
| 49 | 推測解碼與服務端推論 | `49_speculative.py` | `data/49-speculative-record.json` | 重播只需標準函式庫；--live 需要 torch、transformers、Qwen 快取與同資料夾第 39 節程式（示範機約 1 分 30 秒） |
| 50 | 全課增量驗收與評估報告 | `50_acceptance.py` | `data/44-adapter-record.json`、`data/45-latency-record.json`、`data/46-quant-record.json`、`data/eval-set.json`、`data/provider-prices-checked.json` | 必須在放講義（NN-*.md）的資料夾執行，llm-demo 單獨執行會說明並結束；標準函式庫（重跑第 32、40 節需要 langgraph、mcp；缺時標「未重跑」，可加 --quick）；需要同資料夾所有節次的程式與 data/ |

## 資料來源與限制

- `data/eval-set.json`：教材作者依標籤定義編寫的**合成**巡查通報與教學標註，不是真實通報；gold 由教師覆核。
- `data/model-candidates.json`：Hugging Face 公開 API 的**快照**（日期見檔內），授權與檔案大小會變。
- `data/tokenizer-fixtures.json`：教師在一台機器上的 tokenizer 實測紀錄，不是任何 API 的計費保證。
- `data/provider-snapshot.json`：LiteLLM 套件內建的能力與價目**快照**（離線產生，會過期）；`data/provider-prices-checked.json`：教師手動核對官方價目頁的紀錄（日期與網址見檔內）。
- `data/35-tool-round-fixtures.json`：教師在示範機上跑 Qwen2.5-0.5B 錄下的模型輸出（**fixtures**，環境與版本見檔內），不證明你的機器會產生同樣文字。
- `data/44-adapter-record.json`：教師在示範機 4-learn 上實際訓練 LoRA 的設定、loss、評估輸出與 adapter 雜湊（adapter 檔案本身不附），供第 44 節重播。
- `data/46-quant-record.json`：教師在示範機 4-learn 上量的四種精度（float32、bfloat16、兩種 int8）對照與激活值統計，供第 46 節重播。
- `data/49-speculative-record.json`：教師在示範機 4-learn 上跑的推測解碼（prompt lookup）紀錄，供第 49 節重播。
- `data/45-latency-record.json`：教師在示範機 4-learn 上量的每個 token 時間戳，供第 45 節重播；換機器數字會不同。
- `data/41-prompt-rag-traces.json`：教師在示範機 4-learn 上跑 Qwen2.5-0.5B 與 bge 錄下的輸出與近鄰，供第 41 節重播。
- `data/39-agent-traces.json`：教師在示範機 4-learn 上跑 Qwen2.5-0.5B 錄下的 traces（輸出、token 數、秒數）；**秒數是示範機當時的紀錄，不是你的機器效能**。
- `data/sop_corpus.json`、`data/search_cases.json`：與 MariaDB 課（[mariadb-demo](https://github.com/4-learn/mariadb-demo)）同一份合成 SOP 語料與標註，兩課共用。

所有資料都是合成或公開資訊，不含個資。
