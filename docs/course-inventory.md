# 60 課呼叫、圖示與長程式盤點

> 盤點來源：課程 mapping（60 節）與本 repo main 基線 abbb7ea；此 inventory 由 `tools/build_epic12_inventory.py` 產生。Demo 檔依講義 `<!-- demo: ... -->` 錨點核對；行數為錨點所指 demo 最大行數。

| 課號 | 課名 | 講義 | Demo 檔案（依錨點） | 行數(最大) | Mermaid flowchart 數 | 執行型態/依賴線索 | Build SDK 結論 |
|---|---|---|---|---:|---:|---|---|
| 01 | 系統責任與任務 Baseline | `01-model-role.md` | `01_role_baseline.py` | 49 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 依課程目的不需/不宜改，或非模型呼叫 |
| 02 | Tokenizer 與上下文預算 | `02-tokenizer.md` | `02_token_budget.py` | 119 | 6 | replay/fixture; HF/local inference | 依課程目的不需/不宜改，或非模型呼叫 |
| 03 | 自回歸生成、Logits 與 Inference | `03-inference.md` | `03_inference_lab.py` | 114 | 5 | 未見模型呼叫（純流程/演算法或缺 demo） | 不改呼叫方式；可補本機 vs API 對照 |
| 04 | Transformer、Attention、MoE 與推理模型 | `04-transformer-attention.md` | `04_transformer_attention.py` | 26 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 依課程目的不需/不宜改，或非模型呼叫 |
| 05 | Sampling 與生成實驗 | `05-sampling.md` | `E02_sampling.py` | 165 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 部分可替換；保留本地/工具/驗證流程 |
| 06 | Hugging Face 模型選型與授權 | `06-model-selection.md` | `E02_model_selection.py` | 179 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 依課程目的不需/不宜改，或非模型呼叫 |
| 07 | API 呼叫：HTTP、JSON 與安全金鑰 | `07-api-basics.md` | `E02_api_basics.py` | 86 | 1 | replay/fixture; HTTP/API live | 可完整替換文字 LLM live 呼叫 |
| 08 | Prompt 與 Context Engineering | `08-prompt-context.md` | `08_prompt_context.py` | 26 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 依課程目的不需/不宜改，或非模型呼叫 |
| 09 | 本地推論載入與 CPU Smoke Test | `09-local-smoke.md` | `E02_cpu_smoke.py` | 193 | 1 | HF/local inference | 不改呼叫方式；可補本機 vs API 對照 |
| 10 | 評測集、指標與可重現基準 | `10-evaluation.md` | `E03_evaluation.py` | 156 | 1 | HF/local inference | 部分可替換；保留本地/工具/驗證流程 |
| 11 | 進階 LLM/RAG 評估、Judge 校準與回歸測試 | `11-llm-evaluation.md` | `11_llm_evaluation.py` | 12 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 依課程目的不需/不宜改，或非模型呼叫 |
| 12 | 幻覺、提示注入與權限邊界 | `12-boundaries.md` | `E02_boundaries.py` | 142 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 部分可替換；保留本地/工具/驗證流程 |
| 13 | OWASP LLM Top 10 與 Guardrails | `13-llm-guardrails.md` | `13_llm_guardrails.py` | 13 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 依課程目的不需/不宜改，或非模型呼叫 |
| 14 | 多模態輸入、證據與未知 | `14-vision-unknown.md` | `E03_vision_unknown.py` | 226 | 5 | replay/fixture; HF/local inference | 部分可替換；保留本地/工具/驗證流程 |
| E01 | 可核對的圖像描述 | `E01-grounded-caption.md` | `E01_grounded_caption.py` | 252 | 5 | replay/fixture; HF/local inference | 部分可替換；保留本地/工具/驗證流程 |
| E02 | VLM 文件摘要、來源與頁碼 | `E02-doc-citations.md` | `E02_doc_citations.py` | 233 | 5 | replay/fixture; HF/local inference | 部分可替換；保留本地/工具/驗證流程 |
| 15 | Structured Output：語法、Schema 與語義 | `15-structured-output.md` | `E07_structured_output.py` | 176 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 可完整替換文字 LLM live 呼叫 |
| 16 | 原生 Function Calling 完整工具回合 | `16-tool-round.md` | `E03_tool_round.py` | 281 | 5 | replay/fixture; HTTP/API live; tool/MCP | 可完整替換文字 LLM live 呼叫 |
| 17 | 多供應商介面、LiteLLM 與 API 成本 | `17-providers-cost.md` | `E07_provider_contract.py` | 175 | 1 | replay/fixture; HTTP/API live | 可完整替換文字 LLM live 呼叫 |
| E03 | VLM 打標與 Metadata 品質 | `E03-vlm-tagging.md` | `E03_vlm_tagging.py` | 269 | 5 | replay/fixture; HF/local inference | 部分可替換；保留本地/工具/驗證流程 |
| 18 | Embedding 原理與中文適配 | `18-embedding.md` | `E07_embedding_lab.py` | 67 | 1 | HF/local inference | 依課程目的不需/不宜改，或非模型呼叫 |
| 19 | 相似度、失敗案例與評測 | `19-similarity-eval.md` | `E07_similarity_eval.py` | 190 | 1 | HF/local inference | 依課程目的不需/不宜改，或非模型呼叫 |
| 20 | 文件證據抽取（可選性 PDF 文字抽取；OCR／版面表格待補） | `20-document-parsing.md` | `demo_document_parsing.py` | 37 | 1 | replay/fixture | 依課程目的不需/不宜改，或非模型呼叫 |
| 21 | Chunk 與來源資料模型 | `21-chunking.md` | `E07_chunking.py` | 175 | 1 | replay/fixture | 依課程目的不需/不宜改，或非模型呼叫 |
| 22 | 記憶體精確檢索 Baseline | `22-exact-baseline.md` | `E08_exact_topk.py` | 186 | 1 | HF/local inference | 依課程目的不需/不宜改，或非模型呼叫 |
| 23 | MariaDB Vector 持久化銜接 | `23-vector-store.md` | `E07_vector_store.py` | 412 | 5 | replay/fixture | 依課程目的不需/不宜改，或非模型呼叫 |
| 24 | Metadata 過濾與版本更新 | `24-filter-update.md` | `E08_filter_update.py` | 158 | 1 | replay/fixture | 依課程目的不需/不宜改，或非模型呼叫 |
| 25 | 混合檢索與查詢改寫 | `25-hybrid-retrieval.md` | `25_hybrid_retrieval.py` | 64 | 1 | replay/fixture | 部分可替換；保留本地/工具/驗證流程 |
| 26 | Reranking 與 Retrieval Evaluation | `26-rerank-eval.md` | `E07_rerank_eval.py` | 191 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 依課程目的不需/不宜改，或非模型呼叫 |
| 27 | RAG 引用與查無答案 | `27-rag-citations.md` | `E08_rag_citations.py` | 209 | 5 | replay/fixture; HF/local inference | 可完整替換文字 LLM live 呼叫 |
| 28 | Orchestration 與 LangChain LCEL | `28-lcel.md` | `51_lcel.py` | 217 | 5 | HF/local inference | 可完整替換文字 LLM live 呼叫 |
| 29 | 工具權限與副作用控制 | `29-side-effects.md` | `E07_side_effects.py` | 125 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 部分可替換；保留本地/工具/驗證流程 |
| 30 | LangGraph State 與 Reducer | `30-state-reducer.md` | `E08_state_reducer.py` | 122 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 依課程目的不需/不宜改，或非模型呼叫 |
| 31 | 條件分支與 Unknown 測試 | `31-branch-unknown.md` | `51_branch_unknown.py` | 162 | 1 | replay/fixture | 可完整替換文字 LLM live 呼叫 |
| 32 | 重試、退避與冪等 | `32-retry-idempotency.md` | `E07_retry_idempotency.py` | 124 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 依課程目的不需/不宜改，或非模型呼叫 |
| 33 | HITL：真正拒絕與恢復 | `33-hitl-resume.md` | `E08_hitl_resume.py` | 220 | 5 | 未見模型呼叫（純流程/演算法或缺 demo） | 可完整替換文字 LLM live 呼叫 |
| 34 | Trace 觀測與本地免費備援 | `34-local-trace.md` | `51_local_trace.py` | 197 | 1 | tool/MCP | 可完整替換文字 LLM live 呼叫 |
| 35 | 流程里程碑：沿用骨架整合 | `35-integration-milestone.md` | `E07_integration.py` | 211 | 5 | tool/MCP | 可完整替換文字 LLM live 呼叫 |
| 36 | MCP 角色與工具參數契約 | `36-mcp-contract.md` | `E08_mcp_contract.py` | 191 | 1 | tool/MCP | 依課程目的不需/不宜改，或非模型呼叫 |
| 37 | MCP Client Lifecycle 與 Async | `37-mcp-lifecycle.md` | `51_mcp_lifecycle.py` | 158 | 1 | tool/MCP | 依課程目的不需/不宜改，或非模型呼叫 |
| 38 | LLM 與 MCP 完整工具回合 | `38-mcp-tool-round.md` | `E07_mcp_tool_round.py` | 300 | 5 | replay/fixture; HF/local inference; tool/MCP | 部分可替換；保留本地/工具/驗證流程 |
| 39 | 真實檔案資料來源的 MCP Server | `39-mcp-file-source.md` | `E08_mcp_file_source.py` | 212 | 5 | tool/MCP | 依課程目的不需/不宜改，或非模型呼叫 |
| 40 | Streamable HTTP MCP 服務與部署 | `40-mcp-http.md` | `51_mcp_http.py` | 246 | 5 | tool/MCP | 依課程目的不需/不宜改，或非模型呼叫 |
| 41 | Agent 設計模式與記憶 | `41-agent-patterns.md` | `41_agent_patterns.py` | 40 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 可完整替換文字 LLM live 呼叫 |
| 42 | Multi-Agent、固定 Workflow 與 Reducer | `42-multi-agent.md` | `E07_multi_agent.py` | 211 | 7 | 未見模型呼叫（純流程/演算法或缺 demo） | 可完整替換文字 LLM live 呼叫 |
| 43 | 單／多 Agent 品質、延遲與成本比較 | `43-agent-compare.md` | `E08_agent_compare.py` | 219 | 5 | replay/fixture; HF/local inference | 部分可替換；保留本地/工具/驗證流程 |
| 44 | MCP 增量整合里程碑 | `44-mcp-increment.md` | `51_mcp_increment.py` | 214 | 5 | tool/MCP | 依課程目的不需/不宜改，或非模型呼叫 |
| 45 | Prompt、RAG 與 SFT 的選擇 | `45-prompt-rag-sft.md` | `E09_prompt_rag_sft.py` | 210 | 5 | replay/fixture; HF/local inference | 部分可替換；保留本地/工具/驗證流程 |
| 46 | SFT 資料與按來源拆分 | `46-sft-split.md` | `E07_sft_split.py` | 209 | 5 | 未見模型呼叫（純流程/演算法或缺 demo） | 依課程目的不需/不宜改，或非模型呼叫 |
| 47 | LoRA 低秩權重與教師訓練紀錄 | `47-lora.md` | `E08_lora_math.py` | 144 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 依課程目的不需/不宜改，或非模型呼叫 |
| 48 | 微調前後評估與 Adapter 交付 | `48-adapter-eval.md` | `51_adapter_eval.py` | 220 | 5 | replay/fixture; HF/local inference | 不改呼叫方式；可補本機 vs API 對照 |
| 49 | Prefill、Decode 與 Token 成本 | `49-prefill-decode.md` | `E09_prefill_decode.py` | 181 | 1 | replay/fixture; HF/local inference | 不改呼叫方式；可補本機 vs API 對照 |
| E07 | 量化與 CPU 實驗 | `E07-quant-cpu.md` | `E07_quant_cpu.py` | 218 | 5 | replay/fixture; HF/local inference | 不改呼叫方式；可補本機 vs API 對照 |
| E08 | KV Cache 與 Prefix Cache | `E08-prefix-cache.md` | `E08_prefix_cache.py` | 111 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 依課程目的不需/不宜改，或非模型呼叫 |
| 50 | 模型路由、級聯與品質門檻 | `50-routing.md` | `51_routing.py` | 143 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 可完整替換文字 LLM live 呼叫 |
| E09 | 推測解碼與服務端推論 | `E09-speculative.md` | `E09_speculative.py` | 245 | 5 | replay/fixture; HF/local inference | 不改呼叫方式；可補本機 vs API 對照 |
| 51 | 全課增量驗收與評估報告 | `51-acceptance.md` | `51_acceptance.py` | 167 | 1 | replay/fixture; tool/MCP | 部分可替換；保留本地/工具/驗證流程 |
| E04 | 知識圖譜與 GraphRAG | `E04-graphrag.md` | `E04_graphrag.py` | 109 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 部分可替換；保留本地/工具/驗證流程 |
| E05 | 語音 ASR／TTS 與語音代理 | `E05-voice-agents.md` | `E05_voice_agents.py` | 63 | 1 | replay/fixture | 部分可替換；保留本地/工具/驗證流程 |
| E06 | Docker、Gradio/Streamlit 與產品化 | `E06-productization.md` | `E06_productization.py` | 47 | 1 | 未見模型呼叫（純流程/演算法或缺 demo） | 依課程目的不需/不宜改，或非模型呼叫 |

## 覆蓋核對

- Mapping 課數：60（必修 51、選修 9）。
- 指定長 demo 集合：14, 16, 23, 27, 28, 33, 35, 38, 39, 40, 42, 43, 44, 45, 46, 48, E01, E02, E03, E07, E09；程式行數與講義需逐項人工核對。
- 講義缺檔：無（課號 51 無對應講義是既有 mapping/README 特例，請補課或明確標記為彙總驗收）
- 呼叫型態以源碼掃描關鍵詞為提示，逐課決策需人工複核；fixture/replay 不等同執行 live 模型。
