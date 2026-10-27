# 4-learn LLM demos

依課程 mapping 對應講義內嵌 demo 產生。

## 必修課程（依 workspace course mapping）

| ID | 課程 | Demo | Dependencies |
| --- | --- | --- | --- |
| 01 | 系統責任與任務 Baseline | `01_role_baseline.py` | See lesson |
| 02 | Tokenizer 與上下文預算 | `02_token_budget.py` | See lesson |
| 03 | 自回歸生成、Logits 與 Inference | `03_inference_lab.py` | See lesson |
| 04 | Transformer、Attention、MoE 與推理模型 | `04_transformer_attention.py` | See lesson |
| 05 | Sampling 與生成實驗 | `E02_sampling.py` | See lesson |
| 06 | Hugging Face 模型選型與授權 | `E02_model_selection.py` | See lesson |
| 07 | API 呼叫：HTTP、JSON 與安全金鑰 | `E02_api_basics.py` | See lesson |
| 08 | Prompt 與 Context Engineering | `08_prompt_context.py` | See lesson |
| 09 | 本地推論載入與 CPU Smoke Test | `E02_cpu_smoke.py` | See lesson |
| 10 | 評測集、指標與可重現基準 | `E03_evaluation.py` | See lesson |
| 11 | 進階 LLM/RAG 評估、Judge 校準與回歸測試 | `11_llm_evaluation.py` | See lesson |
| 12 | 幻覺、提示注入與權限邊界 | `E02_boundaries.py` | See lesson |
| 13 | OWASP LLM Top 10 與 Guardrails | `13_llm_guardrails.py` | See lesson |
| 14 | 多模態輸入、證據與未知 | `E03_vision_unknown.py` | See lesson |
| 15 | Structured Output：語法、Schema 與語義 | `E07_structured_output.py` | See lesson |
| 16 | 原生 Function Calling 完整工具回合 | `E03_tool_round.py` | See lesson |
| 17 | 多供應商介面、LiteLLM 與 API 成本 | `E07_provider_contract.py` | See lesson |
| 18 | Embedding 原理與中文適配 | `E07_embedding_lab.py` | See lesson |
| 19 | 相似度、失敗案例與評測 | `E07_similarity_eval.py` | See lesson |
| 20 | 文件證據抽取（可選性 PDF 文字抽取；OCR／版面表格待補） | `tools/demo_document_parsing.py` | See lesson |
| 21 | Chunk 與來源資料模型 | `E07_chunking.py` | See lesson |
| 22 | 記憶體精確檢索 Baseline | `E08_exact_topk.py` | See lesson |
| 23 | MariaDB Vector 持久化銜接 | `E07_vector_store.py` | See lesson |
| 24 | Metadata 過濾與版本更新 | `E08_filter_update.py` | See lesson |
| 25 | 混合檢索與查詢改寫 | `25_hybrid_retrieval.py` | See lesson |
| 26 | Reranking 與 Retrieval Evaluation | `E07_rerank_eval.py` | See lesson |
| 27 | RAG 引用與查無答案 | `E08_rag_citations.py` | See lesson |
| 28 | Orchestration 與 LangChain LCEL | `51_lcel.py` | See lesson |
| 29 | 工具權限與副作用控制 | `E07_side_effects.py` | See lesson |
| 30 | LangGraph State 與 Reducer | `E08_state_reducer.py` | See lesson |
| 31 | 條件分支與 Unknown 測試 | `51_branch_unknown.py` | See lesson |
| 32 | 重試、退避與冪等 | `E07_retry_idempotency.py` | See lesson |
| 33 | HITL：真正拒絕與恢復 | `E08_hitl_resume.py` | See lesson |
| 34 | Trace 觀測與本地免費備援 | `51_local_trace.py` | See lesson |
| 35 | 流程里程碑：沿用骨架整合 | `E07_integration.py` | See lesson |
| 36 | MCP 角色與工具參數契約 | `E08_mcp_contract.py` | See lesson |
| 37 | MCP Client Lifecycle 與 Async | `51_mcp_lifecycle.py` | See lesson |
| 38 | LLM 與 MCP 完整工具回合 | `E07_mcp_tool_round.py` | See lesson |
| 39 | 真實檔案資料來源的 MCP Server | `E08_mcp_file_source.py` | See lesson |
| 40 | Streamable HTTP MCP 服務與部署 | `51_mcp_http.py` | See lesson |
| 41 | Agent 設計模式與記憶 | `41_agent_patterns.py` | See lesson |
| 42 | Multi-Agent、固定 Workflow 與 Reducer | `E07_multi_agent.py` | See lesson |
| 43 | 單／多 Agent 品質、延遲與成本比較 | `E08_agent_compare.py` | See lesson |
| 44 | MCP 增量整合里程碑 | `51_mcp_increment.py` | See lesson |
| 45 | Prompt、RAG 與 SFT 的選擇 | `E09_prompt_rag_sft.py` | See lesson |
| 46 | SFT 資料與按來源拆分 | `E07_sft_split.py` | See lesson |
| 47 | LoRA 低秩權重與教師訓練紀錄 | `E08_lora_math.py` | See lesson |
| 48 | 微調前後評估與 Adapter 交付 | `51_adapter_eval.py` | See lesson |
| 49 | Prefill、Decode 與 Token 成本 | `E09_prefill_decode.py` | See lesson |
| 50 | 模型路由、級聯與品質門檻 | `51_routing.py` | See lesson |
| 51 | 全課增量驗收與評估報告 | `51_acceptance.py` | See lesson |

## 選修課程（依 workspace course mapping）

| ID | 課程 | Demo | Dependencies |
| --- | --- | --- | --- |
| E01 | 可核對的圖像描述 | `E01_grounded_caption.py` | See lesson |
| E02 | VLM 文件摘要、來源與頁碼 | `E02_doc_citations.py` | See lesson |
| E03 | VLM 打標與 Metadata 品質 | `E03_vlm_tagging.py` | See lesson |
| E04 | 知識圖譜與 GraphRAG | `E04_graphrag.py` | See lesson |
| E05 | 語音 ASR／TTS 與語音代理 | `E05_voice_agents.py` | See lesson |
| E06 | Docker、Gradio/Streamlit 與產品化 | `E06_productization.py` | See lesson |
| E07 | 量化與 CPU 實驗 | `E07_quant_cpu.py` | See lesson |
| E08 | KV Cache 與 Prefix Cache | `E08_prefix_cache.py` | See lesson |
| E09 | 推測解碼與服務端推論 | `E09_speculative.py` | See lesson |

所有資料為合成或公開資訊，不含個資。課程數與標題以 workspace course mapping 為準；新增課程仍待審核及教學驗證；未納入之提案不包含 demo。
