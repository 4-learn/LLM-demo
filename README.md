# 4-learn LLM demos

依課程 mapping 對應講義內嵌 demo 產生。

## Required lessons

| ID | Lesson | Demo | Dependencies |
| --- | --- | --- | --- |
| 01 | 01｜不用 AI，能做到幾分？ | `01_role_baseline.py` | — |
| 02 | 02｜一筆事件要花多少 token？ | `02_token_budget.py` | — |
| 03 | 03｜模型怎麼一個字一個字寫出報告？ | `03_inference_lab.py` | — |
| 05 | 04｜報告要穩定，還是要活潑？ | `E02_sampling.py` | — |
| 06 | 05｜該用哪個模型？ | `E02_model_selection.py` | 快照 JSON 日期由資料檔決定；凍結輸出需同步日期 |
| 07 | 06｜用 API 請模型回答 | `E02_api_basics.py` | — |
| 09 | 07｜工地電腦跑得動嗎？ | `E02_cpu_smoke.py` | torch、transformers 與 Qwen2.5-0.5B 權重 |
| 10 | 08｜AI 到底幾分？ | `E03_evaluation.py` | 標準函式庫；--live 需相關模型 |
| 12 | 09｜AI 會不會闖禍？ | `E02_boundaries.py` | — |
| 14 | 多模態輸入、證據與未知 | `E03_vision_unknown.py` | Pillow 可選；缺 Pillow 時凍結輸出不同 |
| 15 | Structured Output：語法、Schema 與語義 | `E07_structured_output.py` | — |
| 16 | 原生 Function Calling 完整工具回合 | `E03_tool_round.py` | — |
| 17 | 多供應商介面、LiteLLM 與 API 成本 | `E07_provider_contract.py` | 資料快照日期含舊 lesson code，凍結輸出需同步修正 |
| 18 | Embedding 概念 | `E07_embedding_lab.py` | — |
| 19 | 相似度、失敗案例與門檻評測 | `E07_similarity_eval.py` | torch、sentence-transformers 與模型快取 |
| 21 | Chunk 與來源資料模型 | `E07_chunking.py` | — |
| 22 | 記憶體精確檢索 Baseline | `E08_exact_topk.py` | torch、sentence-transformers 與模型快取 |
| 23 | MariaDB Vector 持久化銜接 | `E07_vector_store.py` | MariaDB 模式需 mariadb==1.1.14 與設定檔 |
| 24 | Metadata 過濾與版本更新 | `E08_filter_update.py` | MariaDB 模式需 mariadb==1.1.14 與設定檔 |
| 26 | Reranking 與 Retrieval Evaluation | `E07_rerank_eval.py` | — |
| 27 | RAG 引用與查無答案 | `E08_rag_citations.py` | — |
| 28 | Orchestration 與 LangChain LCEL | `51_lcel.py` | langchain-core==1.6.6、langgraph==1.2.12 |
| 29 | 工具權限與副作用控制 | `E07_side_effects.py` | — |
| 30 | LangGraph State 與 Reducer | `E08_state_reducer.py` | langgraph==1.2.12 |
| 31 | 條件分支與 Unknown 測試 | `51_branch_unknown.py` | langgraph==1.2.12 |
| 32 | 重試、退避與冪等 | `E07_retry_idempotency.py` | — |
| 33 | HITL：真正拒絕與恢復 | `E08_hitl_resume.py` | langgraph==1.2.12、langgraph-checkpoint-sqlite==3.1.1 |
| 34 | Trace 觀測與本地免費備援 | `51_local_trace.py` | OpenTelemetry 版本依環境，凍結輸出需同步修正 |
| 35 | 流程里程碑：沿用骨架整合 | `E07_integration.py` | langgraph==1.2.12 |
| 36 | MCP 角色與工具參數契約 | `E08_mcp_contract.py` | mcp==2.3.0、pydantic |
| 37 | MCP Client 生命週期與 Async | `51_mcp_lifecycle.py` | mcp==2.3.0 |
| 38 | LLM 與 MCP 的完整工具回合 | `E07_mcp_tool_round.py` | mcp==2.3.0 |
| 39 | 讀真實檔案的 MCP Server | `E08_mcp_file_source.py` | mcp==2.3.0 |
| 40 | Streamable HTTP MCP 服務 | `51_mcp_http.py` | mcp==2.3.0、httpx |
| 42 | Multi-Agent、固定 Workflow 與 Reducer | `E07_multi_agent.py` | langgraph==1.2.12 |
| 43 | 單代理、固定流程、多代理：品質、延遲與成本比較 | `E08_agent_compare.py` | — |
| 44 | MCP 增量整合里程碑 | `51_mcp_increment.py` | mcp==2.3.0、langgraph==1.2.12 |
| 45 | Prompt、RAG 與 SFT 的選擇 | `E09_prompt_rag_sft.py` | — |
| 46 | SFT 資料與按來源拆分 | `E07_sft_split.py` | 語料資料依映射課程版本，凍結輸出需同步修正 |
| 47 | LoRA 低秩權重與訓練紀錄判讀 | `E08_lora_math.py` | 依賴 model-candidates.json；數值輸出含版本快照 |
| 48 | 微調前後評估與 Adapter 交付 | `51_adapter_eval.py` | — |
| 49 | Prefill、Decode 與 Token 成本 | `E09_prefill_decode.py` | — |
| 50 | 模型路由、級聯與品質門檻 | `51_routing.py` | — |
| 04 | 04｜Transformer、Attention、MoE 與推理模型 | `04_transformer_attention.py` | — |
| 08 | 08｜Prompt 與 Context Engineering | `08_prompt_context.py` | — |
| 11 | 11｜檢查模型回答：正確率 | `11_llm_evaluation.py` | — |
| 13 | 13｜資料不等於指令 | `13_llm_guardrails.py` | — |
| 20 | 20｜從文字找出工單編號 | `20_document_parsing.py` | — |
| 25 | 25｜用關鍵字找文件 | `25_hybrid_retrieval.py` | — |
| 41 | 41｜固定流程與 Agent | `41_agent_patterns.py` | — |

## Electives

| ID | Lesson | Demo | Dependencies |
| --- | --- | --- | --- |
| E01 | 可核對的圖像描述 | `E01_grounded_caption.py` | 重播標準函式庫；--live 需 Pillow、torch、transformers |
| E02 | 文件摘要、來源與頁碼 | `E02_doc_citations.py` | 重播標準函式庫；--live 需 Pillow、torch、transformers |
| E03 | VLM 打標與 Metadata 品質 | `E03_vlm_tagging.py` | 重播標準函式庫；--live 需 Pillow、torch、transformers |
| E07 | 量化與 CPU 實驗 | `E07_quant_cpu.py` | 重播標準函式庫；--live 需 torch、transformers |
| E08 | KV Cache 與 Prefix Cache 的機制與局限 | `E08_prefix_cache.py` | 輸出計量依環境，版本會不同，不作逐字比對 |
| E09 | 推測解碼與服務端推論 | `E09_speculative.py` | 重播需 speculative-record.json；--live 需模型快取 |
| E04 | E04｜用關係找到答案 | `E04_graphrag.py` | — |
| E05 | E05｜語音指令要聽清楚 | `E05_voice_agents.py` | — |
| E06 | E06｜先檢查使用者輸入 | `E06_productization.py` | — |

## Acceptance

| 51 | 最終整合驗收 | `51_acceptance.py` | — |

所有資料為合成或公開資訊，不含個資。尚未撰寫的提案不包含 demo。
