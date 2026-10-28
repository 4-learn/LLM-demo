# LLM 課程重組索引

> 課程共 51 節必修與 9 節選修。十篇新增教材已修訂並納入本地 smoke tests，面向具基本 Python 能力的學員，每課聚焦一項核心概念。第 20 課示範純文字欄位抽取，不涵蓋 PDF parser、OCR 或版面表格。HackMD 已有新版課程頁面；索引連結採用公開短網址，頁面未經確認不得刪除。

## 必修主線（01–51）

| 新節 | 來源 | 課題 | 狀態 |
|---:|---:|---|---|
|01|舊01|系統責任與任務 Baseline|既有|
|02|舊02|Tokenizer 與上下文預算|既有|
|03|舊03|自回歸生成、Logits 與 Inference|既有|
|04|新增|Attention｜用權重混合資訊|修訂完成|
|05|舊04|Sampling 與生成實驗|既有|
|06|舊05|Hugging Face 模型選型與授權|既有|
|07|舊06|API 呼叫：HTTP、JSON 與安全金鑰|既有|
|08|新增|Prompt Engineering｜把任務和資料說清楚|修訂完成|
|09|舊07|本地推論載入與 CPU Smoke Test|既有|
|10|舊08|評測集、指標與可重現基準|既有|
|11|新增|LLM 評估｜用正確率檢查回答|修訂完成|
|12|舊09|幻覺、提示注入與權限邊界|既有|
|13|新增|Guardrails｜把資料和指令分開|修訂完成|
|14|舊10|多模態輸入、證據與未知|既有|
|15|舊13|Structured Output：語法、Schema 與語義|既有|
|16|舊14|原生 Function Calling 完整工具回合|既有|
|17|舊15|多供應商介面、LiteLLM 與 API 成本|既有|
|18|舊17|Embedding 原理與中文適配|既有|
|19|舊18|相似度、失敗案例與評測|既有|
|20|新增|文件處理｜從文字找出工單編號|修訂完成；只做純文字欄位抽取|
|21|舊19|Chunk 與來源資料模型|既有|
|22|舊20|記憶體精確檢索 Baseline|既有|
|23|舊21|MariaDB Vector 持久化銜接|既有|
|24|舊22|Metadata 過濾與版本更新|既有|
|25|新增|混合檢索｜用關鍵字找候選文件|修訂完成；本課是關鍵字基線|
|26|舊23|Reranking 與 Retrieval Evaluation|既有|
|27|舊24|RAG 引用與查無答案|既有|
|28|舊25|Orchestration 與 LangChain LCEL|既有|
|29|舊26|工具權限與副作用控制|既有|
|30|舊27|LangGraph State 與 Reducer|既有|
|31|舊28|條件分支與 Unknown 測試|既有|
|32|舊29|重試、退避與冪等|既有|
|33|舊30|HITL：真正拒絕與恢復|既有|
|34|舊31|Trace 觀測與本地免費備援|既有|
|35|舊32|流程里程碑：沿用骨架整合|既有|
|36|舊33|MCP 角色與工具參數契約|既有|
|37|舊34|MCP Client Lifecycle 與 Async|既有|
|38|舊35|LLM 與 MCP 完整工具回合|既有|
|39|舊36|真實檔案資料來源的 MCP Server|既有|
|40|舊37|Streamable HTTP MCP 服務與部署|既有|
|41|新增|Agent 設計｜依結果選擇下一步|修訂完成；程式示意採模型決策的模擬|
|42|舊38|Multi-Agent、固定 Workflow 與 Reducer|既有|
|43|舊39|單／多 Agent 品質、延遲與成本比較|既有|
|44|舊40|MCP 增量整合里程碑|既有|
|45|舊41|Prompt、RAG 與 SFT 的選擇|既有|
|46|舊42|SFT 資料與按來源拆分|既有|
|47|舊43|LoRA 低秩權重與教師訓練紀錄|既有|
|48|舊44|微調前後評估與 Adapter 交付|既有|
|49|舊45|Prefill、Decode 與 Token 成本|既有|
|50|舊48|模型路由、級聯與品質門檻|既有|
|51|舊50|全課增量驗收與評估報告|既有|

## 獨立選修（E01–E09）

| 編號 | 來源 | 課題 | 狀態 |
|---|---:|---|---|
|E01|舊11|可核對的圖像描述|既有|
|E02|舊12|VLM 文件摘要、來源與頁碼|既有；只處理合成文件影像，不替代新20/27|
|E03|舊16|VLM 打標與 Metadata 品質|既有|
|E04|新增|知識圖譜｜沿著兩條關係查位置|修訂完成；最小兩跳查表示範|
|E05|新增|語音代理｜先確認聽到的指令|修訂完成；文字模擬，非 ASR/TTS 實測|
|E06|新增|產品化｜先檢查文字輸入|修訂完成；輸入檢查示範，非部署實測|
|E07|舊46|量化與 CPU 實驗|既有|
|E08|舊47|KV Cache 與 Prefix Cache|既有|
|E09|舊49|推測解碼與服務端推論|既有|

## HackMD 頁面與識別碼

以下 internal note ID 是 API 識別碼，不是應放進公開連結的 slug。公開 canonical URL 使用既有短 slug；短 slug/internal ID 指向同一 note，不構成兩篇重複頁面。

| 課程 | 公開 canonical URL | Internal note ID |
|---|---|---|
|04|https://hackmd.io/@yillkid/SycO0ykpzx|`SRvuzLnZSmqFUtlEUCAaEQ`|
|08|https://hackmd.io/@yillkid/Hk1F0kkazx|`9v8FmoYmRXCHobMJ76j9MA`|
|11|https://hackmd.io/@yillkid/By4Y0ykpzl|`XpXBeKh6QRyBceVWIqg95A`|
|13|https://hackmd.io/@yillkid/H1Ft0yJ6ze|`oB9Rgn8wQySGX5ImRFGxOQ`|
|20|https://hackmd.io/@yillkid/SyatC1JpGg|`vmcgGtS-RV6yE9whSyjmVg`|
|25|https://hackmd.io/@yillkid/SJMqC116Ge|`1l0fBGTGRq-bBC-S7bAqRQ`|
|41|https://hackmd.io/@yillkid/HJwqAJypGe|`G4evPFdxQvavjVKYJL2Dgw`|
|E04|https://hackmd.io/@yillkid/Hyn9AJJ6Me|`8hWr2MWfSu-_vrcF--NBdw`|
|E05|https://hackmd.io/@yillkid/H1ZiA1JaGg|`ay_z4AxwRRGi4xxBcsCNdA`|
|E06|https://hackmd.io/@yillkid/r1SoAy16zg|`NvyKAO48Sc2io8HcWRSS2w`|

### Lani 確認清單：疑似重複頁

曾在 EPIC 連結中出現的公開短 slug 與上述 internal ID 經核對屬於同一頁面的不同識別形式，不是兩份獨立筆記。未刪除任何頁面；目前**沒有已確認可刪的重複頁**。若將來出現同一課程的另一個獨立 note，先列出其 public URL、標題和內容差異，待 Lani 確認後才考慮刪除。

## 依賴邊界

- 必修先修只能指向更早的必修節次；選修不得成為必修完成條件。
- E01–E03 可建議先修必修14，三者互不硬依賴；E02 不替代必修20/27。
- E07–E09 可建議先修必修49；E09 可選擇建議 E07，但不得列為必修前置。
- 新51只驗收必修01–50，不要求選修證據；若引用舊選修量化紀錄，須標為非必要補充。
