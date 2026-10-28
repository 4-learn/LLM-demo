# Day 11｜Unsloth 微調：資料、Adapter 與可重現評估

> **主題：** 認識監督式微調與參數高效微調（如 LoRA）的基本流程；先檢查資料和評估方法，不把「成功跑完訓練」等同於模型變好。

## 學習重點

預訓練模型可透過微調適應特定任務；LoRA（低秩適配）凍結基礎模型並訓練較小的低秩更新，降低可訓練參數量。GPU 是平行處理大量運算的圖形處理器，常用於模型訓練。資料品質、授權、過度擬合與獨立測試集仍然重要。原文用 Unsloth、Colab GPU、模型轉檔及 Ollama/Open WebUI 示範；那些 UI（使用者介面）、模型與套件細節皆屬發文當時狀態。

## 動手做：檢查微調資料

先用小型合成資料練習基本格式檢查；這不是訓練程式，也不會呼叫雲端服務。

```python
examples = [
    {"instruction": "退貨期限？", "input": "", "output": "請於收貨後七日內申請。"},
    {"instruction": "訂單怎麼查？", "input": "", "output": "到會員頁查看訂單狀態。"},
]

required = {"instruction", "input", "output"}
for i, row in enumerate(examples, start=1):
    assert set(row) == required and all(isinstance(row[k], str) for k in required)
    assert row["instruction"].strip() and row["output"].strip()
    print(f"example {i}: valid")
print(f"total: {len(examples)}")
```

預期輸出：

```text
example 1: valid
example 2: valid
total: 2
```

## 小練習

在測試集加入一題訓練資料沒有出現過的問法，寫下模型應該如何回答或拒答。思考如何避免把測試題也拿去訓練，造成評估失真。

## 原文與版本

## 版本與安全

原文：[Day 11｜把知識放到腦袋中：使用 Unsloth 微調大型語言模型](https://ithelp.ithome.com.tw/articles/10357958?sc=rss.iron)

原文日期為 2024 年。Unsloth、Colab GPU、Llama 版本、Transformers/訓練套件、Ollama 及模型授權都會變動；正式實作前核對各官方文件、硬體需求及模型授權。只使用有權使用的合成/授權資料；不要將 API key 或 Hugging Face token 寫入 Notebook、公開 repo 或輸出紀錄。此處只執行本地 Python 資料檢查，未執行 GPU 微調。
