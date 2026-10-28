# 41｜Agent 設計｜依結果選擇下一步

> 本課只學一件事：Agent 依結果選下一步，並設步數上限。

## 固定流程與 Agent

固定流程預排每一步；Agent 依結果決定下一步。步數上限限制重試次數，但不會自動去重或確保選擇正確。

## Demo：選下一步，程式限制步數

<!-- demo: 41_agent_patterns.py -->
```python
weather = "未知"
step_limit = 3
for step in range(step_limit):
    if weather == "未知":
        next_step = "查詢天氣"
        print("第", step + 1, "步：模擬模型選擇：", next_step)
        weather = "晴天"
    else:
        next_step = "整理並回答"
        print("第", step + 1, "步：模擬模型選擇：", next_step)
        break
else:
    print("達到步數上限，停止")
```

預期先查天氣，再整理回答。`if` 模擬模型選擇；真實 Agent 由模型依結果選下一步，程式執行並限步。本例未呼叫模型或天氣服務。

## 這跟 LLM 的關係

Agent 常由 LLM 根據對話或工具結果提出動作。模型負責建議，程式應驗證動作、限制工具並設定停止條件。

## 動手試試

把 `step_limit` 改成 `1`；預期先印查詢步驟，再印「達到步數上限，停止」。上限不負責去重。

## 自我檢查
1. 誰決定下一步？2. `if` 是 LLM 嗎？3. 為何限步？
答案：固定流程預排、Agent 依結果選；不是；限執行次數。
