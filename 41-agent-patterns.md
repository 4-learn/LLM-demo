# 41｜固定流程與 Agent

> 本課只學一件事：固定步驟的 workflow 和會依結果選下一步的 agent 不一樣。

[回課程書目](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 比較兩種做法

**固定流程**：查天氣 → 整理結果 → 回答。

**Agent**：先看問題；如果沒資料就查詢，有資料就整理。下一步會依目前結果改變。

工作越固定，通常越適合先用固定流程。Agent 多了選擇能力，也多了出錯的可能。

## Demo：依結果選擇下一步

<!-- demo: 41_agent_patterns.py -->
```python
weather = "晴天"

if weather == "未知":
    next_step = "查詢天氣"
else:
    next_step = "整理並回答"

print(next_step)
```

## 動手試試

把 `weather` 改成 `"未知"`，再執行。若 Agent 每一步都沒有停止條件，可能一直重複做事；真實系統必須設定步數上限。

**延伸名詞**：記憶、工具權限與多代理協作是進階主題。
