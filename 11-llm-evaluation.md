# 進階 LLM/RAG 評估、Judge 校準與回歸測試

> **正文改版：2026-04-16｜對應新版 LLM 第 11 節：進階 LLM/RAG 評估、Judge 校準與回歸測試。**本節以小型標註資料示範分類指標、裁判校準與回歸比較；本節不是用少量合成案例證明系統可部署。

[回 50 節課綱](https://hackmd.io/@yillkid/HJl8JXTNkl)

## 學習目標與時間

先備：理解預測與標準答案、能讀 Python 列表。50 分鐘：概念 20 分、Demo 10 分、Workshop 15 分、回顧 5 分。完成後能計算混淆計數與一致率、指出 Judge 的限制，並設計固定案例回歸測試。

## 一、評估先定義任務與標註

評分前先明確規定成功條件、資料來源與標準答案如何取得。RAG 不只看答案是否正確，也可分別檢查檢索是否涵蓋支持證據、回答是否忠於證據、引用是否可核對。將各環節混成單一總分，會使失敗原因難以定位。

## 二、指標回答不同問題

Accuracy 是整體正確比例；召回率看正例有多少被找回，精確率看判為正例者有多少正確。類別不平衡時，單報 accuracy 可能掩蓋少數類別失敗。任何指標都依賴資料代表性與標註品質，不自動等同真實世界效用。

## 三、LLM Judge 需要校準

裁判模型可能受提示措辭、答案順序、長度與自身偏好影響。應抽樣人工雙評、看一致率與分歧案例，做順序互換測試，並揭露不確定與裁判模型版本。Judge 分數是另一種測量工具，不是客觀真理。

## 四、回歸測試應保留失敗

固定案例集可在改提示、檢索或模型後比較退化。除了平均值，也追蹤逐案例差異與關鍵安全案例；新版本不得因平均提升而掩蓋重大錯誤。測試集合本身也需版本控管，避免無意洩漏或過度貼合。

## 五、可執行 Demo

以下計算二元分類的 TP、FP、FN 與 precision、recall，並拒絕空資料及長度不符。

<!-- demo: 11_llm_evaluation.py -->
```python
def binary_metrics(truth, predicted):
    if len(truth) != len(predicted) or not truth:
        raise ValueError("nonempty inputs must have equal length")
    if any(type(x) is not int or x not in (0, 1) for x in truth + predicted):
        raise ValueError("labels must be integer 0 or 1")
    tp = sum(t == 1 and p == 1 for t, p in zip(truth, predicted))
    fp = sum(t == 0 and p == 1 for t, p in zip(truth, predicted))
    fn = sum(t == 1 and p == 0 for t, p in zip(truth, predicted))
    return tp, fp, fn, tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0

if __name__ == "__main__":
    print(binary_metrics([1, 1, 0], [1, 0, 1]))
```

執行：`python3 11_llm_evaluation.py`。預期 stdout（須隨教材程式執行比對）：

```text
(1, 1, 1, 0.5, 0.5)
```

輸出僅為合成標籤的計算，不可視為真實模型成績。

## Workshop

12 分鐘：手算三筆案例的 TP/FP/FN 與精確率、召回率，執行核對；3 分鐘：把預測改成全 0，觀察零分母處理。參考判讀：第一組 TP=1、FP=1、FN=1，兩指標皆 0.5。校準練習：兩位標註者獨立判同一組 6 案例，再交換候選答案順序重跑；列出一致率與分歧，不把一致率當正確率。驗收：執行後輸出吻合；另外測空列表、長度不符、布林標籤及標籤 2 均拋 `ValueError`；測全負例 precision=0、全正例 recall=0；新增一筆回歸案例並逐筆比較版本結果。

## 教師紀錄與下一步

證據級別：本機同源測試（2026-04-16；標準函式庫；`test_lessons.py` 擷取同一 demo block 驗證二元指標與錯誤路徑；非模型評估或 Judge 校準實證）。環境為 Linux／Python 3.11，模型 revision 不適用。尚未實班試教。限制：二元指標不涵蓋分級品質、引用忠實度或統計不確定性。下一節依 mapping 為新12 Structured Output。舊教材僅歷史參照。
