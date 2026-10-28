# Day 14｜Dify 安裝前：確認環境與服務邊界

> **主題：** 透過容器部署 LLM 工作流平台之前，先檢查 Docker 是否可用，並理解本地服務、連接埠及模型 API 的界線。本章不直接安裝或啟動服務。

## 動手做：檢查本機命令

```python
import shutil

for command in ("docker", "git", "python3"):
    path = shutil.which(command)
    print(f"{command}: {'available' if path else 'not found'}")
```

預期輸出會依你的環境不同。例如沒有安裝 Docker、但有 Git 和 Python 時：

```text
docker: not found
git: available
python3: available
```

請以自己執行所得為準。若要安裝 Dify，還要確認 Docker Compose、記憶體與磁碟空間、使用的映像版本、暴露的連接埠，以及資料與秘密的保存方式。不要直接將測試服務暴露到公網。

## 小練習

記下你實際看到的三行輸出；若 `docker` 可用，另查 Docker Engine 和 Compose 的版本（不需要貼出任何憑證）。想一想：本機服務的管理者能存取哪些資料？

## 原文與版本

原文：[Day 14｜Dify 安裝：使用 Docker 安裝設定模型並啟動你的第一個 LLM 聊天助手](https://ithelp.ithome.com.tw/articles/10360064?sc=rss.iron)

原文安裝指令、Dify 映像、資源估計與 UI 屬於 2024 年環境；此處不照抄舊安裝命令。實際安裝前核對 Dify 與 Docker 官方文件的支援版本、部署安全指引及模型供應商文件。模型 API key 僅存入核准的本地秘密設定，不寫進教材或 repo。
