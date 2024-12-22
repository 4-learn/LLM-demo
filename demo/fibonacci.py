# 基於 LLM 的簡易智能代理示例
from phi.agent import Agent
from phi.model.openai import OpenAIChat

# 創建一個基於 GPT-4 的智能代理
agent = Agent(
    model=OpenAIChat(id="gpt-4"),
    description="您是一位可以執行任務的智能助手。"
)

# 使用代理進行多步任務
agent.print_response("幫我寫一段 Python 程式碼來計算斐波那契數列。")
