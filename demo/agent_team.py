from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo

def introduce_self():
    """自我介紹"""
    return """
    大家好！我是一個智能助理，我可以：
    1. 自我介紹
    2. 搜尋新聞
    
    您可以對我說「搜尋一下 LLM 目前的新聞」或類似的指令，我會幫您完成操作！
    """

intro_agent = Agent(
    name="Introduction Agent",
    role="進行自我介紹",
    instructions=[
        "1. 使用 introduce_self 工具進行自我介紹",
        "2. 提供友善且專業的介紹"
    ],
    tools=[introduce_self]
)

web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)


agent_team = Agent(
    team=[web_agent, web_agent],
    instructions=["使用一個或多個代理工具"],
    show_tool_calls=True
)

# agent_team.print_response("請自我介紹")
agent_team.print_response("最近 LLM 有啥新聞阿？")
