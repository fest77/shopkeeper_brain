from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()

# 1. 创建 LLM 客户端
llm = ChatOpenAI(
    model="qwen-flash",
    temperature=0.0,           # 低温度 = 更确定性输出
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# 2. 构建消息
messages = [
    SystemMessage(content="你是商品识别专家，只输出字符串。"),
    HumanMessage(content="请从以下信息中识别商品名称：R12万用表如何使用?"),
]

# 3. 调用并获取响应
response = llm.invoke(messages)
item_name = response.content.strip()
print(item_name) #R12万用表