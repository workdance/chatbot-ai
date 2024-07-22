from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="qwen:14b")
chain = llm | StrOutputParser()

question = input("请输入问题：")
print("正在思考中...")
response = chain.invoke(question)
print("AI 模型反馈：")
print(response)