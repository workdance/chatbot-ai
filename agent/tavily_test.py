from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores.faiss import FAISS
from langchain.agents import OpenAIFunctionsAgent, create_openai_functions_agent, AgentExecutor

search = TavilySearchResults()

result = search.invoke("what is the weather in SF")

vector = FAISS.load_local('../temp/vectorstore/9f5cd71a-4c5f-43aa-b936-8c66f1dbb245.index', OllamaEmbeddings())
retriever = vector.as_retriever()
# print(retriever.get_relevant_documents("xixi")[0])

retriever_tool = create_retriever_tool(retriever, "family information", "Search for information about 湖州")

tools = [search, retriever_tool]


# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
# print(prompt.messages)

llm = ChatOllama(model="qwen:14b",temperature=0)

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(agent_executor.invoke({"input": "hi!"}))