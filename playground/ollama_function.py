import json
import requests
from langchain_community.chat_models import ChatOllama
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from app.llm.functions.restful_api import get_current_weather
from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage

model = OllamaFunctions(model="qwen:14b")
chatter = ChatOllama(model="qwen:14b")


avaiable_functions = {
    "get_current_weather": get_current_weather
}

# 构建好数据描述信息
model = model.bind(
    functions=[
        {
            "name": "get_current_weather",
            "description": "获取给定城市的天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "这个城市的名称，例如：杭州",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        }
    ],
)

question = input("请输入问题：")
print("正在思考中...")
ai_response = model.invoke([
    # HumanMessage("您是一个有用的助手，可以访问功能来帮助回答问题。如果问题中缺少信息，可以通过后续问题向用户获取更多信息。一旦所有信息都可用，就可以调用该函数来获得答案"),
    HumanMessage(question)
])
# content='' additional_kwargs={'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "Beijing", "unit": "celsius"}'}}


print("模型第一次返回：")
print(ai_response)

function_name = ai_response.additional_kwargs["function_call"]["name"]
function_to_call = avaiable_functions[function_name]

function_args = json.loads(ai_response.additional_kwargs["function_call"]["arguments"])

print(f"-- 正在调用外部函数:{function_name} --")
function_response = function_to_call(**function_args)
print("--- 结束调用外部函数结果 --")

print(function_response)

# response.append(function_response)

# response.append(AIMessage(function_response))

# print(response)

# model.invoke("what is the weather in")

fresponse = chatter.invoke([
            HumanMessage(content=question),
            AIMessage(content=ai_response.content),
            HumanMessage(content=f"请根据以下的函数调用结果进行回答, 函数调用的结果是: {json.dumps(function_response)}")
        ])

print("--- 最终返回给用户结果 --")
print(fresponse)