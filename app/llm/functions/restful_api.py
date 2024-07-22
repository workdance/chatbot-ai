import json
import os

import requests

from app.llm.functions.utils import serialize_function_to_json, generate_function_call_content
from app.modules.brain.brain_service import BrainService

brain_service = BrainService()


def get_current_weather(location: str) -> str:
    """
    获取给定城市的天气情况
    Parameters:
        location(str): 这个城市的名称，例如：杭州

    Returns:
        (str): 当前城市的天气情况
    """
    APIKEY = os.environ.get('GAODE_API_KEY')
    geoUrl = f"https://restapi.amap.com/v3/geocode/geo?address={location}&output=JSON&key={APIKEY}"
    response = requests.get(geoUrl).json()
    adCode = response["geocodes"][0]["adcode"]
    url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={adCode}&key={APIKEY}"
    weather_response = requests.get(url)
    data = weather_response.json()
    live = data["lives"][0]
    # 也可以直接把数据扔给模型，模型自己会组装
    return live
    # return f"天气：{live['weather']}，温度:{live['temperature']}，风向:{live['winddirection']}"


def get_brain_data(brainId: str) -> dict:
    return brain_service.get_brain_by_id(brain_id=brainId).json()


def get_braindata_by_user_id(userId: str) -> dict:
    return brain_service.get_all_brains(user_id=userId).json()


functions_list_schema = [
    {
        "name": get_current_weather.__name__,
        "description": "获取给定城市的天气情况",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "这个城市的名称，例如：杭州",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": get_brain_data.__name__,
        "description": "根据 brianId 获取 chatbot中的大脑详情",
        "parameters": {
            "type": "object",
            "properties": {
                "brainId": {
                    "type": "string",
                    "description": "大脑的 Id，是一个字符串",
                }
            },
            "required": ["brainId"],
        },
    },
    {
        "name": get_braindata_by_user_id.__name__,
        "description": "根据 userId 获取 chatbot 中的大脑列表",
        "parameters": {
            "type": "object",
            "properties": {
                "userId": {
                    "type": "string",
                    "description": "userId 是一个字符串",
                }
            },
            "required": ["userId"],
        },
    }
]

available_functions = {
    get_current_weather.__name__: get_current_weather,
    get_brain_data.__name__: get_brain_data,
    get_braindata_by_user_id.__name__: get_braindata_by_user_id,
}
