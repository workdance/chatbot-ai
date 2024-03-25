import inspect
import json
from typing import get_type_hints


def get_type_name(t):
    name = str(t)
    if "list" in name or "dict" in name:
        return name
    else:
        return t.__name__


def serialize_function_to_json(func):
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    function_info = {
        "name": func.__name__,
        "description": func.__doc__,
        "parameters": {
            "type": "object",
            "properties": {}
        },
        "returns": type_hints.get('return', 'void').__doc__
    }

    for name, _ in signature.parameters.items():
        param_type = get_type_name(type_hints.get(name, type(None)))
        function_info["parameters"]["properties"][name] = {"type": param_type}

    return json.dumps(function_info, indent=2)


def generate_function_call_content(func):
    """
    Generates the call content for OpenAI Function Calling based on a Python function and its docstring.

    Parameters:
    func (function): The Python function to generate call content for.

    Returns:
    str: A string representation suitable for OpenAI Function Calling.
    """

    # 获取函数名称
    func_name = func.__name__

    # 获取函数签名
    func_signature = inspect.signature(func)

    # 获取类型提示
    type_hints = get_type_hints(func)

    # 解析文档字符串
    docstring = inspect.getdoc(func)
    # (此处可以进一步解析文档字符串以提取更多信息)

    # 构造参数列表
    params_str = ", ".join([f"{param}: {type_hints.get(param, type(param)).__name__}"
                            for param in func_signature.parameters])

    # 生成函数调用内容
    newline = '\n'
    call_content = f"Function: {func_name}\n"
    call_content += f"Parameters: {params_str}\n"
    call_content += f"Description: {docstring.split(newline)[0]}\n"
    # 添加额外所需的生成信息

    return call_content
