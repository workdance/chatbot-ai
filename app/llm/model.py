from pydantic import BaseModel

models_supporting_function_calls = [
    "gpt-4",
    "gpt-4-1106-preview",
    "gpt-4-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613",
]

models_supporting_local = [
    "llama2",
    "gemma:7b",
    "qwen:14b",
]

class LLMModels(BaseModel):
    """LLM models stored in the database that are allowed to be used by the users.
    Args:
        BaseModel (BaseModel): Pydantic BaseModel
    """

    name: str = "llama2"  # 本地模型，支持gemma:7b、llama2 以及 qwen:14b
    price: int = 1
    max_input: int = 512
    max_output: int = 512
