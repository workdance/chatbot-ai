from pydantic.v1 import BaseSettings


class BrainSettings(BaseSettings):
    ollama_api_base_url: str = "http://localhost:11434"


class ChatwebSettings(BaseSettings):
    server_url: str = "http://localhost:8080"

