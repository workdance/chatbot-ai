import asyncio
from typing import List, Tuple
from langchain_community.chat_models import ChatOllama
from langchain.schema import BaseMessage
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama2")
chain = llm | StrOutputParser()

async def get_messages_from_ollma(input: str) -> List[BaseMessage]:
    async for chunk in chain.astream(input):
        yield chunk