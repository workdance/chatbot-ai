from typing import List, Tuple
from langchain_community.chat_models import ChatOllama
from langchain.schema import BaseMessage
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama2")
chain = llm | StrOutputParser()

def get_messages_from_ollma(input: str) -> List[BaseMessage]:
    return chain.stream(input)