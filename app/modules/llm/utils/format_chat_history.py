from typing import List, Dict, Union

from langchain_core.messages import HumanMessage, AIMessage

from app.modules.chat.dto.outputs import GetChatHistoryOutput


def format_chat_history(
    history: List[GetChatHistoryOutput],
) -> list[Union[HumanMessage, AIMessage]]:
    """Format the chat history into a list of HumanMessage and AIMessage"""
    formatted_history = []
    for chat in history:
        if chat['assistant'] is None:
            assistant = "No assistant"
        else:
            assistant = chat['assistant']

        formatted_history.append(HumanMessage(chat['userMessage']))
        formatted_history.append(AIMessage(assistant))
    return formatted_history
