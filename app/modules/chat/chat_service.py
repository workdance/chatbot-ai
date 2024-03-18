from typing import List

from app.config.settings import ChatwebSettings
from app.modules.chat.dto.outputs import GetChatHistoryOutput
from app.util.chatweb_client import ChatwebClient

chatwebClient = ChatwebClient()

class ChatService:
    def __init__(self):
        self.server_url  = ChatwebSettings().server_url

    def get_chat_history(self, chat_id: str) -> List[GetChatHistoryOutput]:
        response = chatwebClient.post("/api/v1/chatHistory/list", {"chatId": chat_id})
        if response.ok:
            history = response.json().get("data")
            return history
        else:
            return []