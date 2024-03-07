from app.config.settings import ChatwebSettings


class ChatService:
    def __init__(self):
        self.server_url  = ChatwebSettings().server_url

    def get_chat_history(self, chat_id: str):
        return [{
            "chat_id": chat_id,
            "message": "message"
        }]