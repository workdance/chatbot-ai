
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel


class GetChatHistoryOutput(BaseModel):
    chat_id: UUID
    message_id: Optional[str] = None
    user_message: str
    assistant: str
    message_time: Optional[str] = None
    prompt_title: Optional[str] = None
    brain_name: Optional[str] = None
    brain_id: Optional[str] = (
        None  # string because UUID is not JSON serializable
    )
    metadata: Optional[dict] = None

    def dict(self, *args, **kwargs):
        chat_history = super().dict(*args, **kwargs)
        chat_history["chat_id"] = str(chat_history.get("chat_id"))
        chat_history["message_id"] = str(chat_history.get("message_id"))

        return chat_history