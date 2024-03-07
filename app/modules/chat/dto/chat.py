from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class ChatQuestion(BaseModel):
    question: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    brain_id: Optional[str] = None
    prompt_id: Optional[str] = None
