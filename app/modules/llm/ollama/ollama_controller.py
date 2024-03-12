from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.modules.llm.ollama.ollama_service import get_messages_from_ollma

ollama_router = APIRouter()

class ChatVO(BaseModel):
    question: str


@ollama_router.post("/ollama")
def read_ollma(body: ChatVO):
    question = body.question
    print(question)
    answer = get_messages_from_ollma(question)
    return StreamingResponse(
        answer, media_type="text/event-stream",
    )