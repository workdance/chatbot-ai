from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.modules.ollama.ollama_service import get_messages_from_ollma

ollama_router = APIRouter()

class ChatVO(BaseModel):
    question: str


@ollama_router.get("/ollama")
def get_ollama(question) -> StreamingResponse:
    print(question)
    answer = get_messages_from_ollma(question)
    try:
        return StreamingResponse(
            answer, media_type="text/event-stream",
        )
    except HTTPException as e:
        raise e

@ollama_router.post("/ollama")
def read_ollama(body: ChatVO) -> StreamingResponse:
    question = body.question
    answer = get_messages_from_ollma(question)
    try:
        return StreamingResponse(
            answer, media_type="text/event-stream",
        )
    except HTTPException as e:
        raise e