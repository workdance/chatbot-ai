from uuid import UUID

from fastapi import APIRouter, Request, HTTPException
from starlette.responses import StreamingResponse

from app.logger import get_logger
from app.modules.brain.brain_model import BrainConfig, BrainType
from app.modules.brain.brainful_chat import BrainfulChat
from app.modules.chat.chat_service import ChatService
from app.modules.chat.dto.chat import ChatQuestion
from app.modules.llm.model import LLMModels

logger = get_logger(__name__)

chat_router = APIRouter()
chat_service = ChatService()

current_user = {
    "user_id": "105766"
}


def init_vector_store(user_id):
    pass


def get_answer_generator(chat_id, chat_question, brain_id):
    chat_instance = BrainfulChat()
    vector_store = init_vector_store(user_id=current_user.get("user_id"))

    # Get History
    history = chat_service.get_chat_history(chat_id)

    model_to_use = LLMModels(  # TODO Implement default models in database
        name="qwen:14b", price=1, max_input=12000, max_output=1000
    )

    brain = BrainConfig(brain_type=BrainType.DOC, brain_id=brain_id)

    gpt_answer_generator = chat_instance.get_answer_generator(
        chat_id=str(chat_id),
        model=model_to_use.name,
        max_tokens=model_to_use.max_output,
        max_input=model_to_use.max_input,
        temperature=0.1,
        streaming=True,
        prompt_id=chat_question.prompt_id,
        brain=brain,
    )

    return gpt_answer_generator


@chat_router.post(
    "/chat/{chat_id}/question/stream",
    tags=["Chat"]
)
async def chat_question_stream(
        request: Request,
        chat_question: ChatQuestion,
        chat_id: UUID,
):
    # chat_instance = BrainfulChat()
    brain_id = chat_question.brain_id
    gpt_answer_generator = get_answer_generator(
        chat_id, chat_question, brain_id
    )
    try:
        return StreamingResponse(
            gpt_answer_generator.generate_stream(
                chat_id, chat_question, save_answer=True
            ),
            media_type="text/event-stream",
        )

    except HTTPException as e:
        raise e


@chat_router.post(
    "/chat/{chat_id}/question",
    tags=["Chat"]
)
async def chat_question(
        request: Request,
        chat_question: ChatQuestion,
        chat_id: UUID,
):
    # chat_instance = BrainfulChat()
    brain_id = chat_question.brain_id
    gpt_answer_generator = get_answer_generator(
        chat_id, chat_question, brain_id
    )
    try:
        return gpt_answer_generator.generate_answer(
            chat_id, chat_question, save_answer=True
        ),

    except HTTPException as e:
        raise e
