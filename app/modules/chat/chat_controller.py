from uuid import UUID

from fastapi import APIRouter, Request, HTTPException
from starlette.responses import StreamingResponse

from app.logger import get_logger
from exceptiongroup import ExceptionGroup
from app.modules.brain.brain_service import BrainService
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

brain_service = BrainService()


def init_vector_store(user_id):
    pass


def get_answer_generator(chat_id: str, chat_question: ChatQuestion, brain_id: str):
    chat_instance = BrainfulChat()
    # vector_store = init_vector_store(user_id=current_user.get("user_id"))

    # Get History
    history = chat_service.get_chat_history(chat_id)

    model_to_use = LLMModels(  # TODO Implement default models in database
        name="qwen:14b", price=1, max_input=12000, max_output=1000
    )

    brain = brain_service.find_brain_from_question(brain_id, chat_question.question, history)
    # brain = BrainConfig(brain_type=BrainType.DOC, brain_id=brain_id)

    gpt_answer_generator = chat_instance.get_answer_generator(
        chat_id=str(chat_id),
        model=brain.model,
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
        chat_id: str,
        chat_question: ChatQuestion,
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
        pass
    except ExceptionGroup as e:
        print(f"捕获到异常组，包含 {len(e.exceptions)} 个子异常")

        # 遍历并打印所有子异常的详细信息

        for index, exc in enumerate(e.exceptions, start=1):
            print(f"子异常 {index}: {type(exc).__name__}: {exc}")


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
