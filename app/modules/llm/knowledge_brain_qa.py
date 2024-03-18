import asyncio
import json
from typing import Optional, List, Awaitable, ClassVar
from uuid import UUID

from langchain.callbacks import AsyncIteratorCallbackHandler

from app.config.settings import BrainSettings
from app.logger import get_logger
from app.modules.chat.dto.chat import ChatQuestion
from app.modules.chat.dto.outputs import GetChatHistoryOutput
from app.modules.llm.basic_brain_qa import BasicBrainQA
from app.modules.llm.rags.doc_rag_v2 import DocRAG

logger = get_logger(__name__)


async def wrap_done(fn: Awaitable, event: asyncio.Event):
    try:
        return await fn
    except Exception as e:
        logger.error(f"Caught exception: {e}")
        return None  # Or some sentinel value that indicates failure
    finally:
        event.set()


class KnowledgeBrainQA(BasicBrainQA):
    class Config:
        """Configuration of the Pydantic Object"""

        arbitrary_types_allowed = True

    brain_setting: ClassVar[BrainSettings] = BrainSettings()
    # Default class attributes
    model: str = None  # pyright: ignore reportPrivateUsage=none
    temperature: float = 0.1
    chat_id: str = None  # pyright: ignore reportPrivateUsage=none
    brain_id: str  # pyright: ignore reportPrivateUsage=none
    max_tokens: int = 2000
    max_input: int = 2000
    streaming: bool = False
    knowledge_qa: Optional = None
    metadata: Optional[dict] = None
    callbacks: List[
        AsyncIteratorCallbackHandler
    ] = None  # pyright: ignore reportPrivateUsage=none

    prompt_id: Optional[UUID]

    def __init__(
            self,
            model: str,
            brain_id: str,
            chat_id: str,
            streaming: bool = False,
            prompt_id: Optional[UUID] = None,
            metadata: Optional[dict] = None,
            **kwargs,
    ):
        super().__init__(
            model=model,
            brain_id=brain_id,
            chat_id=chat_id,
            streaming=streaming,
            prompt_id=prompt_id,
            **kwargs,
        )

        # 默认就是用本地的DocRAG
        self.knowledge_qa = DocRAG(
            model=self.brain.model if self.brain.model else self.model,
            brain_id=brain_id,
            chat_id=chat_id,
            streaming=streaming,
            max_input=self.max_input,
            max_tokens=self.max_tokens,
            **kwargs,
        )

    async def generate_stream(
            self, chat_id: str, question: ChatQuestion, save_answer: bool = True, *custom_params: tuple
    ):
        conversational_qa_chain = self.knowledge_qa.get_chain()
        transformed_history, streamed_chat_history = (
            self.initialize_streamed_chat_history(chat_id, question)
        )
        response_tokens = []
        sources = []
        async for chunk in conversational_qa_chain.astream(
                {
                    "question": question.question,
                    "chat_history": transformed_history,
                    # "custom_personality": (
                    #         self.prompt_to_use.content if self.prompt_to_use else None
                    # ),
                }
        ):
            if chunk.content:
                # logger.info(f"Chunk: {chunk}")
                response_tokens.append(chunk.content)
                streamed_chat_history.assistant = chunk.content
                yield f"data: {json.dumps(streamed_chat_history.dict())}"

    def generate_answer(
            self, chat_id: UUID, question: ChatQuestion, save_answer: bool = True, *custom_params: tuple
    ) -> GetChatHistoryOutput:
        conversational_qa_chain = self.knowledge_qa.get_chain()
        transformed_history = ""
        model_response = conversational_qa_chain.invoke(
            {
                "question": question.question,
                "chat_history": transformed_history,
                "custom_personality": None,
            }
        )

        answer = model_response.content


        return GetChatHistoryOutput(
            **{
                "chat_id": chat_id,
                "user_message": question.question,
                "assistant": answer,
                "message_time": None,
                # "prompt_title": (
                #     self.prompt_to_use.title if self.prompt_to_use else
                # ),
                "prompt_title": None,
                "brain_name": None,
                "message_id": None,
                "brain_id": None,
            }
        )

