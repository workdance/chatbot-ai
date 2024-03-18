import asyncio
import json
from typing import Optional, List, Awaitable, ClassVar
from uuid import UUID

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from pydantic import BaseModel

from app.config.settings import BrainSettings
from app.logger import get_logger
from app.modules.brain.brain_service import BrainService
from app.modules.chat.chat_service import ChatService
from app.modules.chat.dto.chat import ChatQuestion
from app.modules.chat.dto.inputs import CreateChatHistory
from app.modules.chat.dto.outputs import GetChatHistoryOutput
from app.modules.llm.rags.doc_rag_v2 import DocRAG
from app.modules.llm.rags.rag_interface import RAGInterface
from app.modules.llm.utils.format_chat_history import format_chat_history

logger = get_logger(__name__)



brain_service = BrainService()
chat_service = ChatService()

def is_valid_uuid(uuid_to_test, version=4):
    try:
        uuid_obj = UUID(uuid_to_test, version=version)
    except ValueError:
        return False

    return str(uuid_obj) == uuid_to_test

async def wrap_done(fn: Awaitable, event: asyncio.Event):
    try:
        return await fn
    except Exception as e:
        logger.error(f"Caught exception: {e}")
        return None  # Or some sentinel value that indicates failure
    finally:
        event.set()


class KnowledgeBrainQA(BaseModel):
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
    knowledge_qa: Optional[RAGInterface] = None
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
            max_tokens: int,
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
            model=model,
            brain_id=brain_id,
            chat_id=chat_id,
            streaming=streaming,
            **kwargs,
        )
        self.metadata = metadata
        self.max_tokens = max_tokens

    @property
    def prompt_to_use(self):
        if self.brain_id and is_valid_uuid(self.brain_id):
            return None
        else:
            return None
    @property
    def prompt_to_use_id(self) -> Optional[UUID]:
        # TODO: move to prompt service or instruction or something
        if self.brain_id and is_valid_uuid(self.brain_id):
            # return get_prompt_to_use_id(UUID(self.brain_id), self.prompt_id)
            return None
        else:
            return None


    async def generate_stream(
            self, chat_id: str, question: ChatQuestion, save_answer: bool = True
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
            self, chat_id: UUID, question: ChatQuestion, save_answer: bool = True
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

    def initialize_streamed_chat_history(self, chat_id, question):
        history = chat_service.get_chat_history(self.chat_id)
        transformed_history = format_chat_history(history)
        brain = brain_service.get_brain_by_id(self.brain_id)

        streamed_chat_history = CreateChatHistory(
                **{
                    "chat_id": chat_id,
                    "user_message": question.question,
                    "assistant": "",
                    "brain_id": brain["brainId"],
                    "prompt_id": self.prompt_to_use_id,
                }
            )
        return transformed_history, streamed_chat_history