import json
from typing import Optional, ClassVar, List
from uuid import UUID

from langchain.callbacks import AsyncIteratorCallbackHandler
from pydantic import BaseModel

from app.config.settings import BrainSettings
from app.logger import get_logger
from app.modules.brain.brain_model import BrainModel
from app.modules.brain.brain_service import BrainService
from app.modules.chat.chat_service import ChatService
from app.modules.chat.dto.chat import ChatQuestion
from app.modules.chat.dto.inputs import CreateChatHistory
from app.llm.qa_interfacce import QAInterface
from app.llm.rags.no_doc_rag import NoDocRAG
from app.llm.utils.format_chat_history import format_chat_history

logger = get_logger(__name__)


brain_service = BrainService()
chat_service = ChatService()

def is_valid_uuid(uuid_to_test, version=4):
    try:
        uuid_obj = UUID(uuid_to_test, version=version)
    except ValueError:
        return False

    return str(uuid_obj) == uuid_to_test


# 支持简单的大模型调用
class BasicBrainQA(BaseModel, QAInterface):
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
    brain: Optional[BrainModel] = None
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

        self.brain = brain_service.get_brain_by_id(brain_id)
        # 默认就是用本地的DocRAG
        self.knowledge_qa = NoDocRAG(
            model=self.brain.model if self.brain.model else self.model,
            brain_id=brain_id,
            chat_id=chat_id,
            streaming=streaming,
            max_input=self.max_input,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        self.metadata = metadata

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
        self,
        chat_id: UUID,
        chat_question: ChatQuestion,
        save_answer: bool,
        *custom_params: tuple
    ):
        conversational_qa_chain = self.knowledge_qa.get_chain()
        transformed_history, streamed_chat_history = (
            self.initialize_streamed_chat_history(chat_id, chat_question)
        )
        response_tokens = []

        logger.info('knowledgeQA answer question]: %s', chat_question.question)
        async for chunk in conversational_qa_chain.astream(
                {
                    "question": chat_question.question,
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
        logger.info('[knowledgeQA answer done]: %s', "".join(response_tokens))

    async def generate_answer(
            self,
            chat_id: UUID,
            question: ChatQuestion,
            save_answer: bool,
            *custom_params: tuple
    ):
        return "un"

    def initialize_streamed_chat_history(self, chat_id, question):
        history = chat_service.get_chat_history(self.chat_id)
        transformed_history = format_chat_history(history)
        brain = brain_service.get_brain_by_id(self.brain_id)

        streamed_chat_history = CreateChatHistory(
                **{
                    "chat_id": chat_id,
                    "user_message": question.question,
                    "assistant": "",
                    "brain_id": brain.brain_id,
                    "prompt_id": self.prompt_to_use_id,
                }
            )
        return transformed_history, streamed_chat_history
