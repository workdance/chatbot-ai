import asyncio
import json
from typing import Optional, List, Awaitable, ClassVar
from uuid import UUID

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from pydantic import BaseModel

from app.config.settings import BrainSettings
from app.logger import get_logger
from app.modules.chat.dto.chat import ChatQuestion
from app.modules.llm.rags.doc_rag import DocRAG
from app.modules.llm.rags.rag_interface import RAGInterface

logger = get_logger(__name__)


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

    async def generate_stream(
            self, chat_id: UUID, question: ChatQuestion, save_answer: bool = True
    ):
        callback = AsyncIteratorCallbackHandler()

        qa = ConversationalRetrievalChain(
            retriever=self.knowledge_qa.get_retriever(),
            combine_docs_chain=self.knowledge_qa.get_doc_chain(
                callbacks=self.callbacks,
                streaming=True,
            ),
            question_generator=self.knowledge_qa.get_question_generation_llm(),
            verbose=False,
            rephrase_question=False,
            return_source_documents=True,
        )
        transformed_history = ""
        prompt_content = ""
        response_tokens = []

        run = asyncio.create_task(
            wrap_done(
                qa.acall(
                    {
                        "question": question.question,
                        "chat_history": transformed_history,
                        "custom_personality": prompt_content,
                    }
                ),
                callback.done,
            )
        )
        streamed_chat_history = {
            "user_message": question.question,
            "assistant": "",
        }
        try:
            async for token in callback.aiter():
                logger.debug("Token: %s", token)
                response_tokens.append(token)
                streamed_chat_history.assistant = token
                yield f"data: {json.dumps(streamed_chat_history)}"
        except Exception as e:
            logger.error("Error during streaming tokens: %s", e)
        try:
            result = run()
        except Exception as e:
            logger.error("Error processing source documents: %s", e)
        assistant = "".join(response_tokens)
        logger.info("Assistant: %s", assistant)