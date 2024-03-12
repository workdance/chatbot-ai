import asyncio
import json
from typing import Optional, List, Awaitable, ClassVar
from uuid import UUID

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from pydantic import BaseModel

from app.config.settings import BrainSettings
from app.logger import get_logger
from app.modules.chat.dto.chat import ChatQuestion
from app.modules.chat.dto.outputs import GetChatHistoryOutput
from app.modules.llm.rags.doc_rag_v2 import DocRAG
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
        self.callbacks = [callback]

        # 目前是通过ConversationalRetrievalChain来弄，未来看看有哪些更好的方式
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
        prompt_content = None
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
