from operator import itemgetter
from typing import Optional, ClassVar
from uuid import UUID

from langchain_community.chat_models import ChatLiteLLM, ChatOllama
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import format_document, ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel

from app.config.settings import BrainSettings
from app.logger import get_logger
from app.llm.vector_store import CustomVectorStore

logger = get_logger(__name__)


# First step is to create the Rephrasing Prompt
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# Next is the answering prompt
template = """Answer the question in Chinese:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

# How we format documents

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="File: {page_content}"
)


class NoDocRAG(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    brain_settings: ClassVar[BrainSettings] = BrainSettings()

    # Default class attributes
    model: str = None  # pyright: ignore reportPrivateUsage=none
    temperature: float = 0.1
    chat_id: str = None  # pyright: ignore reportPrivateUsage=none
    brain_id: str = None  # pyright: ignore reportPrivateUsage=none
    prompt_id: str = None
    max_tokens: int = 2000  # Output length
    max_input: int = 2000
    streaming: bool = False
    vector_store: Optional[CustomVectorStore] = None


    def __init__(
            self,
            model: str,
            brain_id: str,
            chat_id: str,
            streaming: bool = False,
            prompt_id: Optional[UUID] = None,
            max_tokens: int = 2000,
            max_input: int = 2000,
            **kwargs,
    ):
        super().__init__(
            model=model,
            brain_id=brain_id,
            chat_id=chat_id,
            streaming=streaming,
            max_tokens=max_tokens,
            max_input=max_input,
            **kwargs,
        )
        # self.prompt_id = prompt_id
        self.max_tokens = max_tokens
        self.max_input = max_input
        self.model = model
        self.brain_id = brain_id
        self.chat_id = chat_id
        self.streaming = streaming

        logger.info(f"RAG initialized with model {model} and brain {brain_id}")
        logger.info("Max input length: " + str(self.max_input))

    def get_chain(self):

        _inputs = RunnableParallel(
            standalone_question=RunnablePassthrough.assign(
                chat_history=lambda x: get_buffer_string(x["chat_history"])
            )
                                | CONDENSE_QUESTION_PROMPT
                                | ChatOllama(temperature=0, model=self.model)
                                | StrOutputParser(),
        )

        _context = {
            "context": itemgetter("standalone_question"),
            "question": lambda x: x["standalone_question"],
        }
        conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOllama(model=self.model)

        return conversational_qa_chain