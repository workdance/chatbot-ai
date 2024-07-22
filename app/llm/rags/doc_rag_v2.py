from operator import itemgetter
from typing import Optional, ClassVar
from uuid import UUID

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings, ModelScopeEmbeddings
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import format_document, ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel

from app.config.settings import BrainSettings
from app.llm.model import MODEl_TYPE
from app.llm.vector_store import CustomVectorStore
from app.logger import get_logger

logger = get_logger(__name__)


# First step is to create the Rephrasing Prompt
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# Next is the answering prompt

# template = """仅根据文件中的提供的内容用中文回答问题:
template = """Answer the question in Chinese based only on the following context from files:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

# How we format documents

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="File: {page_content}"
)


class DocRAG(BaseModel):
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

    @property
    def embeddings(self):
        if self.model == MODEl_TYPE.QWEN:
            model_id = "damo/nlp_corom_sentence-embedding_english-base"
            return ModelScopeEmbeddings(model_id=model_id)
        else:
            return OllamaEmbeddings(
                base_url=self.brain_settings.ollama_api_base_url
            )

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
        self.vector_store = self._create_vector_store()
        # self.prompt_id = prompt_id
        self.max_tokens = max_tokens
        self.max_input = max_input
        self.model = model
        self.brain_id = brain_id
        self.chat_id = chat_id
        self.streaming = streaming

        logger.info(f"DocRAG initialized with model {model} and brain {brain_id}")
        logger.info("Max input length: " + str(self.max_input))

    def _create_vector_store(self):
        return CustomVectorStore(
            embedding=self.embeddings,
            brain_id=self.brain_id
        ).init_store()

    def get_retriever(self):
        return self.vector_store.as_retriever()

    def _combine_documents(self, docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
    ):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    def get_chain(self):
        retriever = self.get_retriever()

        _inputs = RunnableParallel(
            standalone_question=RunnablePassthrough.assign(
                chat_history=lambda x: get_buffer_string(x["chat_history"])
            )
                                | CONDENSE_QUESTION_PROMPT
                                | ChatOllama(temperature=0, model=self.model)
                                | StrOutputParser(),
        )

        _context = {
            "context": itemgetter("standalone_question")
                       | retriever
                       | self._combine_documents,
            "question": lambda x: x["standalone_question"],
        }
        conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOllama(model=self.model)

        return conversational_qa_chain