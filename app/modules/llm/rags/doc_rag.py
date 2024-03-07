from typing import Optional, ClassVar
from uuid import UUID

from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings
from pydantic import BaseModel

from app.config.settings import BrainSettings
from app.logger import get_logger
from app.modules.llm.prompts.CONDENSE_PROMPT import CONDENSE_QUESTION_PROMPT
from app.modules.llm.rags.rag_interface import RAGInterface
from app.modules.llm.vector_store import CustomVectorStore

logger = get_logger(__name__)


class DocRAG(BaseModel, RAGInterface):
    brain_settings: ClassVar[BrainSettings] = BrainSettings()

    # Default class attributes
    model: str = None  # pyright: ignore reportPrivateUsage=none
    temperature: float = 0.1
    chat_id: str = None  # pyright: ignore reportPrivateUsage=none
    brain_id: str = None  # pyright: ignore reportPrivateUsage=none
    max_tokens: int = 2000  # Output length
    max_input: int = 2000
    streaming: bool = False
    vector_store: Optional[CustomVectorStore] = None
    @property
    def embeddings(self):
        if self.brain_settings.ollama_api_base_url:
            return OllamaEmbeddings(
                base_url=self.brain_settings.ollama_api_base_url
            )
        else:
            return OpenAIEmbeddings()

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
        self.prompt_id = prompt_id
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

    def _create_llm(
            self,
            callbacks,
            model,
            streaming=False,
            temperature=0,
    ) -> ChatLiteLLM:
        if streaming and callbacks is None:
            raise ValueError(
                "Callbacks must be provided when using streaming language models"
            )

        # 创建本地模型
        api_base = None
        if self.brain_settings.ollama_api_base_url and model.startswith("ollama"):
            api_base = self.brain_settings.ollama_api_base_url

        return ChatLiteLLM(
            temperature=temperature,
            max_tokens=self.max_tokens,
            model=model,
            streaming=streaming,
            verbose=False,
            callbacks=callbacks,
            api_base=api_base,
        )

    def get_doc_chain(self, streaming, callbacks=None):
        answering_llm = self._create_llm(
            model=self.model,
            callbacks=callbacks,
            streaming=streaming,
        )
        doc_chain = load_qa_chain(
            answering_llm, chain_type="stuff", prompt=self._create_prompt_template()
        )
        return doc_chain

    def get_question_generation_llm(self):
        return LLMChain(
            llm=self._create_llm(model=self.model, callbacks=None),
            prompt=CONDENSE_QUESTION_PROMPT,
            callbacks=None,
        )