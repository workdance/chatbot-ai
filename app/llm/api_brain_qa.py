import json
from typing import Optional
from uuid import UUID

from fastapi import HTTPException
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from app.llm.basic_brain_qa import BasicBrainQA
from app.llm.functions.restful_api import available_functions, functions_list_schema
from app.llm.knowledge_brain_qa import KnowledgeBrainQA
from app.llm.qa_interfacce import QAInterface
from app.logger import get_logger
from app.modules.brain.brain_service import BrainService
from app.modules.chat.dto.chat import ChatQuestion
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate

logger = get_logger(__name__)

brain_service = BrainService()

template = """您是一个有用的助手，现在可以访问函数来帮助回答问题。
如果问题中信息比较齐全，你可以直接回答问题。
如果问题中缺少信息，可以通过后续问题向用户获取更多信息。一旦所有信息都可用，就可以调用该函数来获得答案。
这次的问题是：{question}
"""
human_prompt = HumanMessagePromptTemplate.from_template(template)
class APIBrainQA(BasicBrainQA, QAInterface):
    def __init__(
            self,
            model: str,
            brain_id: str,
            chat_id: str,
            streaming: bool = False,
            prompt_id: Optional[UUID] = None,
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

    async def generate_stream(
            self,
            chat_id: UUID,
            chat_question: ChatQuestion,
            save_answer: bool,
            *custom_params: tuple
    ):
        response_tokens = []
        transformed_history, streamed_chat_history = (
            self.initialize_streamed_chat_history(chat_id, chat_question)
        )
        chatter = ChatOllama(model=self.brain.model)
        question = chat_question.question
        ai_response = None
        try:
            model = OllamaFunctions(model=self.brain.model)
            combined_message = transformed_history + [human_prompt.format(question=question)]

            # 构建好数据描述信息
            model = model.bind(
                functions=functions_list_schema,
            )

            ai_response = model.invoke(combined_message)

            print(ai_response)
            if hasattr(ai_response, 'additional_kwargs') and ai_response.additional_kwargs.get('function_call') is not None:

                function_name = ai_response.additional_kwargs["function_call"]["name"]
                function_to_call = available_functions[function_name]

                function_args = json.loads(ai_response.additional_kwargs["function_call"]["arguments"])

                logger.info('正在调用外部函数 %s', function_name)
                function_response = function_to_call(**function_args)

                chat_response = chatter.astream([
                    HumanMessage(content=question),
                    ai_response,
                    HumanMessage(content=f"请根据以下的函数调用结果进行回答, 函数调用的结果是: {json.dumps(function_response)}")
                ])
            else:
                chat_response = ai_response
        except Exception as e:
            logger.error(e)
            # raise HTTPException(status_code=400, detail=e)
            chat_response = chatter.astream([
                HumanMessage(content=question),
            ])
        async for chunk in chat_response:
            if chunk.content:
                # logger.info(f"Chunk: {chunk}")
                response_tokens.append(chunk.content)
                streamed_chat_history.assistant = chunk.content
                yield f"data: {json.dumps(streamed_chat_history.dict())}"