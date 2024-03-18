from app.logger import get_logger
from app.modules.brain.brain_model import BrainType
from app.modules.chat.chat_model import ChatInterface
from app.modules.llm.api_brain_qa import APIBrainQA
from app.modules.llm.basic_brain_qa import BasicBrainQA
from app.modules.llm.knowledge_brain_qa import KnowledgeBrainQA
from app.modules.llm.model import models_supporting_function_calls

logger = get_logger(__name__)


class BrainfulChat(ChatInterface):
    def validate_authorization(self, user_id, required_roles):
        return True

    def get_answer_generator(
            self,
            brain,
            chat_id,
            model,
            temperature,
            streaming,
            prompt_id,
            # metadata

    ):
        if(brain.brain_type == BrainType.BASIC):
            return BasicBrainQA(
                chat_id=chat_id,
                model=model,
                temperature=temperature,
                brain_id=str(brain.brain_id),
                streaming=streaming,
                prompt_id=prompt_id,
            )
        if (
                brain
                and brain.brain_type == BrainType.DOC
        ):
            return KnowledgeBrainQA(
                chat_id=chat_id,
                model=model,
                temperature=temperature,
                brain_id=str(brain.brain_id),
                streaming=streaming,
                prompt_id=prompt_id,
                # metadata=metadata,
            )

        if (brain.brain_type == BrainType.API):
            return APIBrainQA(
                chat_id=chat_id,
                model=model,
                temperature=temperature,
                brain_id=str(brain.brain_id),
                streaming=streaming,
                prompt_id=prompt_id,
                # metadata=metadata,
            )