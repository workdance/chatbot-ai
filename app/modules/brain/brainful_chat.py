from app.logger import get_logger
from app.modules.brain.model import BrainType
from app.modules.chat.model import ChatInterface
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
            max_tokens,
            max_input,
            temperature,
            streaming,
            prompt_id,
            # metadata

    ):
        if (
                brain
                and brain.brain_type == BrainType.DOC
                or model not in models_supporting_function_calls
        ):
            return KnowledgeBrainQA(
                chat_id=chat_id,
                model=model,
                max_tokens=max_tokens,
                max_input=max_input,
                temperature=temperature,
                brain_id=str(brain.brain_id),
                streaming=streaming,
                prompt_id=prompt_id,
                # metadata=metadata,
            )
