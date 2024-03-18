from typing import Optional
from uuid import UUID

from fastapi import HTTPException

from app.modules.llm.knowledge_brain_qa import KnowledgeBrainQA
from app.modules.llm.qa_interfacce import QAInterface


class APIBrainQA(KnowledgeBrainQA, QAInterface):
    def __init__(
            self,
            model: str,
            brain_id: str,
            chat_id: str,
            streaming: bool = False,
            prompt_id: Optional[UUID] = None,
            raw: bool = False,
            jq_instructions: Optional[str] = None,
            **kwargs,
    ):
        user_id = kwargs.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="Cannot find user id")

        super().__init__(
            model=model,
            brain_id=brain_id,
            chat_id=chat_id,
            streaming=streaming,
            prompt_id=prompt_id,
            **kwargs,
        )
        self.user_id = user_id
        self.raw = raw
        self.jq_instructions = jq_instructions

