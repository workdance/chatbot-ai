import asyncio
import unittest

from app.llm.rags.no_doc_rag import NoDocRAG


async def testRag():
    rag = NoDocRAG(
            model="qwen:14b",
            brain_id="e1821f28-7906-4ab2-9a1e-f787b5a66abe",
            chat_id="ca96ad72-e4b8-45f7-a3fe-da08ac0548c6",
            streaming=True,
            max_input=2000,
            max_tokens=2000
        )
    chain = rag.get_chain()
    async for chunk in chain.astream(
            {
                "question": "你是谁",
                "chat_history": [],
                # "custom_personality": (
                #         self.prompt_to_use.content if self.prompt_to_use else None
                # ),
            }
    ):
        if chunk:
            print(f"Chunk: {chunk}")


asyncio.run(testRag())