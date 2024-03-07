from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.modules.chat.chat_controller import chat_router
from app.modules.llm.ollama_controller import ollama_router
from app.modules.upload.upload_controller import upload_router

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ollama_router, tags=["ollama"])
app.include_router(upload_router, tags=["upload"])
app.include_router(chat_router, tags=["chat"])

@app.get("/demo")
def read_root():
    return {"Hello": "Chatbot AI Server"}



if __name__ == "__main__":
    # run main.py to debug backend
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=5050, reload=True)
