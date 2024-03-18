import os
import time

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import CharacterTextSplitter

from app.logger import get_logger
from app.util.files.file import get_knowledge_directory

# nltk.download('punkt')

logger = get_logger(__name__)


# 这里要先去查数据库，然后数据库的内容做更新
# 然后每次更新 knowledge 的时候，主动删除 vectorstores 的缓存
def get_knowledge_data(file_directory):
    target_directory = get_knowledge_directory(file_directory)
    data = []
    sources = []

    for file in os.listdir(target_directory):
        file_path = os.path.join(target_directory, file)
        if file.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            data.extend(loader.load())
        elif file.endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_path)
            data.append(loader.load())
        elif file.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
            data.append(loader.load())
        else:
            loader = TextLoader(file_path)
            data.append(loader.load())
        sources.append(file_path)
    return data, sources


def get_vectorstore_directory(brainId):
    temp_vs_path = os.path.join(os.getcwd(), "temp/vectorstore")
    return os.path.join(temp_vs_path, f"{brainId}.index")

class CustomVectorStore:

    number_docs: int = 35
    max_input: int = 2000

    def __init__(
            self,
            embedding: Embeddings,
            brain_id: str = "none",
            user_id: str = "none",
            number_docs: int = 35,
            max_input: int = 2000,
    ):
        self.embedding = embedding
        self.brain_id = brain_id
        self.user_id = user_id
        self.number_docs = number_docs
        self.max_input = max_input

    # 利用 FAISS 初始化向量数据库
    # 文档地址：https://python.langchain.com/docs/integrations/vectorstores/faiss_async#saving-and-loading
    def init_store(self):

        # get index from cache
        index_local_path = get_vectorstore_directory(self.brain_id)
        if os.path.exists(index_local_path):
            store = FAISS.load_local(index_local_path, self.embedding)
        else:

            # get all files to embedding
            data, sources = get_knowledge_data(self.brain_id)
            text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
            docs = []
            # metadatas = []
            if "page_content" in data[0]:
                docs = text_splitter.split_documents(data)
            else:
                for i, d in enumerate(data):
                    splits = text_splitter.split_documents(d)
                    docs.extend(splits)
            logger.info("[Start Embedding] docs length: " + str(len(docs)))

                # metadatas.extend([{"source": sources[i]}] * len(splits))
            start_time = time.time()  # 获取当前时间

            store = FAISS.from_documents(docs, self.embedding)

            end_time = time.time()  # 再次获取当前时间
            elapsed_time = end_time - start_time  # 计算经过的时间

            # 持久化
            store.save_local(index_local_path)

            logger.info("[Complete Embedding] with time length: %s", int(elapsed_time))

        return store
