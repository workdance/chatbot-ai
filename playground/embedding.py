from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = PyPDFLoader("../temp/haomo/阿里巴巴财报 CY23Q4.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OllamaEmbeddings(
)
db = FAISS.from_documents(docs, embeddings)

query = "阿里巴巴的四季度利润"
docs = db.similarity_search(query)
print(docs[0].page_content)