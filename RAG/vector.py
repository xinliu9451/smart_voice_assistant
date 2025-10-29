from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# 基准目录：固定为当前文件所在目录，避免受工作目录影响
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据文件与持久化目录均相对于 RAG 目录
csv_path = os.path.join(BASE_DIR, "realistic_restaurant_reviews.csv")
# df = pd.read_csv(csv_path)
df = pd.read_csv(csv_path, encoding="gbk")
embeddings = OllamaEmbeddings(model="bge-m3")

db_location = os.path.join(BASE_DIR, "chrome_langchain_db")
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)