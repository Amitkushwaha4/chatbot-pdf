from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

def create_faiss_index(texts: List[str], index_path: str):
    """Creates and saves a FAISS index from a list of texts."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

def retrieve_relevant_docs(query: str, index_path: str, k: int = 5) -> List[str]:
    """Retrieves the top-k most relevant documents for a query from a FAISS index."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(query, k=k)
    return docs
