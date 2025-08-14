from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from app.mongo_vectorstore import MongoVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Read environment variables for MongoDB
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

# FAISS helpers (storing)
def create_faiss_index(texts: List[str]):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_texts(texts, embeddings)

# retrieving
def retrieve_relevant_docs_faiss(vectorstore, query: str, k: int = 3):
    return vectorstore.similarity_search(query, k=k)

# MongoDB-backed vector store helpers
def create_mongo_store(mongo_uri: str = None, db_name: str = None, collection_name: str = None):
    mongo_uri = mongo_uri or MONGO_URI
    db_name = db_name or MONGO_DB
    collection_name = collection_name or MONGO_COLLECTION
    return MongoVectorStore(mongo_uri, db_name, collection_name, embedding_model_name=EMBEDDING_MODEL)

def upsert_texts_to_mongo(mongo_store: MongoVectorStore, texts: List[str], metadatas: List[dict] = None):
    return mongo_store.upsert_texts(texts, metadatas)

def retrieve_relevant_docs_mongo(mongo_store: MongoVectorStore, query: str, k: int = 3):
    # returns list of dicts { text, metadata, score }
    return mongo_store.retrieve(query, k=k)
