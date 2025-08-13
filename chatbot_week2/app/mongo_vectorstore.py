from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
import numpy as np
import os


class MongoVectorStore:
    def __init__(self, uri: str, db_name: str, collection_name: str, embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model_name)

    def clear_collection(self):
        self.collection.delete_many({})

    def upsert_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Compute embeddings and insert documents. Overwrites collection (append behavior is possible)."""
        metadatas = metadatas or [{} for _ in texts]
        embeddings = self.embedder.embed_documents(texts)  # returns List[List[float]]
        docs = []
        for i, text in enumerate(texts):
            doc = {
                "text": text,
                "metadata": metadatas[i],
                "embedding": embeddings[i]
            }
            docs.append(doc)
        if docs:
            self.collection.insert_many(docs)
        return len(docs)

    def _cosine_similarities(self, query_emb, embeddings_np):
        # query_emb: (d,), embeddings_np: (n,d)
        # return cosine similarities as (n,)
        q = query_emb / (np.linalg.norm(query_emb) + 1e-12)
        embs = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-12)
        sims = embs.dot(q)
        return sims

    def retrieve(self, query: str, k: int = 3):
        """Return top-k documents (text, metadata) relevant to query."""
        # compute query embedding
        query_emb = np.array(self.embedder.embed_query(query), dtype=float)
        # load all candidate embeddings and ids
        docs_cursor = list(self.collection.find({}, {"text": 1, "metadata": 1, "embedding": 1}))
        if not docs_cursor:
            return []

        embeddings = np.array([doc["embedding"] for doc in docs_cursor], dtype=float)
        sims = self._cosine_similarities(query_emb, embeddings)
        topk_idx = np.argsort(-sims)[:k]

        results = []
        for idx in topk_idx:
            d = docs_cursor[int(idx)]
            results.append({
                "text": d["text"],
                "metadata": d.get("metadata", {}),
                "score": float(sims[int(idx)])
            })
        return results
