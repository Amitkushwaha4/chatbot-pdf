import time
import streamlit as st
from app.search_utils import search_internet
from app.summarization_utils import summarize_context
from app.chat_utils import ask_chat_model
from sentence_transformers import SentenceTransformer, util
import hashlib

# Load embedding model once
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def get_cache_key(prompt, vectorstore):
    """
    Generate a unique cache key based on:
    - user prompt
    - vectorstore state (documents uploaded or not)
    """
    doc_hash = "no_docs"
    if vectorstore:
        if vectorstore["type"] == "faiss":
            try:
                all_ids = "".join(vectorstore["store"].docstore._dict.keys())
                doc_hash = hashlib.md5(all_ids.encode()).hexdigest()
            except AttributeError:
                doc_hash = "no_docs"
        elif vectorstore["type"] == "mongo":
            docs = list(vectorstore["store"].collection.find({}, {"_id": 1}))
            if docs:
                all_ids = "".join([str(d["_id"]) for d in docs])
                doc_hash = hashlib.md5(all_ids.encode()).hexdigest()
    return f"{hashlib.md5(prompt.encode()).hexdigest()}_{doc_hash}"


def format_conversation(messages):
    """Format conversation history into Q&A style string."""
    return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages)


def build_system_prompt(history, docs, question):
    return f"""You are Chat Pro, an intelligent document assistant.
Here is the conversation history:
{history}

Based on the following documents/search results, answer the user's latest question:
Documents:
{docs}

User Question: {question}
Answer:"""


def get_response(chat_model, messages, vectorstore, prompt):
    """
    Document-first retrieval with web fallback:
    - Handles mid-conversation document uploads
    - Maintains previous chat history
    - Caches answers per prompt + document state
    """

    if 'query_cache' not in st.session_state:
        st.session_state.query_cache = {}

    # 1️⃣ Check if documents exist
    has_docs = False
    if vectorstore:
        if vectorstore["type"] == "faiss":
            has_docs = vectorstore["store"].index.ntotal > 0
        elif vectorstore["type"] == "mongo":
            has_docs = vectorstore["store"].collection.count_documents({}) > 0

    # 2️⃣ Generate cache key
    cache_key = get_cache_key(prompt, vectorstore)

    # 3️⃣ Return cached response if exists
    if cache_key in st.session_state.query_cache:
        cached_response, cached_time = st.session_state.query_cache[cache_key]
        return cached_response, cached_time

    start_time = time.time()
    summarized_chunks = []

    # 4️⃣ Retrieve document chunks first
    if has_docs:
        relevant_docs = []
        if vectorstore["type"] == "faiss":
            from app.vectorstore_utils import retrieve_relevant_docs_faiss
            relevant_docs = retrieve_relevant_docs_faiss(vectorstore["store"], prompt)
        elif vectorstore["type"] == "mongo":
            from app.vectorstore_utils import retrieve_relevant_docs_mongo
            relevant_docs = retrieve_relevant_docs_mongo(vectorstore["store"], prompt, k=3)

        # Summarize each chunk and keep source
        for doc in relevant_docs:
            text = doc.page_content.strip()
            source = doc.metadata.get("source", "from your document")
            if text:
                summary = summarize_context(text)
                summarized_chunks.append((summary, source))

    # 5️⃣ Only fallback to web search if no documents found
    if not summarized_chunks:
        search_results = search_internet(prompt)
        for r in search_results:
            text = f"{r['title']}: {r['snippet']}"
            source = r['link']

    # 6️⃣ Build system prompt
    context_text = "\n\n".join([f"{text} (Source: {source})" for text, source in summarized_chunks])
    history = format_conversation(messages[:-1])
    system_prompt = build_system_prompt(history, context_text, prompt)

    # 7️⃣ Call chat model
    response = ask_chat_model(chat_model, system_prompt)

    # 8️⃣ Append source attribution
    source_attribution = "Sources used in answer:\n" + "\n".join([f"- {source}" for _, source in summarized_chunks])
    response += f"\n\n---\n{source_attribution}"

    # 9️⃣ Cache response
    end_time = time.time()
    query_time = end_time - start_time
    st.session_state.query_cache[cache_key] = (response, query_time)

    return response, query_time
