import streamlit as st
from app.ui import pdf_uploader
from app.pdf_utils import extract_text_from_pdf
from app.vectorstore_utils import (
    create_faiss_index,
    create_mongo_store,
    upsert_texts_to_mongo,
)
from app.chunking import chunk_pdf_texts
from app.chat_utils import get_chat_model
from app.chat_manager import get_response
from dotenv import load_dotenv
import time
import os

load_dotenv()

EURI_API_KEY = os.getenv("EURI_API_KEY", "")
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB = os.getenv("MONGO_DB", "")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "")

st.set_page_config(
    page_title="PDF document instructor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

#session setup
if "messages" not in st.session_state or st.session_state.messages is None:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_model" not in st.session_state:
    st.session_state.chat_model = None

if "store_type" not in st.session_state:
    st.session_state.store_type = "faiss"

if "timings" not in st.session_state or st.session_state.timings is None:
    st.session_state.timings = {"faiss": [], "mongo": []}

#header
st.markdown(
    """
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #ff4b4b; font-size: 3rem; margin-bottom: 0.5rem;"> PDF Pro</h1>
    <p style="font-size: 1.0rem; color: #666; margin-bottom: 1rem;"> Your intelligent Document Assistant (Week-2: MongoDB RAG)</p>
</div>
""",
    unsafe_allow_html=True,
)

#Sidebar: Settings & Upload
with st.sidebar:
    st.markdown("### Settings")
    store_choice = st.radio("Vector store", ["faiss (local)", "mongo (persistent)"], index=0)
    st.session_state.store_type = store_choice

    st.markdown("### Document Upload")
    uploaded_files = pdf_uploader()

    if store_choice.startswith("mongo"):
        st.markdown("**MongoDB settings**")
        st.text_input("Mongo URI", value=MONGO_URI, key="mongo_uri_input")
        st.text_input("Mongo DB", value=MONGO_DB, key="mongo_db_input")
        st.text_input("Mongo Collection", value=MONGO_COLLECTION, key="mongo_collection_input")
        st.markdown("Make sure your MongoDB is accessible from this machine (local or Atlas).")

    if uploaded_files:
        st.success(f"{len(uploaded_files)} document(s) selected")
        if st.button("Process Documents"):
            with st.spinner("Processing your documents..."):
                all_chunks, all_metadatas = chunk_pdf_texts(uploaded_files, extract_text_from_pdf)

                if st.session_state.store_type == "faiss (local)":
                    vectorstore = create_faiss_index(all_chunks)
                    st.session_state.vectorstore = {"type": "faiss", "store": vectorstore}
                    st.success("FAISS index created in-memory.")
                else:
                    mongo_uri = st.session_state.get("mongo_uri_input") or MONGO_URI
                    mongo_db = st.session_state.get("mongo_db_input") or MONGO_DB
                    mongo_collection = st.session_state.get("mongo_collection_input") or MONGO_COLLECTION
                    mongo_store = create_mongo_store(mongo_uri, mongo_db, mongo_collection)
                    mongo_store.clear_collection()
                    upsert_texts_to_mongo(mongo_store, all_chunks, all_metadatas)
                    st.session_state.vectorstore = {"type": "mongo", "store": mongo_store}
                    st.success(f"Uploaded {len(all_chunks)} chunks to MongoDB.")

                try:
                    chat_model = get_chat_model(EURI_API_KEY)
                    st.session_state.chat_model = chat_model
                except Exception as e:
                    st.error("Failed to initialize chat model. Check your EURI API KEY.")
                    st.exception(e)
                st.balloons()

    # Show average query times
    avg_faiss_time = sum(st.session_state.timings["faiss"])/len(st.session_state.timings["faiss"]) if st.session_state.timings["faiss"] else 0
    avg_mongo_time = sum(st.session_state.timings["mongo"])/len(st.session_state.timings["mongo"]) if st.session_state.timings["mongo"] else 0
    st.markdown("---")
    st.markdown("### Average Query Times")
    st.write(f"FAISS: {avg_faiss_time:.3f} s")
    st.write(f"MongoDB: {avg_mongo_time:.3f} s")

#chat interface
st.subheader("Chat with Your Documents")

# Ensure messages is always iterable
messages = st.session_state.messages if st.session_state.messages is not None else []

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.caption(message.get("timestamp", ""))

if prompt := st.chat_input("Ask about your documents..."):
    timestamp = time.strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(timestamp)

    if st.session_state.vectorstore and st.session_state.chat_model:
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                response, query_time = get_response(
                    st.session_state.chat_model,
                    st.session_state.messages,
                    st.session_state.vectorstore,
                    prompt
                )

                # Log timing
                st.session_state.timings[st.session_state.vectorstore["type"]].append(query_time)
                st.markdown(f"**Query time:** {query_time:.3f} seconds")

            st.markdown(response)
            st.caption(timestamp)
        st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": timestamp})
    else:
        with st.chat_message("assistant"):
            st.error("Please upload and process documents first!")
            st.caption(timestamp)

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; font-size: 0.9rem;">
<p> Powered by Euri AI & LangChain | Document Intelligence (Week-2)</p>
</div>
""",
    unsafe_allow_html=True,
)
