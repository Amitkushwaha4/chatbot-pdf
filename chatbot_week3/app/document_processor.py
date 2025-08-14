import streamlit as st
from app.pdf_utils import extract_text_from_pdf
from app.chunking import chunk_pdf_texts
from app.vectorstore_utils import create_faiss_index, create_mongo_store, upsert_texts_to_mongo

def process_uploaded_documents(uploaded_files):
    all_chunks, all_metadatas = chunk_pdf_texts(uploaded_files, extract_text_from_pdf)

    if st.session_state.store_type == "faiss":
        vectorstore = create_faiss_index(all_chunks)
        st.session_state.vectorstore = {"type": "faiss", "store": vectorstore}
        st.success("FAISS index created in-memory.")
    else:
        mongo_uri = st.session_state.get("mongo_uri_input")
        mongo_db = st.session_state.get("mongo_db_input")
        mongo_collection = st.session_state.get("mongo_collection_input")
        mongo_store = create_mongo_store(mongo_uri, mongo_db, mongo_collection)
        mongo_store.clear_collection()
        upsert_texts_to_mongo(mongo_store, all_chunks, all_metadatas)
        st.session_state.vectorstore = {"type": "mongo", "store": mongo_store}
        st.success(f"Uploaded {len(all_chunks)} chunks to MongoDB.")

    st.balloons()
