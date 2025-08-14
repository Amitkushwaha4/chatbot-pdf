import streamlit as st
from app.document_processor import process_uploaded_documents
import os

def pdf_uploader():
    return st.file_uploader(
        'Upload PDF files',
        type='pdf',
        accept_multiple_files=True,
        help="Upload one or more PDF documents"
    )

def render_sidebar():
    st.markdown("### Settings")
    store_choice = st.radio("Vector store", ["faiss", "mongo"], index=0)
    st.session_state.store_type = store_choice

    st.markdown("### Document Upload")
    uploaded_files = pdf_uploader()

    if store_choice == "mongo":
        st.text_input("Mongo URI", value=os.getenv("MONGO_URI", ""), key="mongo_uri_input")
        st.text_input("Mongo DB", value=os.getenv("MONGO_DB", ""), key="mongo_db_input")
        st.text_input("Mongo Collection", value=os.getenv("MONGO_COLLECTION", ""), key="mongo_collection_input")

    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing your documents..."):
            process_uploaded_documents(uploaded_files)

    avg_faiss = _get_avg_time("faiss")
    avg_mongo = _get_avg_time("mongo")
    st.markdown("### Average Query Times")
    st.write(f"FAISS: {avg_faiss:.3f} s")
    st.write(f"MongoDB: {avg_mongo:.3f} s")

def _get_avg_time(store):
    times = st.session_state.timings.get(store, [])
    return sum(times) / len(times) if times else 0
