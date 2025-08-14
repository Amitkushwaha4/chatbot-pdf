import streamlit as st
from app.chat_utils import get_chat_model
import os

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_cache" not in st.session_state:
        st.session_state.query_cache = {}
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = get_chat_model(os.getenv("EURI_API_KEY", ""))
    if "store_type" not in st.session_state:
        st.session_state.store_type = "faiss"
    if "timings" not in st.session_state:
        st.session_state.timings = {"faiss": [], "mongo": []}
