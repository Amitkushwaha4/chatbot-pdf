import streamlit as st
from dotenv import load_dotenv
from app.state import init_session_state
from app.ui import render_sidebar
from app.chat_interface import render_chat

load_dotenv()
st.set_page_config(page_title="PDF Pro", layout="wide", initial_sidebar_state="expanded")

init_session_state()

# Header
st.markdown("""
<div style="text-align: center;">
<h1 style="color: #ff4b4b;">PDF Pro</h1>
<p>Your intelligent Document Assistant (Week-2: MongoDB RAG)</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    render_sidebar()

# Chat interface
render_chat()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666;'>Powered by Euri AI & LangChain | Week-3</div>",
    unsafe_allow_html=True
)
