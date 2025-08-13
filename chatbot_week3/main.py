import streamlit as st
import time
import os
from app.ui import pdf_uploader
from app.pdf_utils import extract_text_from_pdf
from app.vectorstore_utils import create_faiss_index
from app.chat_utils import get_chat_model
from app.agent_utils import create_multi_tool_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()
EURI_API_KEY = os.getenv("EURI_API_KEY", "")
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB = os.getenv("MONGO_DB", "")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "")

# ====== STREAMLIT PAGE CONFIG ======
st.set_page_config(
    page_title="PDF Pro",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== CUSTOM CSS ======
st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.chat-message.user {
    background-color: #2b313e;
    color: white;
}
.chat-message.assistant {
    background-color: #f0f2f6;
    color: black;
}
.stButton > button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 0.5rem;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #ff3333;
}
.upload-selection {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ====== SESSION STATE INIT ======
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_model" not in st.session_state:
    st.session_state.chat_model = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

# ====== HEADER ======
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #ff4b4b; font-size: 3rem; margin-bottom: 0.5rem;">PDF Pro</h1>
    <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Your Intelligent Document & Research Assistant</p>
</div>
""", unsafe_allow_html=True)

# ====== SIDEBAR: PDF UPLOAD ======
with st.sidebar:
    st.markdown("### Document Upload")
    uploaded_files = pdf_uploader()

    if uploaded_files:
        st.success(f"{len(uploaded_files)} document(s) uploaded")

        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing your documents..."):
                # 1. Extract all text
                all_texts = []
                for file in uploaded_files:
                    text = extract_text_from_pdf(file)
                    all_texts.append(text)

                # 2. Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                chunks = []
                for text in all_texts:
                    chunks.extend(text_splitter.split_text(text))

                # 3. Create FAISS index (PDF RAG)
                vectorstore = create_faiss_index(chunks)
                st.session_state.vectorstore = vectorstore

                # 4. Initialize Chat Model
                chat_model = get_chat_model(EURI_API_KEY)
                st.session_state.chat_model = chat_model

                # 5. Create Multi-Tool Agent
                st.session_state.agent = create_multi_tool_agent(chat_model, vectorstore)

                st.success("Documents processed successfully!")
                st.balloons()

# ====== MAIN CHAT ======
st.markdown("### Chat About Your Documents or Search the Web")

# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.caption(message["timestamp"])

# Chat input
if prompt := st.chat_input("Ask about your PDFs or do research online..."):
    timestamp = time.strftime("%H:%M")
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})

    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(timestamp)

    # Generate answer
    if st.session_state.agent:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.run(prompt)
                except Exception as e:
                    response = f"Error: {e}"
            st.markdown(response)
            st.caption(timestamp)

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": timestamp})
    else:
        with st.chat_message("assistant"):
            st.error("Please upload and process documents first!")
            st.caption(timestamp)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.memory.clear() 

# ====== FOOTER ======
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
<p>Powered by Euri AI, LangChain, and Multi-Tool Agents</p>
</div>
""", unsafe_allow_html=True)
