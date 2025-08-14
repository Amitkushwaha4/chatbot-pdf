import streamlit as st
import time
from app.chat_manager import get_response

def render_chat():
    st.subheader("Chat with Your Documents (or the Web)")

    # Display existing conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.caption(message.get("timestamp", ""))

    # Chat input box
    if prompt := st.chat_input("Ask a question..."):
        _handle_user_input(prompt)


def _handle_user_input(prompt):
    timestamp = time.strftime("%H:%M")

    # Add user message to history immediately
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })

    # Show user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(timestamp)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Check cache first
            if prompt in st.session_state.query_cache:
                response, query_time = st.session_state.query_cache[prompt]
            else:
                # Pass full history (including previous assistant messages)
                response, query_time = get_response(
                    st.session_state.chat_model,
                    st.session_state.messages,  # Now includes all past conversation
                    st.session_state.vectorstore,
                    prompt
                )
                st.session_state.query_cache[prompt] = (response, query_time)

            # Record query time per store type
            if st.session_state.vectorstore:
                store_type = st.session_state.vectorstore["type"]
                st.session_state.timings[store_type].append(query_time)

            # Display assistant response
            st.markdown(f"**Query time:** {query_time:.3f} seconds")
            st.markdown(response)
            st.caption(timestamp)

    # Add assistant message to history after showing it
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "timestamp": timestamp
    })
