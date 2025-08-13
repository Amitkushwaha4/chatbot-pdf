import time

def format_conversation(messages):
    """Format conversation history into a Q&A style string."""
    convo = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            convo += f"User: {content}\n"
        elif role == "assistant":
            convo += f"Assistant: {content}\n"
    return convo


def build_system_prompt(history, docs, question):
    """
    Build the prompt for the chat model based on conversation history,
    retrieved documents, and the latest user question.
    """
    return f"""You are Chat Pro, an intelligent document assistant.
Here is the conversation history:
{history}

Based on the following documents, answer the user's latest question:
Documents:
{docs}

User Question: {question}
Answer:"""


def get_response(chat_model, messages, vectorstore, prompt):
    """
    Handles the workflow:
    - Retrieves relevant documents
    - Measures query time
    - Builds prompt
    - Calls the chat model
    """
    start = time.time()

    # Retrieve context from vectorstore
    if vectorstore["type"] == "faiss":
        from app.vectorstore_utils import retrieve_relevant_docs_faiss
        relevant_docs = retrieve_relevant_docs_faiss(vectorstore["store"], prompt)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
    else:
        from app.vectorstore_utils import retrieve_relevant_docs_mongo
        mongo_store = vectorstore["store"]
        results = retrieve_relevant_docs_mongo(mongo_store, prompt, k=3)
        context = "\n\n".join([r["text"] for r in results])

    end = time.time()
    query_time = end - start

    history = format_conversation(messages[:-1])
    system_prompt = build_system_prompt(history, context, prompt)

    # Ask the chat model
    from app.chat_utils import ask_chat_model
    response = ask_chat_model(chat_model, system_prompt)

    return response, query_time
