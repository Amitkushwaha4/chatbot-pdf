from app.chat_utils import ask_chat_model, get_chat_model

def summarize_context(context: str) -> str:
    chat_model = get_chat_model()
    summary_prompt = (
        f"Summarize the following documents/search results into a concise, clear answer:\n\n{context}"
    )
    return ask_chat_model(chat_model, summary_prompt)
