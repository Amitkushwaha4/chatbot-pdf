from langchain.agents import Tool, initialize_agent
from app.vectorstore_utils import retrieve_relevant_docs
from app.tools_utils import get_search_tool, summarize_url, create_citations
import os

def create_multi_tool_agent(llm, vectorstore):
    # Define PDF RAG Tool
    def pdf_rag_tool(query: str):
        docs = retrieve_relevant_docs(vectorstore, query, k=3)
        return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant PDF content found."

    pdf_tool = Tool(
        name="PDF_RAG",
        func=pdf_rag_tool,
        description="Use this tool to answer questions from uploaded PDF documents."
    )

    # Web Search Tool
    search_api = get_search_tool()

    def search_and_summarize(query: str):
        results = search_api.results(query, num=3)
        summaries_with_urls = []
        for r in results:
            url = r.get("link")
            summary = summarize_url(url, llm)
            summaries_with_urls.append((summary, url))
        all_summaries = "\n\n".join([s[0] for s in summaries_with_urls])
        citations = create_citations(summaries_with_urls)
        return f"{all_summaries}\n\nSources:\n{citations}"

    web_tool = Tool(
        name="Web_Search_Summarizer",
        func=search_and_summarize,
        description="Use when the answer is not in PDFs and needs online search."
    )

    tools = [pdf_tool, web_tool]

    # Create Agent
    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True, handle_parsing_errors=True
    )
    return agent
