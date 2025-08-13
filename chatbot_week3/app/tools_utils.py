from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms import OpenAI
import os

# Search Tool
def get_search_tool():
    return GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))

# Summarizer Tool
def summarize_url(url, llm):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    prompt_template = """Provide a concise summary of the following content while preserving key facts.
    Content:
    {text}
    Summary with bullet points and clear structure:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt)
    return chain.run(split_docs)

# Citation Tracker
def create_citations(summaries_with_urls):
    citations = []
    for i, (summary, url) in enumerate(summaries_with_urls, start=1):
        citations.append(f"[{i}] {url}")
    return "\n".join(citations)
