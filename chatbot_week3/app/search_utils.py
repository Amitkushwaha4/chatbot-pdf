import os
import requests
from dotenv import load_dotenv

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def search_internet(query: str, max_results: int = 5):
    """
    Search the internet using Tavily API and return a list of dicts with title, snippet, and link.
    """
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("results", []):
        results.append({
            "title": item.get("title", "No title"),
            "snippet": item.get("content", "No description"),
            "link": item.get("url", "#")
        })

    return results
