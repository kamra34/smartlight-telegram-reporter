import os
import requests
from dotenv import load_dotenv

# Load API keys
load_dotenv()
CX = "f56cdb28d79fb4ff9"
API_KEY = "AIzaSyBIJLlt0IVAk5DmCJqH2wrJ3EhnX5BeQrE"

def google_search(query, num=5):
    """Simple Google Custom Search"""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CX,
        "q": query,
        "num": num
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()  # Raise error if failed
    data = resp.json()

    results = []
    for item in data.get("items", []):
        results.append({
            "title": item["title"],
            "link": item["link"],
            "snippet": item["snippet"]
        })
    return results

if __name__ == "__main__":
    query = "Smart home lighting design 2025"
    print(f"Searching for: {query}\n")
    results = google_search(query)
    for r in results:
        print(f"{r['title']}\n{r['link']}\n{r['snippet']}\n")
