import webbrowser
import urllib.parse
import random
import re


def search(query, engine=None):
    search_engines = [
        {"name": "Google", "url": "https://www.google.com/search?q="},
        {"name": "Bing", "url": "https://www.bing.com/search?q="},
        {"name": "DuckDuckGo", "url": "https://duckduckgo.com/?q="},
    ]
    if not query:
        return "Error: Please enter a search query."
    prohibited_words = [
        "javascript",
        "cookie",
        "alert",
        "prompt",
        "confirm",
        "document",
        "location",
        "window",
    ]
    pattern = "|".join(prohibited_words)
    if re.search(pattern, query, re.IGNORECASE):
        return "Error: The search query contains prohibited words or characters."
    specified_engine = None
    if engine:
        for engine_info in search_engines:
            if engine.lower() == engine_info["name"].lower():
                specified_engine = engine_info
                break
        if not specified_engine:
            specified_engine = random.choice(search_engines)
    if not specified_engine:
        specified_engine = random.choice(search_engines)
    encoded_query = urllib.parse.quote_plus(query)
    search_url = specified_engine["url"] + encoded_query
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    headers = {"User-Agent": user_agent}
    try:
        webbrowser.open(search_url, new=2, headers=headers)
    except webbrowser.Error:
        return "Error: Failed to open web browser."
    return f"Here are the search results from {specified_engine['name']} for {query}."
