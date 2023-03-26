import webbrowser
import urllib.parse
import random
import re
# =============================================================================================================


def perform_search(query, engine=None):
    # List of search engines to choose from
    search_engines = [
        {
            "name": "Google",
            "url": "https://www.google.com/search?q="
        },
        {
            "name": "Bing",
            "url": "https://www.bing.com/search?q="
        },
        {
            "name": "DuckDuckGo",
            "url": "https://duckduckgo.com/?q="
        }
        # Add more search engines here
    ]

    # Check if the query is empty or None
    if not query:
        return "Error: Please enter a search query."

    # Check if the query contains any prohibited words or characters
    prohibited_words = ["javascript", "cookie", "alert",
                        "prompt", "confirm", "document", "location", "window"]
    pattern = "|".join(prohibited_words)
    if re.search(pattern, query, re.IGNORECASE):
        return "Error: The search query contains prohibited words or characters."

    # Check if the user specified a search engine
    specified_engine = None
    if engine:
        # Check if the specified engine is valid
        for engine_info in search_engines:
            if engine.lower() == engine_info['name'].lower():
                specified_engine = engine_info
                break
        # If the specified engine is not valid, select a random search engine
        if not specified_engine:
            specified_engine = random.choice(search_engines)

    # Select a random search engine if not specified
    if not specified_engine:
        specified_engine = random.choice(search_engines)

    # Encode the query
    encoded_query = urllib.parse.quote_plus(query)

    # Generate the search URL using the selected search engine
    search_url = specified_engine["url"] + encoded_query

    # Set the user agent string
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    headers = {"User-Agent": user_agent}

    # Open the search URL in the default web browser
    try:
        webbrowser.open(search_url, new=2, headers=headers)
    except webbrowser.Error:
        return "Error: Failed to open web browser."

    # Speak the response
    raven_speaker(
        f"Here are the search results from {specified_engine['name']} for {query}.")
# =============================================================================================================
