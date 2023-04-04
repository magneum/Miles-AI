import wikipedia
import re


def wikipedia_search(usersaid):
    regex_wikipedia = r"(?P<wikipedia>wikipedia|wiki) (?P<search_query>.+)"
    match = re.match(regex_wikipedia, usersaid)
    if match:
        search_query = match.group("search_query")
        wiki = wikipedia.Wikipedia("en")

        try:
            page = wiki.page(search_query)
            if page.exists():
                if hasattr(page, "summary"):
                    summary = page.summary[0:500] + "..."
                    return f"Here's what I found on Wikipedia: {summary}"
                else:
                    return f"No summary available for {search_query}."
            else:
                return (
                    f"Sorry, I could not find anything on Wikipedia for {search_query}"
                )
        except Exception as e:
            return f"An error occurred while searching on Wikipedia: {str(e)}"
