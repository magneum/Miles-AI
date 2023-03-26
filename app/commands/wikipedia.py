import wikipediaapi


def handle_wikipedia(usersaid):
    # Define regex pattern to match user input for searching on Wikipedia
    regex_wikipedia = r"(?P<wikipedia>wikipedia|wiki) (?P<search_query>.+)"

    # Match user input to Wikipedia regex pattern
    match = re.match(regex_wikipedia, usersaid)

    # If the user input matches the Wikipedia regex pattern, extract the search query and search on Wikipedia
    if match:
        # Extract search query
        search_query = match.group("search_query")

        # Initialize Wikipedia API object
        wiki = wikipediaapi.Wikipedia('en')

        # Search for the given query
        page = wiki.page(search_query)

        # If a page is found, extract the summary and speak it
        if page.exists():
            summary = page.summary[0:500] + "..."
            raven_speaker(f"Here's what I found on Wikipedia: {summary}")
        # If no page is found, notify the user
        else:
            raven_speaker(
                f"Sorry, I could not find anything on Wikipedia for {search_query}")
