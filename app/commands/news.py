import requests
import json
import random
# =============================================================================================================


def get_news(usersaid):
    categories = ["business", "entertainment", "general",
                  "health", "science", "sports", "technology"]
    base_url = "https://newsapi.org/v2/top-headlines"
    api_key = "<YOUR_API_KEY>"  # Replace with your News API key

    category = None
    for c in categories:
        if c in usersaid:
            category = c
            break

    if not category:
        category = random.choice(categories)  # select a random category:
        # category = "general"  # or set default category if none provided

    params = {
        "apiKey": api_key,
        "category": category,
        "pageSize": 5  # number of articles to retrieve
    }

    response = requests.get(base_url, params=params)
    data = json.loads(response.text)

    if data["status"] == "ok":
        articles = data["articles"]
        if len(articles) > 0:
            for article in articles:
                title = article["title"]
                description = article["description"]
                source = article["source"]["name"]
                url = article["url"]
                news_text = f"{title}\n{description}\n{source}\n{url}\n"
                print(news_text)
                raven_speaker(news_text)
        else:
            print("No articles found.")
            raven_speaker("No articles found.")
    else:
        print("An error occurred while retrieving the news.")
        raven_speaker("An error occurred while retrieving the news.")
# =============================================================================================================
