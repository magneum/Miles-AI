from app.miles_speaker import miles_speaker
# =============================================================================================================


import requests
import json
import random
from bs4 import BeautifulSoup
from translate import Translator


def get_news(usersaid):
    categories = ["business", "entertainment", "general",
                  "health", "science", "sports", "technology"]
    base_url = "https://newsapi.org/v2/top-headlines"
    api_key = "860b08c7c72841e6a8a53a3c3cfa8ec6"  # Replace with your News API key

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
        "country": "in",  # set default location to India
        "category": category,
        "pageSize": 4  # number of articles to retrieve
    }

    response = requests.get(base_url, params=params)
    data = json.loads(response.text)

    if data["status"] == "ok":
        articles = data["articles"]
        if len(articles) > 0:
            for article in articles:
                title = article["title"]
                url = article["url"]

                # make a separate request to the article URL to get the full content
                article_response = requests.get(url)
                soup = BeautifulSoup(article_response.text, 'html.parser')

                # extract the article content from the HTML using Beautiful Soup
                content = soup.find('div', class_='article-body')
                if content:
                    news_text = f"{title}\n{content.get_text()}"
                else:
                    news_text = title

                # translate the news_text to the target language
                translator = Translator(to_lang='en')
                news_text = translator.translate(news_text)

                print(news_text)
                miles_speaker(news_text)

        else:
            print("No articles found.")
            miles_speaker("No articles found.")
    else:
        print("An error occurred while retrieving the news.")
        miles_speaker("An error occurred while retrieving the news.")
