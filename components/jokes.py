import requests
import random
import re


def get_joke(usersaid):
    joke_categories = [
        "any",
        "programming",
        "knock-knock",
        "dad",
        "miscellaneous",
        "animal",
        "punny",
        "one-liner",
        "blonde",
        "doctor",
        "food",
        "school",
        "travel",
    ]

    category_match = re.search(
        r"(" + "|".join(joke_categories) + r")\s+joke\b", usersaid
    )
    if category_match:
        category = category_match.group(1)
    else:
        category = random.choice(joke_categories)
    url = f"https://v2.jokeapi.dev/joke/{category}?type=single"
    response = requests.get(url)
    joke_data = response.json()
    if joke_data["type"] == "single":
        return joke_data["joke"]
    else:
        return "Sorry, I couldn't find a joke for that category."
