import requests
import random
# =============================================================================================================


def get_joke(usersaid):
    # Define the available joke categories
    joke_categories = ["any", 'programming', 'knock-knock', 'dad', "miscellaneous",
                       'animal', 'punny', 'one-liner', 'blonde', 'doctor', 'food', 'school', 'travel']

    # Find the joke category in the user's input
    category_match = re.search(
        r"(" + "|".join(joke_categories) + r")\s+joke\b", usersaid)
    if category_match:
        # Get the specified joke category
        category = category_match.group(1)
    else:
        # If no category is specified, choose a random one
        category = random.choice(joke_categories)
    # Build the API URL for the selected category
    url = f"https://v2.jokeapi.dev/joke/{category}?type=single"
    # Send a GET request to the API URL
    response = requests.get(url)
    # Parse the response as JSON
    joke_data = response.json()
    # Check if the API returned a joke
    if joke_data["type"] == "single":
        # Speak the joke
        miles_speaker(joke_data["joke"])
    else:
        # If no joke is returned, inform the user
        miles_speaker("Sorry, I couldn't find a joke for that category.")
# =============================================================================================================
