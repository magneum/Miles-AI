import re
import json
import random
import datetime


# =============================================================================================================
feelings = r"\bhow\s+(?:are\s+you|are\s+you\s+doing|do\s+you\s+feel|have\s+you\s+been)\s+(?:feeling|today|lately|these\s+days)\b"
affection_response = json.load(
    open("database/feelings/Affection.json"))["response"]
anxiety_response = json.load(
    open("database/feelings/Anxiety.json"))["response"]
excitement_response = json.load(
    open("database/feelings/Excitement.json"))["response"]
negative_response = json.load(
    open("database/feelings/Negative.json"))["response"]
positive_response = json.load(
    open("database/feelings/Positive.json"))["response"]
# =============================================================================================================


def provide_time():
    now = datetime.datetime.now()
    hour = now.hour
    if hour >= 12 and hour < 18:
        return "Good Afternoon"
    elif hour >= 18 or hour < 6:
        return "Good Evening"
    else:
        return "Good Morning"
# =============================================================================================================


def random_list(*args):
    selected_list = random.choice(args)
    return selected_list
# =============================================================================================================


def generate_feelings_response(user_input):
    convoness = {
        r"\bhow\s+(?:are\s+you|are\s+you\s+doing|do\s+you\s+feel|have\s+you\s+been)\s+(?:feeling|today|lately|these\s+days)\b":
        random_list(affection_response, anxiety_response,
                    excitement_response, negative_response, positive_response),

    }
    for pattern, response in convoness.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            output = random.choice(response)
            return output.format(provide_time())
    return "I'm sorry, I don't understand what you're asking."
# =============================================================================================================
