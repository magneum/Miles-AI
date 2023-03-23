import re
import json
import random
import datetime


# ================================================================= DATA-LISTS FOR GREETINGS =================================================================
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


# =============================================== DETERMINE WHETHER IT IS AFTERNOON, NIGHT OR MORNING ===============================================
def provide_time():
    now = datetime.datetime.now()
    hour = now.hour
    if hour >= 12 and hour < 18:
        return "Good Afternoon"
    elif hour >= 18 or hour < 6:
        return "Good Evening"
    else:
        return "Good Morning"


#  ============================================================ SELECT A RANDOM LIST FROM THE INPUT LISTS ============================================================
def random_list(*args):
    selected_list = random.choice(args)
    return selected_list


#  ============================================================ DEFINE A FUNCTION TO GENERATE A RESPONSE ============================================================
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
