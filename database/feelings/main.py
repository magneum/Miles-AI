import re
import json
import random
import datetime


# ================================================================= DATA-LISTS FOR GREETINGS =================================================================
casual_response = json.load(open("database/feelings/Casual.json"))["response"]
formal_response = json.load(open("database/feelings/Formal.json"))["response"]
unique_response = json.load(open("database/feelings/Unique.json"))["response"]
friendly_response = json.load(
    open("database/feelings/Friendly.json"))["response"]
informal_response = json.load(
    open("database/feelings/Informal.json"))["response"]


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
        random_list(casual_response, formal_response,
                    friendly_response, informal_response, unique_response),

    }
    for pattern, response in convoness.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            output = random.choice(response)
            return output.format(provide_time())
    return "I'm sorry, I don't understand what you're asking."
