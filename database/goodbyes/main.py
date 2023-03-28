import re
import json
import random
import datetime

# =============================================================================================================
goodbyes = r"bye\b|goodbye\b|farewell\b|see you\b|take care\b|cheerio\b|ciao\b|so long\b|until next time\b|peace out\b|later\badios\b|au revoir\b|bye for now\b|catch you later\b|have a good one\b|keep in touch\b|leaving now\b|parting ways\b|so farewell\b|stay safe\b|till we meet again\b"
casual_response = json.load(open("database/goodbyes/Casual.json"))["response"]
formal_response = json.load(open("database/goodbyes/Formal.json"))["response"]
unique_response = json.load(open("database/goodbyes/Unique.json"))["response"]
friendly_response = json.load(
    open("database/goodbyes/Friendly.json"))["response"]
informal_response = json.load(
    open("database/goodbyes/Informal.json"))["response"]
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


def generate_goodbyes_response(user_input):
    convoness = {
        r"bye\b|goodbye\b|farewell\b|see you\b|take care\b|cheerio\b|ciao\b|so long\b|until next time\b|peace out\b|later\badios\b|au revoir\b|bye for now\b|catch you later\b|have a good one\b|keep in touch\b|leaving now\b|parting ways\b|so farewell\b|stay safe\b|till we meet again\b":
        random_list(casual_response, formal_response,
                    friendly_response, informal_response, unique_response),

    }
    for pattern, response in convoness.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            output = random.choice(response)
            return output.format(provide_time())
    return "I'm sorry, I don't understand what you're asking."
# =============================================================================================================
