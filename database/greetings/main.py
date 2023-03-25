import re
import json
import random
import datetime


# ================================================================= DATA-LISTS FOR GREETINGS =================================================================
casual_response = json.load(open("database/greetings/Casual.json"))["response"]
formal_response = json.load(open("database/greetings/Formal.json"))["response"]
friendly_response = json.load(
    open("database/greetings/Friendly.json"))["response"]
informal_response = json.load(
    open("database/greetings/Informal.json"))["response"]
unique_response = json.load(open("database/greetings/Unique.json"))["response"]


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
def generate_greeting_response(user_input):
    convoness = {
        r"hi\b|hello\b|hey\b|greetings\b|salutations\b|yo\b|hiya\b|howdy\bsup\b|hi there\b|hello there\b|what's up\b|yoohoo\b|hey there\b|hiya there\b|g'day\b|cheerio\b|hihi\b|aloha\b|bonjour\b|hallo\b|ciao\b|namaste\b|konichiwa\b|hola\b|szia\b|hei\b|hej\b|tjena\b|heya\b|hey ya\b|sup dude\b|sup bro\b|sup everyone\b|wassup\b|whaddup\b":
        random_list(casual_response,
                    formal_response,
                    friendly_response,
                    informal_response,
                    unique_response),

    }
    for pattern, response in convoness.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            output = random.choice(response)
            return output.format(provide_time())
    return "I'm sorry, I don't understand what you're asking."
