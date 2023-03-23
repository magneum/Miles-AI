import os
import re
import json
import random
import datetime
import requests
from flask import Flask
from colorama import Fore, Style
# os.system("python3 -m app")


# api_key = "860b08c7c72841e6a8a53a3c3cfa8ec6"
# response = requests.get(
#     f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}")
# data = response.json()
# articles = data["articles"]
# for article in articles:
#     print(article["title"])


# api_url = f"http://localhost:3000/news"
# response = requests.get(api_url)
# data = response.json()
# print(f"> {Fore.GREEN}TITLE: {Style.RESET_ALL}" + data["title"])
# print(f"> {Fore.GREEN}AUTHOR: {Style.RESET_ALL}" + data["author"])
# print(f"> {Fore.GREEN}DESCRIPTION: {Style.RESET_ALL}" + data["description"])
# print(f"> {Fore.GREEN}PUBLISHEDAT: {Style.RESET_ALL}" + data["publishedAt"])
# print(f"> {Fore.GREEN}CONTENT: {Style.RESET_ALL}" + data["content"])
# print(f"> {Fore.GREEN}URL: {Style.RESET_ALL}" + data["url"])
# print(f"> {Fore.GREEN}IMAGE: {Style.RESET_ALL}" + data["urlToImage"])


# api_url = f"http://localhost:3000/weather"
# response = requests.get(api_url)
# data = response.json()

def provide_time():
    # Determine whether it is afternoon, night or morning
    now = datetime.datetime.now()
    hour = now.hour
    if hour >= 12 and hour < 18:
        return "Good Afternoon"
    elif hour >= 18 or hour < 6:
        return "Good Evening"
    else:
        return "Good Morning"
    return "none"


def random_list(*args):
    # select a random list from the input lists
    selected_list = random.choice(args)
    return selected_list


def generate_response(user_input):
    # Define a function to generate a response
    convoness = {
        r"hi\b|hello\b|hey\b|greetings\b|salutations\b|yo\b|hiya\b|howdy\bsup\b|hi there\b|hello there\b|what's up\b|yoohoo\b|hey there\b|hiya there\b|g'day\b|cheerio\b|hihi\b|aloha\b|bonjour\b|hallo\b|ciao\b|namaste\b|konichiwa\b|hola\b|szia\b|hei\b|hej\b|tjena\b|heya\b|hey ya\b|sup dude\b|sup bro\b|sup everyone\b|wassup\b|whaddup\b":
        random_list(json.load(open("database/greetings/Casual.json"))["response"], json.load(open("database/greetings/Formal.json"))["response"], json.load(open(
            "database/greetings/Friendly.json"))["response"], json.load(open("database/greetings/Informal.json"))["response"], json.load(open("database/greetings/Unique.json"))["response"]),

    }
    for pattern, response in convoness.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            output = random.choice(response)
            return output.format(provide_time())
    return "I'm sorry, I don't understand what you're asking."
