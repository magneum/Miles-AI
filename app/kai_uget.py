from .kai_command import kai_command
from .kai_speaker import kai_speaker
import os
import json
import random
import requests
import webbrowser
import openai as assistant
from termcolor import cprint
from colorama import Fore, Style
current_dir = os.path.dirname(__file__)


KAI_Responses = json.load(open("database/responses.json"))

greetings = r"hi\b|hello\b|hey\b|greetings\b|salutations\b|yo\b|hiya\b|howdy\bsup\b|hi there\b|hello there\b|what's up\b|yoohoo\b|hey there\b|hiya there\b|g'day\b|cheerio\b|hihi\b|aloha\b|bonjour\b|hallo\b|ciao\b|namaste\b|konichiwa\b|hola\b|szia\b|hei\b|hej\b|tjena\b|heya\b|hey ya\b|sup dude\b|sup bro\b|sup everyone\b|wassup\b|whaddup\b"


def kai_uget():
    while True:
        usersaid = kai_command()
        if usersaid in greetings:
            kai_speaker(random.choice(KAI_Responses["hello"]["responses"]))
            break
        elif "shutdown" in usersaid:
            kai_speaker(random.choice(KAI_Responses["shutdown"]["responses"]))
            os.system("shutdown /s /t 1")
        elif "play" in usersaid:
            try:
                songname = usersaid.split("play", 1)[1]
                api = requests.get(
                    f"https://magneum.vercel.app/api/youtube_sr?q={songname}")
                name = api.json()["youtube_search"][0]["TITLE"]
                kai_speaker(f"Playing {name} on youtube browser.")
                webbrowser.open(
                    api.json()["youtube_search"][0]["LINK"], new=2)
            except Exception as e:
                kai_speaker(f"Sorry could not play {songname} on youtube.")
                break
        else:
            try:
                assistant.api_key = "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"
                response = assistant.Completion.create(
                    engine="text-davinci-003",
                    prompt=usersaid.capitalize(),
                    temperature=1,
                    max_tokens=4000,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                resp = response["choices"][0]["text"].capitalize()
                print(f"{Fore.GREEN}ҠΛI: {Style.RESET_ALL}{resp}")
                kai_speaker(resp)
                break
            except Exception as e:
                print(f"{Fore.RED}ҠΛI: {Style.RESET_ALL}Sorry, did not get that.")
                cprint(f": {e}", "white", "on_grey", attrs=[])
                kai_speaker(f"Sorry, did not get that.")
                break
