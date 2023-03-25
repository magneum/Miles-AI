import os
import wave
import json
import random
import openai
import logging
import struct
import pyttsx3
import asyncio
import pyaudio
import traceback
import simpleaudio
import pvporcupine
from termcolor import cprint
from dotenv import load_dotenv
import speech_recognition as sr
from colorama import Fore, Style
from .raven_command import raven_command
from .raven_speaker import raven_speaker
from .generate_open_response import generate_open_response

from database.feelings import generate_feelings_response
from database.goodbyes import generate_goodbyes_response
from database.greetings import generate_greeting_response


greetings = r"hi\b|hello\b|hey\b|greetings\b|salutations\b|yo\b|hiya\b|howdy\bsup\b|hi there\b|hello there\b|what's up\b|yoohoo\b|hey there\b|hiya there\b|g'day\b|cheerio\b|hihi\b|aloha\b|bonjour\b|hallo\b|ciao\b|namaste\b|konichiwa\b|hola\b|szia\b|hei\b|hej\b|tjena\b|heya\b|hey ya\b|sup dude\b|sup bro\b|sup everyone\b|wassup\b|whaddup\b"
goodbyes = r"bye\b|goodbye\b|farewell\b|see you\b|take care\b|cheerio\b|ciao\b|so long\b|until next time\b|peace out\b|later\badios\b|au revoir\b|bye for now\b|catch you later\b|have a good one\b|keep in touch\b|leaving now\b|parting ways\b|so farewell\b|stay safe\b|till we meet again\b"
feelings = r"\bhow\s+(?:are\s+you|are\s+you\s+doing|do\s+you\s+feel|have\s+you\s+been)\s+(?:feeling|today|lately|these\s+days)\b"

raven_responses = json.load(open("database/responses.json"))

def raven_uget():
    while True:
        try:
            usersaid = raven_command()  # Retrieve user's input
            if usersaid in greetings:  # If the user input is a greeting
                # Generate and output a greeting response
                raven_speaker(generate_greeting_response(usersaid))
                break  # Exit the loop
            elif usersaid in goodbyes:  # If the user input is a goodbye
                # Generate and output a goodbye response
                raven_speaker(generate_goodbyes_response(usersaid))
                break  # Exit the loop
            elif usersaid in feelings:  # If the user input is a feeling
                # Generate and output a feeling response
                raven_speaker(generate_feelings_response(usersaid))
                break  # Exit the loop
            else:  # If the user input does not match any of the above categories
                openresponse = generate_open_response(
                    usersaid)  # Generate an open response
                raven_speaker(openresponse)  # Output the open response
                break  # Exit the loop
        except Exception as e:  # If an exception occurs
            logging.error(traceback.format_exc())  # Log the error message
            print(
                f"{Fore.RED}ЯΛVΣП: {Style.RESET_ALL}Sorry, did not get that.")  # Output an error message to the console
            # Output an error message using the conversational agent
            raven_speaker(f"Sorry, did not get that.")
            break  # Exit the loop
