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


def kaida_uget():
    while True:
        try:
            usersaid = kaida_command()  # Retrieve user's input
            if usersaid in greetings:  # If the user input is a greeting
                # Generate and output a greeting response
                kaida_speaker(generate_greeting_response(usersaid))
                break  # Exit the loop
            elif usersaid in goodbyes:  # If the user input is a goodbye
                # Generate and output a goodbye response
                kaida_speaker(generate_goodbyes_response(usersaid))
                break  # Exit the loop
            elif usersaid in feelings:  # If the user input is a feeling
                # Generate and output a feeling response
                kaida_speaker(generate_feelings_response(usersaid))
                break  # Exit the loop
            else:  # If the user input does not match any of the above categories
                openresponse = generate_open_response(
                    usersaid)  # Generate an open response
                kaida_speaker(openresponse)  # Output the open response
                break  # Exit the loop
        except Exception as e:  # If an exception occurs
            logging.error(traceback.format_exc())  # Log the error message
            print(
                f"{Fore.RED}KΛIDΛ: {Style.RESET_ALL}Sorry, did not get that.")  # Output an error message to the console
            # Output an error message using the conversational agent
            kaida_speaker(f"Sorry, did not get that.")
            break  # Exit the loop
