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

# Define a function to speak the response using the raven_speaker function


def raven_speaker(usersaid):
    if not isinstance(usersaid, str):
        print(f"{Fore.RED}Error: Input must be a string.{Style.RESET_ALL}")
        return
    try:
        speaker = pyttsx3.init("sapi5")
        speaker.setProperty("rate", 150)
        # speaker.setProperty("volume", 0.5)
        voices = speaker.getProperty("voices")
        speaker.setProperty("voice", voices[0].id)
        print(f"{Fore.BLUE}ЯΛVΣП: {Style.RESET_ALL}{usersaid}")
        speaker.say(usersaid)
        speaker.runAndWait()
    except Exception as e:
        print(f"{Fore.RED}Error: {Style.RESET_ALL}{e}")
