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


# Defining function to generate open response using OpenAI API
def generate_open_response(usersaid):
    # Setting API key for OpenAI
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Checking if API key is present
    if not openai.api_key:
        return "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."

    # Checking if user input is valid
    if not usersaid or not usersaid.strip():
        return "Invalid input. Please provide a valid question or statement."

    """
    These are the parameters used in the OpenAI API for creating a text completion request using the text-davinci-003 language model. Here is a brief explanation of each parameter:

    > engine: Specifies the name of the language model to use. In this case, it's text-davinci-003.
    > prompt: Specifies the text prompt for which the language model should generate a continuation. The usersaid variable contains the user's input, which is used as the prompt in this case.
    > max_tokens: Specifies the maximum number of tokens (words or punctuation marks) that the language model should generate as the continuation. The default value is 2048, but in this case, it's set to 1000.
    > n: Specifies the number of different continuations that the language model should generate. The default value is 1.
    > stop: Specifies a list of strings that the language model should use as stop sequences. If any of the stop sequences are generated as part of the continuation, the generation process will stop at that point. The default value is None, which means that there are no stop sequences.
    > temperature: Specifies the level of randomness and creativity in the generated text. A higher temperature results in more diverse and unexpected output, while a lower temperature results in more predictable and conservative output. The default value is 1.0, but in this case, it's set to 0.5.
    """
    # Generating response using OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=usersaid,
        max_tokens=500,  # 2048 is max
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Returning response text
    if response.choices[0].text:
        return response.choices[0].text.strip()
    else:
        return "Sorry, I couldn't understand your question or statement."
