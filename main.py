# Raven - This Japanese name means "little dragon," and could be a good fit for an AI chatbot that is powerful and efficient.


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
import threading
import traceback
# import winsound
import simpleaudio
from app import *
import pvporcupine
from termcolor import cprint
from dotenv import load_dotenv
import speech_recognition as sr
from colorama import Fore, Style


# ===========================================================================================

load_dotenv()
current_dir = os.path.dirname(__file__)

greetings = r"hi\b|hello\b|hey\b|greetings\b|salutations\b|yo\b|hiya\b|howdy\bsup\b|hi there\b|hello there\b|what's up\b|yoohoo\b|hey there\b|hiya there\b|g'day\b|cheerio\b|hihi\b|aloha\b|bonjour\b|hallo\b|ciao\b|namaste\b|konichiwa\b|hola\b|szia\b|hei\b|hej\b|tjena\b|heya\b|hey ya\b|sup dude\b|sup bro\b|sup everyone\b|wassup\b|whaddup\b"
goodbyes = r"bye\b|goodbye\b|farewell\b|see you\b|take care\b|cheerio\b|ciao\b|so long\b|until next time\b|peace out\b|later\badios\b|au revoir\b|bye for now\b|catch you later\b|have a good one\b|keep in touch\b|leaving now\b|parting ways\b|so farewell\b|stay safe\b|till we meet again\b"
feelings = r"\bhow\s+(?:are\s+you|are\s+you\s+doing|do\s+you\s+feel|have\s+you\s+been)\s+(?:feeling|today|lately|these\s+days)\b"

raven_responses = json.load(open("database/responses.json"))


# ===========================================================================================


def main():
    try:
        # Initialize variables
        paud = None
        porcupine = None
        audio_stream = None

        # Use the raven_speaker function to greet the user with a response to "hi"
        raven_speaker(generate_greeting_response("hi"))

        # Play a tone sound to indicate the program is ready
        # winsound.Beep(200, 200)

        # Print that kaida is now listening
        print(f"{Fore.YELLOW}ЯΛVΣП: {Style.RESET_ALL}listening...")

        try:
            # Initialize Porcupine with the access key and the path to the keyword file
            porcupine = pvporcupine.create(
                access_key=os.getenv("PORCUPINE"), keyword_paths=["models/hey-evo-windows.ppn"])

            # Initialize PyAudio and the audio stream with Porcupine's settings
            paud = pyaudio.PyAudio()
            audio_stream = paud.open(
                channels=1,
                input=True,
                format=pyaudio.paInt16,
                rate=porcupine.sample_rate,
                frames_per_buffer=porcupine.frame_length
            )

            # Continuously listen for the wake word and commands
            while True:
                # Read audio data from the stream
                pcm = audio_stream.read(porcupine.frame_length)

                # Convert the binary data to an array of integers
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                # Process the audio data with Porcupine
                wake_index = porcupine.process(pcm)

                # If the wake word is detected
                if wake_index == 0:
                    # Notify the user and ask for a command
                    print(
                        f"{Fore.YELLOW}ЯΛVΣП: {Style.RESET_ALL}wake word detected.")
                    raven_uget()
                    print(
                        f"{Fore.MAGENTA}ЯΛVΣП: {Style.RESET_ALL}waiting for command.")

                    # Add a loop to re-listen for the wake word and commands
                    while True:
                        # Read audio data from the stream
                        pcm = audio_stream.read(porcupine.frame_length)

                        # Convert the binary data to an array of integers
                        pcm = struct.unpack_from(
                            "h" * porcupine.frame_length, pcm)

                        # Process the audio data with Porcupine
                        wake_index = porcupine.process(pcm)
                        if wake_index == 0:
                            # If the wake word is detected, print a message, speak a response, and wait for a command
                            print(
                                f"{Fore.YELLOW}ЯΛVΣП: {Style.RESET_ALL}wake word detected.")
                            raven_uget()
                            print(
                                f"{Fore.MAGENTA}ЯΛVΣП: {Style.RESET_ALL}waiting for command.")
                        else:
                            # If a command is detected, process it and break out of the loop
                            # The loop is used to keep listening for the wake word after a command has been processed
                            break
                    continue
        except Exception as e:
            # If there's an exception, speak an error message and print the exception
            raven_speaker(random.choice(raven_responses["error"]["responses"]))
            print(f"{Fore.RED}ЯΛVΣП: {Style.RESET_ALL}{e}")

    except KeyboardInterrupt:
        cprint("ЯΛVΣП: Shutting down...", "green")
        raven_speaker(generate_goodbyes_response("bye"))
    except Exception as e:
        # If there's any exception other than KeyboardInterrupt, speak an error message and print the exception
        raven_speaker(random.choice(raven_responses["error"]["responses"]))
        print(f"{Fore.RED}ЯΛVΣП: {Style.RESET_ALL}{e}")
    finally:
        # Cleanup resources
        if porcupine is not None:
            porcupine.delete()
        if audio_stream is not None:
            audio_stream.close()
        if paud is not None:
            paud.terminate()


# ===========================================================================================


if __name__ == "__main__":
    # asyncio.run(main())
    main()
