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
import numpy as np
import pvporcupine
from app import *
from termcolor import cprint
from dotenv import load_dotenv
import speech_recognition as sr
from colorama import Fore, Style
# ********************************************************************************************************
load_dotenv()


async def play_notif(freq, duration):
    signal = np.sin(freq * 2 * np.pi * np.linspace(0,
                    duration, int(duration * 44100), False))
    # Play the audio signal
    stream = pyaudio.PyAudio().open(format=pyaudio.paFloat32,
                                    channels=1,
                                    rate=44100,
                                    output=True)
    stream.write(signal.astype(np.float32).tobytes())
    stream.close()
    pyaudio.PyAudio().terminate()


async def wakeword_listen_and_process(porcupine, audio_stream):
    while True:
        # Read audio data from the stream
        pcm = audio_stream.read(porcupine.frame_length)
        # Convert the binary data to an array of integers
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
        # Process the audio data with Porcupine
        wake_index = porcupine.process(pcm)
        if wake_index == 0:
            # If the wake word is detected, print a message, speak a response, and wait for a command
            print(f"{Fore.YELLOW}ЯΛVΣП: {Style.RESET_ALL}wake word detected.")
            raven_uget()
        else:
            # Keep listening for the wake word
            await asyncio.sleep(0)


async def main():
    try:
        # Initialize variables
        paud = None
        porcupine = None
        audio_stream = None

        # Use the raven_speaker function to greet the user with a response to "hi"
        # raven_speaker(generate_greeting_response("hi"))
        # Play a tone sound to indicate the program is ready
        await play_notif(800, 0.2)
        # Print that kaida is now listening
        print(f"{Fore.YELLOW}ЯΛVΣП: {Style.RESET_ALL}Ready...")

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
                # Listen for the wake word
                await wakeword_listen_and_process(porcupine, audio_stream)

        except Exception as e:
            # If there's an exception, speak an error message and print the exception
            raven_speaker(random.choice(raven_responses["error"]["responses"]))
            print(f"{Fore.RED}ЯΛVΣП: {Style.RESET_ALL}{e}")
        except KeyboardInterrupt:
            cprint("ЯΛVΣП: Shutting down...", "green")
            raven_speaker("Shutting down..")
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
    except Exception as e:
        # If there's an exception, speak an error message and print the exception
        raven_speaker(random.choice(raven_responses["error"]["responses"]))
        print(f"{Fore.RED}ЯΛVΣП: {Style.RESET_ALL}{e}")


# create an event loop and run the coroutine
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
