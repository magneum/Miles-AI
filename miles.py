# =============================================================================================================
from colorama import Fore, Style
from dotenv import load_dotenv
import pvporcupine
import numpy as np
import threading
import pyaudio
import struct
import pyttsx3
import asyncio
import random
import logging
import os
from app import *
from pygame import mixer
from database.greetings import *
from database.feelings import *
from database.goodbyes import *
import logging
load_dotenv()
speaker = pyttsx3.init("sapi5")
# =============================================================================================================


def listen_process(porcupine, audio_stream, paud):
    # Read audio data from the stream
    pcm = audio_stream.read(porcupine.frame_length)
    # Convert the binary data to an array of integers
    pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
    # Process the audio data with Porcupine
    wake_index = porcupine.process(pcm)
    if wake_index == 0:
        # If the wake word is detected, print a message, speak a response, and wait for a command
        print(f"{Fore.YELLOW}MÌLΣƧ. ΛI: wake word detected.")
        miles_uget(porcupine, audio_stream, paud)
        print(f"{Fore.MAGENTA}MÌLΣƧ. ΛI: waiting for command.")
    else:
        pass
# =============================================================================================================


def listening_loop(porcupine, audio_stream, paud):
    while True:  # loop indefinitely
        # call listen_process function to listen for wake word and commands
        Listening = listen_process(porcupine, audio_stream, paud)
        while True:  # loop indefinitely
            if Listening and Listening is not None:  # if wake word is detected and the Listening variable is not None
                # print message to indicate that Raven is re-listening
                print(f"{Fore.YELLOW}MÌLΣƧ. ΛI: re-listening...")
                # call listen_process function to listen for wake word and commands
                Listening = listen_process(porcupine, audio_stream, paud)
            else:  # if wake word is not detected or Listening variable is None
                # call listen_process function to listen for wake word and commands
                listen_process(porcupine, audio_stream, paud)
# =============================================================================================================


async def my_coroutine():
    try:
        # Initialize variables
        paud = None
        porcupine = None
        audio_stream = None

        # Play a tone sound to indicate the program is ready
        play_notif(800, 0.2)
        # Print that miles is now listening
        print(f"{Fore.YELLOW}MÌLΣƧ. ΛI: Ready...")

        # Attempt to execute the following block of code
        try:
            # Initialize Porcupine with the access key and the path to the keyword file
            porcupine = pvporcupine.create(
                access_key=os.getenv("PORCUPINE_KEY"), keyword_paths=["models/hey-evo-windows.ppn"])
            # Initialize PyAudio and the audio stream with Porcupine's settings
            paud = pyaudio.PyAudio()
            audio_stream = paud.open(
                channels=1,
                input=True,
                format=pyaudio.paInt16,
                rate=porcupine.sample_rate,
                frames_per_buffer=porcupine.frame_length
            )

            # Start the listening loop in a separate thread
            t = threading.Thread(target=listening_loop, args=(
                porcupine, audio_stream, paud), daemon=True)
            t.start()

            # Run the main loop indefinitely
            while True:
                # Run any other code that needs to run in the main loop
                listen_process(porcupine, audio_stream, paud)
        # Catch any exception and speak a random error message from the responses.json file
        except Exception as e:
            miles_speaker(random.choice(
                json.load(open("database/responses.json"))["error"]["responses"]))
            # Print the error message in red text
            print(f"{Fore.RED}MÌLΣƧ. ΛI: {e}")

    # Catch the KeyboardInterrupt exception and speak a random goodbye message from the responses.json file
    except KeyboardInterrupt:
        miles_speaker(random.choice(
            json.load(open("database/responses.json"))["goodbye"]["responses"]))

    # Catch any other exception and speak a random error message from the responses.json file
    except Exception as e:
        miles_speaker(random.choice(
            json.load(open("database/responses.json"))["error"]["responses"]))
        # Print the error message in red text
        print(f"{Fore.RED}MÌLΣƧ. ΛI: {e}")

    # Clean up resources
    finally:
        # If porcupine object exists, delete it
        if porcupine is not None:
            porcupine.delete()
        # If audio stream object exists, close it
        if audio_stream is not None:
            audio_stream.close()
        # If PyAudio object exists, terminate it
        if paud is not None:
            paud.terminate()
# =============================================================================================================
# create an event loop and run the coroutine
if not asyncio.get_event_loop().is_running():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(my_coroutine())
# =============================================================================================================
