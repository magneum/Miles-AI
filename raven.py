
from colorama import Fore, Style
from dotenv import load_dotenv
from termcolor import cprint
from app import *
import pvporcupine
import numpy as np
import pyaudio
import struct
import asyncio
import random
import time
import os


load_dotenv()


def play_notif(freq, duration):
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


def listen_process(porcupine, audio_stream):
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
            print(f"{Fore.MAGENTA}ЯΛVΣП: {Style.RESET_ALL}waiting for command.")
        else:
            # Keep listening for the wake word
            time.sleep(0.1)


async def main():
    try:
        # Initialize variables
        paud = None
        Listening = None
        porcupine = None
        audio_stream = None

        # Use the raven_speaker function to greet the user with a response to "hi"
        # raven_speaker(generate_greeting_response("hi"))
        # Play a tone sound to indicate the program is ready
        play_notif(800, 0.2)
        # Print that kaida is now listening
        print(f"{Fore.YELLOW}ЯΛVΣП: {Style.RESET_ALL}Ready...")

        # Attempt to execute the following block of code
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
                if Listening is not None:
                    listen_process(porcupine, audio_stream)
                    Listening = False
                else:
                    Listening = True

        # Catch any exception and speak a random error message from the responses.json file
        except Exception as e:
            raven_speaker(random.choice(
                json.load(open("database/responses.json"))["error"]["responses"]))
            # Print the error message in red text
            print(f"{Fore.RED}ЯΛVΣП: {Style.RESET_ALL}{e}")

    # Catch the KeyboardInterrupt exception and speak a random goodbye message from the responses.json file
    except KeyboardInterrupt:
        raven_speaker(random.choice(
            json.load(open("database/responses.json"))["goodbye"]["responses"]))

    # Catch any other exception and speak a random error message from the responses.json file
    except Exception as e:
        raven_speaker(random.choice(
            json.load(open("database/responses.json"))["error"]["responses"]))
        # Print the error message in red text
        print(f"{Fore.RED}ЯΛVΣП: {Style.RESET_ALL}{e}")

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


# create an event loop and run the coroutine
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
