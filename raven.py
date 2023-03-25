
from colorama import Fore, Style
from dotenv import load_dotenv
from termcolor import cprint
import pvporcupine
import numpy as np
import simpleaudio
import threading
import traceback
import pyaudio
import struct
import pyttsx3
import speech_recognition as sr
import asyncio
import random
import openai
import time
import os
from pygame import mixer
from database.greetings import *
from database.feelings import *
from database.goodbyes import *
import logging
load_dotenv()
speaker = pyttsx3.init("sapi5")


# +==================================================================================================================
#
#
# +==================================================================================================================

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

# +==================================================================================================================
#
#
# +==================================================================================================================


def raven_command():

    # initialize speech recognition and microphone
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    # adjust microphone for ambient noise and prompt user to speak
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print(f"{Fore.YELLOW}RAVEN: listening...")
        simpleaudio.WaveObject.from_wave_file("src/Tone_.wav").play()
        audio = recognizer.listen(source, timeout=4)

    # try to recognize user input from the audio
    try:
        # play sound to indicate audio recognition is complete
        simpleaudio.WaveObject.from_wave_file("src/_Tone.wav").play()
        print(
            f"{Fore.BLUE}RAVEN: recognizing {len(audio.frame_data)} bytes of audio")
        user_input = recognizer.recognize_google(audio)
        print(f"{Fore.GREEN}USER: {user_input}")

    # handle error when audio is not recognized
    except sr.UnknownValueError:
        print(f"{Fore.RED}RAVEN: Sorry, I didn't understand that.")
        return ""

    # handle error when speech recognition service is unavailable
    except sr.RequestError as e:
        print(
            f"{Fore.RED}RAVEN: Could not request results from Google Speech Recognition service; {e}")
        return ""

    # handle error when audio playback fails
    except sa.PlaybackError as e:
        print(f"{Fore.RED}RAVEN: Could not play sound; {e}")
        raven_speaker(f"Could not play sound; {e}")
        return ""

    # return recognized user input in lowercase
    return user_input.lower()

# Define a function to speak the response using the raven_speaker function

# +==================================================================================================================
#
#
# +==================================================================================================================


def raven_speaker(usersaid):
    try:
        # Set the speaking rate to 140
        speaker.setProperty("rate", 145)
        # Get a list of all available voices and set the default voice to the first voice in the list
        voices = speaker.getProperty("voices")
        speaker.setProperty("voice", voices[0].id)
        # Print the message to be spoken in blue color
        print(f"{Fore.BLUE}RAVEN: {usersaid}")
        # Say the message using the text-to-speech engine
        speaker.say(usersaid)
        # Wait until the text-to-speech engine has finished speaking before continuing
        speaker.runAndWait()
    except Exception as e:
        # If an error occurs, print the error message in red color
        print(f"{Fore.RED}Error: {e}")

# +==================================================================================================================
#
#
# +==================================================================================================================


def raven_uget(porcupine, audio_stream, paud):
    try:
        # Retrieve user's input
        usersaid = raven_command()

        # Raise an error if user's input is empty
        if not usersaid:
            raise ValueError("Empty input received from user")

        # Check if user's input matches any of the greetings in the greetings list
        if usersaid in greetings:
            # Generate a greeting response based on the user's input
            response = generate_greeting_response(usersaid)
        # Check if user's input matches any of the goodbyes in the goodbyes list
        elif usersaid in goodbyes:
            # Generate a goodbye response based on the user's input
            response = generate_goodbyes_response(usersaid)
        # Check if user's input matches any of the feelings in the feelings list
        elif usersaid in feelings:
            # Generate a response based on the user's input
            response = generate_feelings_response(usersaid)
        # Check if the user's input contains the word "shutdown"
        elif "shutdown" in usersaid:
            # Shut down the program
            raven_speaker("Shutting down...")
            os._exit(0)
        else:
            # Generate a response for any other input
            response = generate_open_response(usersaid)

        # Use text-to-speech to speak the generated response
        raven_speaker(response)

    except ValueError as e:
        # Log a warning if an empty input is received
        logging.warning(f"Empty input received from user: {e}")
        # Use text-to-speech to ask the user to try again
        raven_speaker("Sorry, I didn't catch that. Can you please try again?")
    except Exception as e:
        # Log an error if an exception occurs
        logging.error(traceback.format_exc())
        # Use text-to-speech to inform the user that an error has occurred
        raven_speaker("Oops! Something went wrong. Please try again later.")


# +==================================================================================================================
#
#
# +==================================================================================================================

def play_notif(freq, duration):
    # Generate a sine wave audio signal with the given frequency and duration
    signal = np.sin(freq * 2 * np.pi * np.linspace(0,
                    duration, int(duration * 44100), False))

    # Open a PyAudio stream for audio output
    stream = pyaudio.PyAudio().open(format=pyaudio.paFloat32,
                                    channels=1,
                                    rate=44100,
                                    output=True)

    # Write the audio signal to the stream
    stream.write(signal.astype(np.float32).tobytes())

    # Close the audio stream
    stream.close()

    # Terminate the PyAudio interface
    pyaudio.PyAudio().terminate()

# +==================================================================================================================
#
#
# +==================================================================================================================


def listen_process(porcupine, audio_stream, paud):
    # Read audio data from the stream
    pcm = audio_stream.read(porcupine.frame_length)
    # Convert the binary data to an array of integers
    pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
    # Process the audio data with Porcupine
    wake_index = porcupine.process(pcm)
    if wake_index == 0:
        # If the wake word is detected, print a message, speak a response, and wait for a command
        print(f"{Fore.YELLOW}RAVEN: wake word detected.")
        raven_uget(porcupine, audio_stream, paud)
        print(f"{Fore.MAGENTA}RAVEN: waiting for command.")
    else:
        pass

# +==================================================================================================================
#
#
# +==================================================================================================================


def listening_loop(porcupine, audio_stream, paud):
    while True:  # loop indefinitely
        # call listen_process function to listen for wake word and commands
        Listening = listen_process(porcupine, audio_stream, paud)
        while True:  # loop indefinitely
            if Listening and Listening is not None:  # if wake word is detected and the Listening variable is not None
                # print message to indicate that Raven is re-listening
                print(f"{Fore.YELLOW}RAVEN: re-listening...")
                # call listen_process function to listen for wake word and commands
                Listening = listen_process(porcupine, audio_stream, paud)
            else:  # if wake word is not detected or Listening variable is None
                # call listen_process function to listen for wake word and commands
                listen_process(porcupine, audio_stream, paud)

# +==================================================================================================================
#
#
# +==================================================================================================================


async def my_coroutine():
    try:
        # Initialize variables
        paud = None
        porcupine = None
        audio_stream = None

        # Play a tone sound to indicate the program is ready
        play_notif(800, 0.2)
        # Print that raven is now listening
        print(f"{Fore.YELLOW}RAVEN: Ready...")

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
            raven_speaker(random.choice(
                json.load(open("database/responses.json"))["error"]["responses"]))
            # Print the error message in red text
            print(f"{Fore.RED}RAVEN: {e}")

    # Catch the KeyboardInterrupt exception and speak a random goodbye message from the responses.json file
    except KeyboardInterrupt:
        raven_speaker(random.choice(
            json.load(open("database/responses.json"))["goodbye"]["responses"]))

    # Catch any other exception and speak a random error message from the responses.json file
    except Exception as e:
        raven_speaker(random.choice(
            json.load(open("database/responses.json"))["error"]["responses"]))
        # Print the error message in red text
        print(f"{Fore.RED}RAVEN: {e}")

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

# +==================================================================================================================
#
#
# +==================================================================================================================
# create an event loop and run the coroutine
if not asyncio.get_event_loop().is_running():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(my_coroutine())
