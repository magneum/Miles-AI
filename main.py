# Raven - This Japanese name means "little dragon," and could be a good fit for an AI chatbot that is powerful and efficient.


import os
import time
import json
import random
import pyttsx3
import pygame
import pyaudio
import numpy as np
import pvporcupine
from app import *
from termcolor import cprint
from dotenv import load_dotenv
import speech_recognition as sr
from colorama import Fore, Style
load_dotenv()


def porcupine_listen(porcupine, audio_stream):
    while True:
        # Read audio data from the stream
        pcm = audio_stream.read(porcupine.frame_length)
        # Convert the binary data to an array of integers
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
        # Process the audio data with Porcupine
        wake_index = porcupine.process(pcm)
        if wake_index == 0:
            # If the wake word is detected, print a message, speak a response, and wait for a command
            print(f"{Fore.YELLOW}ЯΛVΣП: wake word detected.")
            return True
        else:
            # Keep listening for the wake word
            time.sleep(0.1)


# Define a function called _notification that takes two arguments: duration and freq
def _notification(duration, freq):
    # Create a sinusoidal signal with a frequency of freq and a duration of duration using the numpy library
    signal = np.sin(freq * 2 * np.pi * np.linspace(0,
                    duration, int(duration * 44100), False))
    # Open an output stream to the default audio device using PyAudio
    stream = pyaudio.PyAudio().open(format=pyaudio.paFloat32,
                                    channels=1, rate=44100, output=True)
    # Write the signal to the stream and close the stream
    stream.write(signal.astype(np.float32).tobytes())
    stream.close()
    # Terminate the PyAudio object
    pyaudio.PyAudio().terminate()


# Define a function called _speaker that takes a string argument called usersaid
def _speaker(usersaid):
    try:
        # Initialize a new pyttsx3 speech synthesis engine with the "sapi5" driver for Windows
        speaker = pyttsx3.init("sapi5")
        # Set the speech rate of the engine to 150 words per minute
        speaker.setProperty("rate", 150)
        # Set the voice property of the engine to the first available voice in the list of available voices
        speaker.setProperty("voice", speaker.getProperty("voices")[0].id)
        # Use the say method of the engine to speak the string passed as an argument
        speaker.say(usersaid)
        print(f"{Fore.MAGENTA}RAVEN: {usersaid}")
        # Use the runAndWait method of the engine to wait until the speech is complete
        speaker.runAndWait()

    # If there is an exception, print a message to the console in red text using the Fore.RED constant from the colorama library,
    # play a short notification sound using a function called _notification, and continue program execution
    except Exception as e:
        print(f"{Fore.RED}Error: {e}")
        _notification(0.2, 200)


def main():
    try:
        # Initialize a variable to keep track of whether the program is listening for speech input
        listening = False
        # Create a speech recognizer object
        recognizer = sr.Recognizer()
        # Create a microphone object to capture audio input
        microphone = sr.Microphone()
        # Generate a greeting response for the user saying "hi"
        greet_init = generate_greeting_response("hi")
        # Speak the greeting message using a text-to-speech function called _speaker
        _speaker(greet_init)
        # Print that raven is now listening
        print(f"{Fore.YELLOW}RAVEN: listening...")
        # Play a notification sound using a function called _notification
        _notification(0.2, 800)

        # Initialize variables
        paud = None
        detected = False
        porcupine = None
        audio_stream = None

        # Create a loop that runs indefinitely
        while True:

            # Use the microphone as the audio source
            with microphone as source:
                # Adjust for ambient noise before capturing audio
                recognizer.adjust_for_ambient_noise(source)
                # Record audio from the microphone
                audio = recognizer.listen(source)

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
            detected = porcupine_listen(porcupine, audio_stream)
            if detected:

                try:
                    # Use Google's speech recognition API to convert speech to text
                    usersaid = recognizer.recognize_google(audio)
                    # Print the recognized text to the console in blue text using the Fore library
                    print(f"{Fore.BLUE}RAVEN: {usersaid.lower()}")
                    # If the program is not already listening
                    if usersaid.lower() and not listening:
                        # Print a message indicating that the program is now listening in yellow text using the Fore library
                        print(f"{Fore.YELLOW}RAVEN: wake word detected...")
                        # Play a notification sound using a function called _notification
                        _notification(0.2, 600)
                        # Set the listening variable to True to indicate that the program is now listening
                        listening = True

                    # If the program is already listening
                    elif usersaid.lower() and listening:
                        # Print a message indicating that the program is still listening in yellow text using the Fore library
                        print(f"{Fore.YELLOW}RAVEN: listening again...")
                        # Play a notification sound using a function called _notification
                        _notification(0.2, 500)
                        # Check if the user said the same thing twice
                        if usersaid in usersaid.lower():
                            pass
                        # check for commands
                        elif usersaid in greetings:  # If the user input is a greeting
                            # Generate and output a greeting response
                            _speaker(generate_greeting_response(usersaid))
                            break  # Exit the loop
                        elif usersaid in goodbyes:  # If the user input is a goodbye
                            # Generate and output a goodbye response
                            _speaker(generate_goodbyes_response(usersaid))
                            break  # Exit the loop
                        elif usersaid in feelings:  # If the user input is a feeling
                            # Generate and output a feeling response
                            _speaker(generate_feelings_response(usersaid))
                            break  # Exit the loop

                        else:
                            # If the user said something different, ask them to repeat their command
                            _speaker("Sorry, I did not understand.")
                            # Break out of the loop
                            break

                        # Set the listening variable to False to indicate that the program is no longer listening
                        listening = False

                # If there is an error during speech recognition
                except Exception as e:
                    # Print an error message to the console in red text using the Fore library
                    print(f"{Fore.RED}RAVEN: Sorry. There has been an error; {e}")
                    # Use a text-to-speech function called _speaker to speak the error message
                    _speaker(f"Sorry. There has been an error;")
                finally:
                    # Cleanup resources
                    if porcupine is not None:
                        porcupine.delete()
                    if audio_stream is not None:
                        audio_stream.close()
                    if paud is not None:
                        paud.terminate()

    # If there is an exception during program execution, speak a random error message
    # from a predefined list of responses and print the exception message to the console
    except Exception as e:
        _speaker(random.choice(
            json.load(open("database/responses.json"))["error"]["responses"]))
        print(f"{Fore.RED}RAVEN: {e}")

    # If the user presses Ctrl-C to interrupt the program, print a message to the console in green text using the cprint function from the termcolor library,
    # speak a goodbye message using a function called generate_goodbyes_response, and exit the program
    except KeyboardInterrupt:
        cprint("RAVEN: Shutting down...", "green")
        _speaker(generate_goodbyes_response("bye"))

    # If there is any other exception besides KeyboardInterrupt, speak a random error message
    # from a predefined list of responses and print the exception message to the console
    except Exception as e:
        _speaker(random.choice(
            json.load(open("database/responses.json"))["error"]["responses"]))
        print(f"{Fore.RED}RAVEN: {e}")


main()


"""
elif "play" in usersaid.lower():
    print(f"{Fore.GREEN}RAVEN:  started!")
    pygame.init()
    pygame.mixer.music.play()
    _speaker("Music started")
    engine.runAndWait()

elif "stop" in usersaid.lower():
    print(f"{Fore.RED}RAVEN:  stopped!")
    pygame.mixer.music.stop()
    _speaker("Music stopped")
    engine.runAndWait()

elif "pause" in usersaid.lower():
    print(f"{Fore.BLUE}RAVEN:  paused!")
    pygame.mixer.music.pause()
    _speaker("Music paused")
    engine.runAndWait()

elif "resume" in usersaid.lower():
    print(f"{Fore.GREEN}RAVEN: resumed!")
    pygame.mixer.music.unpause()
    _speaker("Music resumed")
    engine.runAndWait()

elif "volume" in usersaid.lower():
    words = usersaid.split()
    index = words.index("volume")
    if index + 1 < len(words):
        try:
            volume = float(words[index + 1])
            pygame.mixer.music.set_volume(volume)
            _speaker(f"Volume set to {volume}")
            engine.runAndWait()
        except ValueError:
            _speaker("Please specify a valid volume")
            engine.runAndWait()
    else:
        _speaker("Please specify a volume level")
        engine.runAndWait()

elif "quit" in usersaid.lower():
    print("Goodbye!")
    _speaker("Goodbye!")
    engine.runAndWait()
"""
