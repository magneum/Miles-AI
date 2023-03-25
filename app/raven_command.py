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
from .raven_speaker import raven_speaker


# define function to recognize user speech input


def raven_command():

    # initialize speech recognition and microphone
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    # adjust microphone for ambient noise and prompt user to speak
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        simpleaudio.WaveObject.from_wave_file("src/_Tone.wav").play()
        print(f"{Fore.YELLOW}ЯΛVΣП: {Style.RESET_ALL}listening...")
        audio = recognizer.listen(source)

    # try to recognize user input from the audio
    try:
        # play sound to indicate audio recognition is complete
        simpleaudio.WaveObject.from_wave_file("src/Tone_.wav").play()
        print(
            f"{Fore.BLUE}ЯΛVΣП: {Style.RESET_ALL}recognizing {len(audio.frame_data)} bytes of audio")
        user_input = recognizer.recognize_google(audio)
        print(f"{Fore.GREEN}ЯΛVΣП: {Style.RESET_ALL}user input: {user_input}")

    # handle error when audio is not recognized
    except sr.UnknownValueError:
        print(f"{Fore.RED}ЯΛVΣП: {Style.RESET_ALL}Sorry, I didn't understand that.")
        raven_speaker("Sorry, I didn't understand that.")
        return "none"

    # handle error when speech recognition service is unavailable
    except sr.RequestError as e:
        print(f"{Fore.RED}ЯΛVΣП: {Style.RESET_ALL}Could not request results from Google Speech Recognition service; {e}")
        raven_speaker("Sorry, I didn't understand that.")
        return "none"

    # handle error when audio playback fails
    except sa.PlaybackError as e:
        print(f"{Fore.RED}ЯΛVΣП: {Style.RESET_ALL}Could not play sound; {e}")
        raven_speaker(f"Could not play sound; {e}")
        return "none"

    # return recognized user input in lowercase
    return user_input.lower()
