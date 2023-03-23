from database.greetings import generate_greeting_response
import os
import re
import sys
import time
import json
import random
import struct
import pyaudio
import datetime
import requests
import pyttsx3
import webbrowser
import pvporcupine
import simpleaudio
import openai as assistant
from termcolor import cprint
from colorama import Fore, Style
import speech_recognition as sprecog
current_dir = os.path.dirname(__file__)


# ================================================================= FUNCTIONS FOR RESPONSES =================================================================


# simpleaudio.WaveObject.from_wave_file("src/_Tone.wav").play()
# simpleaudio.WaveObject.from_wave_file("src/Tone_.wav").play().wait_done()
# sys.exit()


def kai_speaker(KAI_TEXT):
    # windows
    speaker = pyttsx3.init("sapi5")
    # linux - apt install espeak && apt install libespeak-dev
    # speaker = pyttsx3.init("espeak")
    speaker.setProperty("rate", 150)
    voices = speaker.getProperty("voices")
    speaker.setProperty("voice", voices[0].id)
    print(f"{Fore.BLUE}ҠΛI: {Style.RESET_ALL}{KAI_TEXT}")
    speaker.say(KAI_TEXT)
    speaker.runAndWait()


def KAI_Command():
    recog = sprecog.Recognizer()
    with sprecog.Microphone() as mic:
        userquery = ""
        recog.adjust_for_ambient_noise(mic, duration=0.2)
        simpleaudio.WaveObject.from_wave_file("src/_Tone.wav").play()
        print(f"{Fore.YELLOW}ҠΛI: {Style.RESET_ALL}listening...")
        # recog.pause_threshold = 4
        # recog.operation_timeout = 4
        audio = recog.listen(mic)
        try:
            simpleaudio.WaveObject.from_wave_file("src/Tone_.wav").play()
            print(f"{Fore.BLUE}ҠΛI: {Style.RESET_ALL}recognizing {audio}")
            userquery = recog.recognize_google(
                audio_data=audio, language="en-us")
            print(f"{Fore.GREEN}ҠΛI: {Style.RESET_ALL}usersaid {userquery}")
        except Exception as e:
            print(f"{Fore.RED}ҠΛI: {Style.RESET_ALL}Sorry, did not get that.")
            cprint(f": {e}", "white", "on_grey", attrs=[])
            kai_speaker(f"Sorry, did not get that.")
            return "none"
        return userquery.lower()


greetings = ["hello", "hey", "hi"]
goodbye = ["bye", "goodbye", "tata"]
feeling = ["how are you feeling"]
areyou = ["who are you", "what are you"]
shutdown = ["shutdown", "poweroff"]

music_genres = [
    "Pop",
    "Rock",
    "Hip hop",
    "Electronic",
    "Country",
    "R&B",
    "Jazz",
    "Classical",
    "Folk",
    "Blues",
    "Reggae",
    "World",
    "Metal",
    "Indie",
    "Punk",
    "Alternative",
    "Gospel",
    "Soul",
    "Funk"
]


def kai_uget():
    while True:
        usersaid = KAI_Command()
        if usersaid in greetings:
            kai_speaker(random.choice(KAI_Responses["hello"]["responses"]))
            break
        elif usersaid in goodbye:
            kai_speaker(random.choice(KAI_Responses["goodbye"]["responses"]))
            break
        elif usersaid in feeling:
            kai_speaker(random.choice(KAI_Responses["feeling"]["responses"]))
            break
        elif usersaid in areyou:
            kai_speaker(random.choice(KAI_Responses["areyou"]["responses"]))
            break
        elif usersaid in areyou:
            kai_speaker(random.choice(KAI_Responses["areyou"]["responses"]))
            break
        elif usersaid in shutdown:
            kai_speaker(random.choice(KAI_Responses["shutdown"]["responses"]))
            os.system("shutdown /s /t 1")
        elif "play" in usersaid:
            try:
                songname = usersaid.split("play", 1)[1]
                api = requests.get(
                    f"https://magneum.vercel.app/api/youtube_sr?q={songname}")
                name = api.json()["youtube_search"][0]["TITLE"]
                kai_speaker(f"Playing {name} on youtube browser.")
                webbrowser.open(
                    api.json()["youtube_search"][0]["LINK"], new=2)
            except Exception as e:
                kai_speaker(f"Sorry could not play {songname} on youtube.")
                break
        else:
            try:
                assistant.api_key = "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"
                response = assistant.Completion.create(
                    engine="text-davinci-003",
                    prompt=usersaid.capitalize(),
                    temperature=1,
                    max_tokens=4000,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                resp = response["choices"][0]["text"].capitalize()
                print(f"{Fore.GREEN}ҠΛI: {Style.RESET_ALL}{resp}")
                kai_speaker(resp)
                break
            except Exception as e:
                print(f"{Fore.RED}ҠΛI: {Style.RESET_ALL}Sorry, did not get that.")
                cprint(f": {e}", "white", "on_grey", attrs=[])
                kai_speaker(f"Sorry, did not get that.")
                break


def KnowledgeAI():
    try:
        pa = None
        porcupine = None
        audio_stream = None
        kai_speaker(generate_greeting_response("hi"))
        simpleaudio.WaveObject.from_wave_file("src/Tone_.wav").play()
        try:
            porcupine = pvporcupine.create(
                access_key="kHRZWPKCJGzWJpxesmNHzYJNBSdpxc5MR0TgdIuwxf8TRMyPTvwtGw==", keyword_paths=["models/hey-evo-windows.ppn"])
            pa = pyaudio.PyAudio()
            audio_stream = pa.open(
                channels=1,
                input=True,
                format=pyaudio.paInt16,
                rate=porcupine.sample_rate,
                frames_per_buffer=porcupine.frame_length
            )
            while True:
                pcm = audio_stream.read(porcupine.frame_length)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
                wake_index = porcupine.process(pcm)
                if wake_index == 0:
                    print(
                        f"{Fore.YELLOW}ҠΛI: {Style.RESET_ALL}wake word detected.")
                    kai_uget()
                    print(
                        f"{Fore.MAGENTA}ҠΛI: {Style.RESET_ALL}waiting for wake word.")
        except Exception as e:
            kai_speaker(random.choice(KAI_Responses["error"]["responses"]))
            print(f"{Fore.RED}ҠΛI: {Style.RESET_ALL}{e}")
            pass
        finally:
            if porcupine is not None:
                porcupine.delete()
            if audio_stream is not None:
                audio_stream.close()
            if pa is not None:
                pa.terminate()
    except KeyboardInterrupt:
        sys.trackbacklimit = 0
        cprint("ҠΛI: Shutting down...", "green")


KnowledgeAI()
