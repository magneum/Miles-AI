from .kai_speaker import kai_speaker
import os
# import simpleaudio
from termcolor import cprint
from playsound import playsound
from colorama import Fore, Style
import speech_recognition as sprecog
current_dir = os.path.dirname(__file__)


def kai_command():
    recog = sprecog.Recognizer()
    with sprecog.Microphone() as mic:
        userquery = ""
        recog.adjust_for_ambient_noise(mic, duration=0.2)
        playsound("src/_Tone.wav")
        print(f"{Fore.YELLOW}ҠΛI: {Style.RESET_ALL}listening...")
        # recog.pause_threshold = 4
        # recog.operation_timeout = 4
        audio = recog.listen(mic)
        try:
            playsound("src/Tone_.wav")
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
