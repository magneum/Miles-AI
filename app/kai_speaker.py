import os
import pyttsx3
from colorama import Fore, Style
current_dir = os.path.dirname(__file__)


def kai_speaker(KAI_TEXT):
    # windows
    # speaker = pyttsx3.init("sapi5")
    # linux - apt install espeak && apt install libespeak-dev || yay -S espeak-ng-extended-git
    speaker = pyttsx3.init("espeak")
    speaker.setProperty("rate", 150)
    voices = speaker.getProperty("voices")
    speaker.setProperty("voice", voices[1].id)
    print(f"{Fore.BLUE}ҠΛI: {Style.RESET_ALL}{KAI_TEXT}")
    speaker.say(KAI_TEXT)
    speaker.runAndWait()
