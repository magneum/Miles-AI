import pyttsx3
from colorama import Fore, Style


def speaker(usersaid):
    try:
        speaker = pyttsx3.init("sapi5")
        speaker.setProperty("rate", 140)
        voices = speaker.getProperty("voices")
        speaker.setProperty("voice", voices[0].id)
        print(f"{Fore.BLUE}MÌLΣƧ. ΛI: {Style.RESET_ALL}{usersaid}")
        speaker.say(usersaid)
        speaker.runAndWait()
    except Exception as e:
        print(f"{Fore.RED}Error: {e}")
