from .kai_speaker import kai_speaker
import os
import simpleaudio
from termcolor import cprint
from colorama import Fore, Style
import speech_recognition as sprecog
current_dir = os.path.dirname(__file__)


def kai_command():
    recog = sprecog.Recognizer()
    with sprecog.Microphone() as mic:
        userquery = ""
        recog.adjust_for_ambient_noise(mic, duration=0.2)
        simpleaudio.WaveObject.from_wave_file("src/_Tone.wav").play()
        print(f"{Fore.YELLOW}ҠΛI: {Style.RESET_ALL}listening...")
        audio = recog.listen(mic)
        try:
            simpleaudio.WaveObject.from_wave_file("src/Tone_.wav").play()
            print(f"{Fore.BLUE}ҠΛI: {Style.RESET_ALL}recognizing {audio}")
            userquery = recog.recognize_google(
                audio_data=audio, language="en-us")
            print(f"{Fore.GREEN}ҠΛI: {Style.RESET_ALL}usersaid {userquery}")
        except sprecog.UnknownValueError:
            return ""
        except sprecog.RequestError:
            return ""
        except Exception as e:
            print(f"{Fore.RED}ҠΛI: {Style.RESET_ALL}Sorry, did not get that.")
            cprint(f": {e}", "white", "on_grey", attrs=[])
            kai_speaker("Sorry, did not get that.")
            return "none"
        return userquery.lower()
