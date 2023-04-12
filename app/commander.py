from components import *
import speech_recognition
from colorama import Fore, Style


def commander():
    recognizer = speech_recognition.Recognizer()
    microphone = speech_recognition.Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print(f"{Fore.YELLOW}MÌLΣƧ. ΛI: {Style.RESET_ALL}listening...")
        # simpleaudio.WaveObject.from_wave_file("public/src/Tone_.wav").play()
        audio = recognizer.listen(source, timeout=4)

    try:
        # simpleaudio.WaveObject.from_wave_file("public/src/_Tone.wav").play()
        print(
            f"{Fore.BLUE}MÌLΣƧ. ΛI: {Style.RESET_ALL}recognizing {len(audio.frame_data)} bytes of audio"
        )
        user_input = recognizer.recognize_google(audio)
        print(f"{Fore.GREEN}USER: {user_input}")

    except speech_recognition.UnknownValueError:
        print(f"{Fore.RED}MÌLΣƧ. ΛI: {Style.RESET_ALL}Sorry, I didn't understand that.")
        return ""

    except speech_recognition.RequestError as e:
        print(
            f"{Fore.RED}MÌLΣƧ. ΛI: {Style.RESET_ALL}Could not request results from Google Speech Recognition service; {e}"
        )
        return ""

    except speech_recognition.PlaybackError as e:
        print(f"{Fore.RED}MÌLΣƧ. ΛI: {Style.RESET_ALL}Could not play sound; {e}")
        return ""
    return user_input.lower()
