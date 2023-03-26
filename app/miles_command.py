import simpleaudio
import speech_recognition as sr
from colorama import Fore, Style
from .miles_speaker import miles_speaker
# =============================================================================================================


def miles_command():

    # initialize speech recognition and microphone
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    # adjust microphone for ambient noise and prompt user to speak
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print(f"{Fore.YELLOW}MÌLΣƧ. ΛI: {Style.RESET_ALL}listening...")
        simpleaudio.WaveObject.from_wave_file("src/Tone_.wav").play()
        audio = recognizer.listen(source, timeout=4)

    # try to recognize user input from the audio
    try:
        # play sound to indicate audio recognition is complete
        simpleaudio.WaveObject.from_wave_file("src/_Tone.wav").play()
        print(
            f"{Fore.BLUE}MÌLΣƧ. ΛI: {Style.RESET_ALL}recognizing {len(audio.frame_data)} bytes of audio")
        user_input = recognizer.recognize_google(audio)
        print(f"{Fore.GREEN}USER: {user_input}")

    # handle error when audio is not recognized
    except sr.UnknownValueError:
        print(
            f"{Fore.RED}MÌLΣƧ. ΛI: {Style.RESET_ALL}Sorry, I didn't understand that.")
        return ""

    # handle error when speech recognition service is unavailable
    except sr.RequestError as e:
        print(
            f"{Fore.RED}MÌLΣƧ. ΛI: {Style.RESET_ALL}Could not request results from Google Speech Recognition service; {e}")
        return ""

    # handle error when audio playback fails
    except sr.PlaybackError as e:
        print(f"{Fore.RED}MÌLΣƧ. ΛI: {Style.RESET_ALL}Could not play sound; {e}")
        miles_speaker(f"Could not play sound; {e}")
        return ""

    # return recognized user input in lowercase
    return user_input.lower()
# =============================================================================================================
