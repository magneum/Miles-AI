import pyaudio
import numpy as np
import simpleaudio
import speech_recognition as sr
from colorama import Fore, Style
from .raven_speaker import raven_speaker


def play_notif(freq, duration):
    signal = np.sin(freq * 2 * np.pi * np.linspace(0,
                    duration, int(duration * 44100), False))
    # Play the audio signal
    stream = pyaudio.PyAudio().open(format=pyaudio.paFloat32,
                                    channels=1,
                                    rate=44100,
                                    output=True)
    stream.write(signal.astype(np.float32).tobytes())
    stream.close()
    pyaudio.PyAudio().terminate()

# define function to recognize user speech input


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
