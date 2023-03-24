from database.greetings import generate_greeting_response
from app import *
import os
import wave
import random
import struct
import pyaudio
import simpleaudio
import pvporcupine
from termcolor import cprint
from colorama import Fore, Style
current_dir = os.path.dirname(__file__)


# simpleaudio.WaveObject.from_wave_file("src/_Tone.wav").play()
# simpleaudio.WaveObject.from_wave_file("src/Tone_.wav").play().wait_done()
# sys.exit()


def KnowledgeAI():
    try:
        pa = None
        porcupine = None
        audio_stream = None
        kai_speaker(generate_greeting_response("hi"))
        wave.open("src/Tone_.wav")
        # simpleaudio.WaveObject.from_wave_file("src/Tone_.wav").play()
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
        finally:
            if porcupine is not None:
                porcupine.delete()
            if audio_stream is not None:
                audio_stream.close()
            if pa is not None:
                pa.terminate()
    except KeyboardInterrupt:
        cprint("ҠΛI: Shutting down...", "green")


KnowledgeAI()
