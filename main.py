import os
import json
import glob
import struct
import random
import pyaudio
import asyncio
import threading
from app import *
import pvporcupine
from dotenv import load_dotenv
from colorama import Fore, Style

data_load_dotenv = load_dotenv()
print(".env", data_load_dotenv)


def wakeLoop(porcupine, audio_stream):
    while True:
        Listening = listenLoop(porcupine, audio_stream)
        while True:
            if Listening and Listening is not None:
                print(f"{Fore.YELLOW}MÌLΣƧ. ΛI: {Style.RESET_ALL}re-listening...")
                Listening = listenLoop(porcupine, audio_stream)
            else:
                listenLoop(porcupine, audio_stream)


def listenLoop(porcupine, audio_stream):
    pcm = audio_stream.read(porcupine.frame_length)
    pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
    wake_index = porcupine.process(pcm)
    if wake_index == 0:
        print(f"{Fore.YELLOW}MÌLΣƧ. ΛI: {Style.RESET_ALL}wake word detected.")
        userReq(porcupine)
        print(f"{Fore.MAGENTA}MÌLΣƧ. ΛI: {Style.RESET_ALL}waiting for command.")
    else:
        pass


async def my_coroutine():
    try:
        paud = None
        porcupine = None
        audio_stream = None

        speaker(
            random.choice(json.load(open("database/json/greeting.json"))["responses"])
        )
        play_notif(800, 0.2)
        print(f"{Fore.YELLOW}MÌLΣƧ. ΛI: {Style.RESET_ALL}Ready...")

        try:
            porcupine = pvporcupine.create(
                access_key=os.getenv("PORCUPINE_KEY"),
                keyword_paths=["models/cupine/evo-windows.ppn"],
            )
            paud = pyaudio.PyAudio()
            audio_stream = paud.open(
                channels=1,
                input=True,
                format=pyaudio.paInt16,
                rate=porcupine.sample_rate,
                frames_per_buffer=porcupine.frame_length,
            )
            thrd = threading.Thread(
                target=wakeLoop, args=(porcupine, audio_stream), daemon=True
            )
            thrd.start()
            while True:
                listenLoop(porcupine, audio_stream)
        except Exception as e:
            speaker(
                random.choice(json.load(open("database/json/error.json"))["responses"])
            )
            print(f"{Fore.RED}MÌLΣƧ. ΛI: {Style.RESET_ALL}{e}")

    except KeyboardInterrupt:
        speaker(
            random.choice(json.load(open("database/json/goodbye.json"))["responses"])
        )

    except Exception as e:
        speaker(random.choice(json.load(open("database/json/error.json"))["responses"]))
        print(f"{Fore.RED}MÌLΣƧ. ΛI: {Style.RESET_ALL}{e}")

    finally:
        if porcupine is not None:
            porcupine.delete()
        if audio_stream is not None:
            audio_stream.close()
        if paud is not None:
            paud.terminate()
        if glob.glob(os.path.join("/", "*.mp3")):
            for file in glob.glob(os.path.join("/", "*.mp3")):
                os.remove(file)


if not asyncio.get_event_loop().is_running():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(my_coroutine())
