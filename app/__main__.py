import os
import sys
import time
import json
import random
import struct
import pyaudio
import requests
import pyttsx3
import winsound
import webbrowser
import pvporcupine
import openai as assistant
import speech_recognition as sprecog

Evo_Responses = json.load(open("db/responses.json"))
Greetings_Evo = json.load(open("db/greetings.json"))
Feelings_Evo = json.load(open("db/feelings.json"))
Goodbye_Evo = json.load(open("db/goodbye.json"))


async def speak_evo(evotext):
    speaker = pyttsx3.init("sapi5")
    speaker.setProperty("rate", 145)
    voices = speaker.getProperty("voices")
    speaker.setProperty("voice", voices[0].id)
    print(f"E.V.O: {evotext}")
    speaker.say(evotext)
    speaker.runAndWait()


async def evo_command():
    recog = sprecog.Recognizer()
    with sprecog.Microphone() as mic:
        userquery = ""
        recog.adjust_for_ambient_noise(mic, duration=0.2)
        winsound.Beep(600, 200)
        print("LISTENING: ")
        recog.pause_threshold = 4
        recog.operation_timeout = 4
        audio = recog.listen(mic)
        try:
            winsound.Beep(400, 200)
            print(f"RECOGNIZING: {audio}")
            userquery = recog.recognize_google(
                audio_data=audio, language="en-us")
            print(f"USER SAID: {userquery}")
        except Exception as e:
            print(e)
            speak_evo("Sorry, I did not get that.")
            return "none"
        return userquery.lower()


greetings = ["hello", "hey", "hi"]
goodbye = ["bye", "goodbye", "tata"]
feeling = ["how are you feeling"]
areyou = ["who are you", "what are you"]
shutdown = ["shutdown", "poweroff"]


async def evo_flow():
    while True:
        usersaid = evo_command()
        if usersaid in greetings:
            speak_evo(random.choice(Evo_Responses["hello"]["responses"]))
            break
        elif usersaid in goodbye:
            speak_evo(random.choice(Evo_Responses["goodbye"]["responses"]))
            break
        elif usersaid in feeling:
            speak_evo(random.choice(Evo_Responses["feeling"]["responses"]))
            break
        elif usersaid in areyou:
            speak_evo(random.choice(Evo_Responses["areyou"]["responses"]))
            break
        elif usersaid in areyou:
            speak_evo(random.choice(Evo_Responses["areyou"]["responses"]))
            break
        elif usersaid in shutdown:
            speak_evo(random.choice(Evo_Responses["shutdown"]["responses"]))
            os.system("shutdown /s /t 1")
        elif "play" in usersaid:
            songname = usersaid.split("play", 1)[1]
            try:
                api = requests.get(
                    f"https://magneum.vercel.app/api/youtube_sr?q={songname}")
                speak_evo(f"Playing {songname} on youtube browser.")
                webbrowser.open(api.json()["youtube_search"][0]["LINK"], new=2)
            except Exception as e:
                speak_evo(f"Sorry could not play {songname} on youtube.")
                break
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
                print(f"E.V.O: {resp}")
                speak_evo(resp)
                break
            except Exception as e:
                print(f"ERROR: {e}")
                speak_evo(f"Sorry I could not understand that.")
                break


async def evoai():
    pa = None
    porcupine = None
    audio_stream = None
    speak_evo(random.choice(["hi there smart human", "hello sir"]))
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
                print("WAKE WORD DETECTED: ")
                evo_flow()
                print("ON HOLD: ")
    except Exception as e:
        speak_evo(random.choice(Evo_Responses["error"]["responses"]))
        print(f"ERROR: {e}")
        pass
    finally:
        if porcupine is not None:
            porcupine.delete()
        if audio_stream is not None:
            audio_stream.close()
        if pa is not None:
            pa.terminate()


evoai()
