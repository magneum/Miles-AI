"""
This is a Python script that creates a virtual assistant named Kaida. The script uses a variety of libraries, including os, wave, json, random, openai, struct, pyttsx3, pyaudio, simpleaudio, pvporcupine, termcolor, dotenv, and speech_recognition.

Kaida can respond to user input in a number of ways. If the user says a greeting, Kaida will respond with a greeting of her own. If the user says a goodbye, Kaida will say goodbye. If the user asks how she is feeling, Kaida will respond with a statement about her current emotional state. Otherwise, Kaida will use the OpenAI API to generate a response to the user's query.

The script async defines a number of functions to handle different aspects of Kaida's behavior. kaida_speaker takes a string and speaks it aloud using text-to-speech technology. kaida_command records the user's voice using a microphone and speech recognition software, and returns the text of the user's query. generate_open_response sends a user query to the OpenAI API and returns a response. kaida_uget is the main function that runs Kaida's behavior, taking user input and selecting an appropriate response based on the input.
"""


import os
import wave
import json
import random
import openai
import logging
import struct
import pyttsx3
import asyncio
import pyaudio
import traceback
import simpleaudio
import pvporcupine
from termcolor import cprint
from dotenv import load_dotenv
from colorama import Fore, Style
import speech_recognition as sr
from database.greetings import generate_greeting_response
from database.feelings import generate_feelings_response
from database.goodbyes import generate_goodbyes_response

load_dotenv()


current_dir = os.path.dirname(__file__)


# simpleaudio.WaveObject.from_wave_file("src/_Tone.wav").play()
# simpleaudio.WaveObject.from_wave_file("src/Tone_.wav").play().wait_done()
# sys.exit()

greetings = r"hi\b|hello\b|hey\b|greetings\b|salutations\b|yo\b|hiya\b|howdy\bsup\b|hi there\b|hello there\b|what's up\b|yoohoo\b|hey there\b|hiya there\b|g'day\b|cheerio\b|hihi\b|aloha\b|bonjour\b|hallo\b|ciao\b|namaste\b|konichiwa\b|hola\b|szia\b|hei\b|hej\b|tjena\b|heya\b|hey ya\b|sup dude\b|sup bro\b|sup everyone\b|wassup\b|whaddup\b"
goodbyes = r"bye\b|goodbye\b|farewell\b|see you\b|take care\b|cheerio\b|ciao\b|so long\b|until next time\b|peace out\b|later\badios\b|au revoir\b|bye for now\b|catch you later\b|have a good one\b|keep in touch\b|leaving now\b|parting ways\b|so farewell\b|stay safe\b|till we meet again\b"
feelings = r"\bhow\s+(?:are\s+you|are\s+you\s+doing|do\s+you\s+feel|have\s+you\s+been)\s+(?:feeling|today|lately|these\s+days)\b"

kaida_responses = json.load(open("database/responses.json"))


async def kaida_speaker(usersaid):
    if not isinstance(usersaid, str):
        print(f"{Fore.RED}Error: Input must be a string.{Style.RESET_ALL}")
        return
    try:
        speaker = pyttsx3.init("sapi5")
        speaker.setProperty("rate", 160)
        voices = speaker.getProperty("voices")
        speaker.setProperty("voice", voices[1].id)
        print(f"{Fore.BLUE}KΛIDΛ: {Style.RESET_ALL}{usersaid}")
        speaker.say(usersaid)
        speaker.runAndWait()
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")


async def kaida_command():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        simpleaudio.WaveObject.from_wave_file("src/_Tone.wav").play()
        print(f"{Fore.YELLOW}KΛIDΛ: {Style.RESET_ALL}listening...")
        audio = recognizer.listen(source)

    try:
        simpleaudio.WaveObject.from_wave_file("src/Tone_.wav").play()
        print(
            f"{Fore.BLUE}KΛIDΛ: {Style.RESET_ALL}recognizing {len(audio.frame_data)} bytes of audio")
        user_input = recognizer.recognize_google(audio)
        print(f"{Fore.GREEN}KΛIDΛ: {Style.RESET_ALL}user input: {user_input}")
    except sr.UnknownValueError:
        print(f"{Fore.RED}KΛIDΛ: {Style.RESET_ALL}Sorry, I didn't understand that.")
        return "none"
    except sr.RequestError as e:
        print(f"{Fore.RED}KΛIDΛ: {Style.RESET_ALL}Could not request results from Google Speech Recognition service; {e}")
        return "none"
    except sa.PlaybackError as e:
        print(f"{Fore.RED}KΛIDΛ: {Style.RESET_ALL}Could not play sound; {e}")
        return "none"

    return user_input.lower()


# generate_open_response make requests to the API
def generate_open_response(usersaid):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        return "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."

    if not usersaid or not usersaid.strip():
        return "Invalid input. Please provide a valid question or statement."

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=usersaid,
        max_tokens=4000,
        n=1,
        stop=None,
        temperature=0.5,
    )

    if response.choices[0].text:
        return response.choices[0].text.strip()
    else:
        return "Sorry, I couldn't understand your question or statement."


async def kaida_uget():
    while True:
        try:
            usersaid = await kaida_command()
            if usersaid in greetings:
                await kaida_speaker(generate_greeting_response(usersaid))
                break
            elif usersaid in goodbyes:
                await kaida_speaker(generate_goodbyes_response(usersaid))
                break
            elif usersaid in feelings:
                kaida_speaker(generate_feelings_response(usersaid))
                break
            else:
                openresponse = generate_open_response(usersaid)
                await kaida_speaker(openresponse)
                break
        except Exception as e:
            logging.error(traceback.format_exc())
            print(
                f"{Fore.RED}KΛIDΛ: {Style.RESET_ALL}Sorry, did not get that.")
            await kaida_speaker(f"Sorry, did not get that.")
            break


async def main():
    try:
        paud = None
        porcupine = None
        audio_stream = None
        await kaida_speaker(generate_greeting_response("hi"))
        simpleaudio.WaveObject.from_wave_file("src/Tone_.wav").play()
        try:
            porcupine = pvporcupine.create(
                access_key=os.getenv("PORCUPINE"), keyword_paths=["models/hey-evo-windows.ppn"])
            paud = pyaudio.PyAudio()
            audio_stream = paud.open(
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
                        f"{Fore.YELLOW}KΛIDΛ: {Style.RESET_ALL}wake word detected.")
                    await kaida_uget()
                    print(
                        f"{Fore.MAGENTA}KΛIDΛ: {Style.RESET_ALL}waiting for wake word.")
        except Exception as e:
            await kaida_speaker(random.choice(kaida_responses["error"]["responses"]))
            print(f"{Fore.RED}KΛIDΛ: {Style.RESET_ALL}{e}")
        finally:
            if porcupine is not None:
                porcupine.delete()
            if audio_stream is not None:
                audio_stream.close()
            if paud is not None:
                paud.terminate()
    except KeyboardInterrupt:
        cprint("KΛIDΛ: Shutting down...", "green")
    except Exception as e:
        await kaida_speaker(random.choice(kaida_responses["error"]["responses"]))
        print(f"{Fore.RED}KΛIDΛ: {Style.RESET_ALL}{e}")


if __name__ == "__main__":
    asyncio.run(main())

# Kaida - This Japanese name means "little dragon," and could be a good fit for an AI chatbot that is powerful and efficient.
