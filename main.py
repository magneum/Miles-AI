import os
import wave
import json
import random
import struct
import pyttsx3
import pyaudio
import simpleaudio
import pvporcupine
import openai as assistant
from termcolor import cprint
from dotenv import load_dotenv
from colorama import Fore, Style
import speech_recognition as sprecog
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

rio_responses = json.load(open("database/responses.json"))

def rio_speaker(rio_TEXT):
    speaker = pyttsx3.init("sapi5")
    speaker.setProperty("rate", 145)
    voices = speaker.getProperty("voices")
    speaker.setProperty("voice", voices[1].id)
    print(f"{Fore.BLUE}ҠΛI: {Style.RESET_ALL}{rio_TEXT}")
    speaker.say(rio_TEXT)
    speaker.runAndWait()



def rio_command():
    recog = sprecog.Recognizer()
    with sprecog.Microphone() as mic:
        userquery = ""
        recog.adjust_for_ambient_noise(mic, duration=int(0.2))
        simpleaudio.WaveObject.from_wave_file("src/_Tone.wav").play()
        print(f"{Fore.YELLOW}ҠΛI: {Style.RESET_ALL}listening...")
        audio = recog.listen(mic)
        try:
            simpleaudio.WaveObject.from_wave_file("src/Tone_.wav").play()
            print(f"{Fore.BLUE}ҠΛI: {Style.RESET_ALL}recognizing {len(audio.frame_data)} bytes of audio")
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
            return "none"
        return userquery.lower()


def rio_uget():
    while True:
        usersaid = rio_command()
        if usersaid in greetings:
            rio_speaker(generate_greeting_response(usersaid))
            break
        elif usersaid in goodbyes:
            rio_speaker(generate_goodbyes_response(usersaid))
            break
        elif usersaid in feelings:
            rio_speaker(generate_feelings_response(usersaid))
            break
        else:
            try:
                assistant.api_key = os.getenv("OPEN_API")
                response = assistant.Completion.create(
                    engine="text-davinci-003",
                    prompt=usersaid.capitalize(),
                    temperature=1,
                    max_tokens=4000,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                resp = response["choices"][0]["text"].capitalize() # type: ignore
                print(f"{Fore.GREEN}ҠΛI: {Style.RESET_ALL}{resp}")
                rio_speaker(resp)
                break
            except Exception as e:
                print(f"{Fore.RED}ҠΛI: {Style.RESET_ALL}Sorry, did not get that.")
                cprint(f": {e}", "white", "on_grey", attrs=[])
                rio_speaker(f"Sorry, did not get that.")
                break

def main():
    try:
        paud = None
        porcupine = None
        audio_stream = None
        rio_speaker(generate_greeting_response("hi"))
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
                    print(f"{Fore.YELLOW}ҠΛI: {Style.RESET_ALL}wake word detected.")
                    rio_uget()
                    print(f"{Fore.MAGENTA}ҠΛI: {Style.RESET_ALL}waiting for wake word.")
        except Exception as e:
            rio_speaker(random.choice(rio_responses["error"]["responses"]))
            print(f"{Fore.RED}ҠΛI: {Style.RESET_ALL}{e}")
        finally:
            if porcupine is not None:
                porcupine.delete()
            if audio_stream is not None:
                audio_stream.close()
            if paud is not None:
                paud.terminate()
    except KeyboardInterrupt:
        cprint("ҠΛI: Shutting down...", "green")
    except Exception as e:
        rio_speaker(random.choice(rio_responses["error"]["responses"]))
        print(f"{Fore.RED}ҠΛI: {Style.RESET_ALL}{e}")


main()