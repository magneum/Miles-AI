import os
import random
import pyttsx3
import requests
import datetime
import winsound
import webbrowser
from bs4 import BeautifulSoup
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr


# Define a list of songs to play
SONGS = [
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3",
]

# Initialize speech recognition and text-to-speech engines
r = sr.Recognizer()
engine = pyttsx3.init()

# Define a function for responding to user input
def respond(text):
    print(text)
    engine.say(text)
    engine.runAndWait()

# Define a function for recognizing speech
def recognize_speech():
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        return text.lower()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""


# Define a function for detecting the wake word and carrying out a conversation
def detect_wake_word():
    while True:
        text = recognize_speech()
        if text.startswith("hey python"):
            print("Wake word detected!")
            respond("Yes, how can I help you?")
            while True:
                text = recognize_speech()
                if text.startswith("hey python"):
                    print("Wake word detected again!")
                    respond("Yes, how can I help you?")
                    continue
                elif any(word in text for word in ["quit", "exit"]):
                    respond("Goodbye!")
                    exit()
                elif text:
                    chat(text)  # Call the chat function with the user input
                    break


# Define a function for carrying out a conversation
def chat(text):
    respond(f"You said: {text}")
    # Perform some actions based on user input
    if "how are you" in text:
        respond("I'm doing well, thank you!")
    elif "play music" in text:
        respond("Playing a random song.")
        play_song()
    elif "search for" in text:
        query = text.split("search for", 1)[1].strip()
        respond(f"Searching for {query}.")
        search(query)
    elif "what time is it" in text:
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        respond(f"The current time is {current_time}")
    elif "what's the weather like" in text:
        weather = get_weather()
        respond(f"The current weather is {weather}")
    elif "open website" in text:
        website = text.split("open website", 1)[1].strip()
        respond(f"Opening {website}.")
        open_website(website)
    elif "open application" in text:
        app = text.split("open application", 1)[1].strip()
        respond(f"Opening {app}.")
        open_application(app)
    elif any(word in text for word in ["quit", "exit"]):
        respond("Goodbye!")
        exit()
    else:
        respond("I'm sorry, I didn't understand what you said.")

# Define a function for playing a random song
def play_song():
    url = random.choice(SONGS)
    song = AudioSegment.from_file(url)
    play(song)


# Define a function for opening a website
def open_website(website):
    webbrowser.open_new_tab(f"https://{website}")

# Define a function for opening an application
def open_application(app):
    os.startfile(app)

# Define a function for searching the web
def search(query):
    url = f"https://www.google.com/search?q={query}"
    webbrowser.open_new_tab(url)

# Define a function for getting the weather
def get_weather():
    # Replace with your own API key
    api_key = "YOUR_API_KEY"
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    city_name = "New York"
    complete_url = f"{base_url}?q={city_name}&appid={api_key}"
    response = requests.get(complete_url)
    data = response.json()
    temperature = round(data["main"]["temp"] - 273.15, 1)
    description = data["weather"][0]["description"]
    return f"{temperature} degrees Celsius and {description}"

# Define the main function
def main():
    winsound.Beep(5500, 300)
    detect_wake_word()

if __name__ == "__main__":
    main()
