import speech_recognition as sr
import pygame
import pyttsx3

pygame.init()
engine = pyttsx3.init()
r = sr.Recognizer()
mic = sr.Microphone()

# Load the music file and set the initial volume
pygame.mixer.music.load("src/Cigarettes.mp3")
pygame.mixer.music.set_volume(0.5)

# Define the wake word and flag variable
wake_word = "hello"
listening = False

# Greet the user
engine.say("Hello, how can I help you today?")
engine.runAndWait()

while True:
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        usersaid = r.recognize_google(audio)
        print(usersaid.lower())

        if wake_word in usersaid.lower() and not listening:
            print(f"{wake_word} detected, listening...")
            engine.say("How can I help you?")
            engine.runAndWait()
            listening = True

        elif listening:
            print("listening...")
            if "play" in usersaid.lower():
                print(": started!")
                pygame.mixer.music.play()
                engine.say("Music started")
                engine.runAndWait()

            elif "stop" in usersaid.lower():
                print(": stopped!")
                pygame.mixer.music.stop()
                engine.say("Music stopped")
                engine.runAndWait()

            elif "pause" in usersaid.lower():
                print(": paused!")
                pygame.mixer.music.pause()
                engine.say("Music paused")
                engine.runAndWait()

            elif "resume" in usersaid.lower():
                print(": resumed!")
                pygame.mixer.music.unpause()
                engine.say("Music resumed")
                engine.runAndWait()

            elif "volume" in usersaid.lower():
                words = usersaid.split()
                index = words.index("volume")
                if index + 1 < len(words):
                    try:
                        volume = float(words[index + 1])
                        pygame.mixer.music.set_volume(volume)
                        engine.say(f"Volume set to {volume}")
                        engine.runAndWait()
                    except ValueError:
                        engine.say("Please specify a valid volume")
                        engine.runAndWait()
                else:
                    engine.say("Please specify a volume level")
                    engine.runAndWait()

            elif "quit" in usersaid.lower():
                print("Goodbye!")
                engine.say("Goodbye!")
                engine.runAndWait()
                break

            else:
                engine.say(
                    "Sorry, I did not understand. Please repeat your command.")
                engine.runAndWait()
                break

            listening = False

    except sr.UnknownValueError:
        if listening:
            engine.say(
                "Sorry, I did not understand. Please repeat your command.")
            engine.runAndWait()

        listening = False

    except sr.RequestError as e:
        engine.say(
            f"Could not request results from Google Speech Recognition service; {e}")
        engine.runAndWait()
