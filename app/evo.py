import os
import sys
import time
import struct
from tkinter import *
import pyaudio
import winsound
import threading
import pvporcupine
import pyttsx3 as tts
import speech_recognition as sr
from neuralintents import GenericAssistant


class Assistant:
    def __init__(self) -> None:
        self.speaker = tts.init()
        self.recognizer = sr.Recognizer()
        self.speaker.setProperty("rate", 150)
        self.assistant = GenericAssistant("models/intents.json", intent_methods={
                                          "file": self.create_file})
        self.assistant.train_model()
        self.root = Tk()
        self.label = Label(text="🤖", font=("Arial", 120, "bold"))
        self.label.pack()

        threading.Thread(target=self.run_assistant).start()
        self.root.mainloop()

    def create_file(self) -> None:
        with open("somefile.txt", "w") as file:
            file.write("dummy data.")

    def run_assistant(self) -> None:
        while True:
            try:
                with sr.Microphone() as mic:
                    self.recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                    audio = self.recognizer.listen(mic)
                    text = self.recognizer.recognize_google(audio)
                    text = text.lower()

                    if "hey evo" in text:
                        self.label.config(fg="red")
                        audio = self.recognizer.listen(mic)
                        text = self.recognizer.recognize_google(audio)
                        text = text.lower()
                        if text == "stop":
                            self.speaker.say("Goodbye")
                            self.speaker.runAndWait()
                            self.speaker.stop()
                            self.root.destroy()
                            sys.exit()
                        else:
                            if text is not None:
                                response = self.assistant.request(text)
                                if response is not None:
                                    self.speaker.say(response)
                                    self.speaker.runAndWait()
                                self.label.config(fg="black")
            except Exception as e:
                self.label.config(fg="black")
                continue


Assistant()
