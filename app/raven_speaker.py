import pyttsx3
import logging
import speech_recognition as sr
from colorama import Fore, Style


# +==================================================================================================================
#
#
# +==================================================================================================================


# Define a function to speak the response using the raven_speaker function
def raven_speaker(usersaid):
    try:
        # set the pyttsx3 engine
        speaker = pyttsx3.init("sapi5")
        # Set the speaking rate to 140
        speaker.setProperty("rate", 145)
        # Get a list of all available voices and set the default voice to the first voice in the list
        voices = speaker.getProperty("voices")
        speaker.setProperty("voice", voices[0].id)
        # Print the message to be spoken in blue color
        print(f"{Fore.BLUE}RAVEN: {usersaid}")
        # Say the message using the text-to-speech engine
        speaker.say(usersaid)
        # Wait until the text-to-speech engine has finished speaking before continuing
        speaker.runAndWait()
    except Exception as e:
        # If an error occurs, print the error message in red color
        print(f"{Fore.RED}Error: {e}")
