from .raven_command import raven_command
from .raven_speaker import raven_speaker
from .generate_open_response import *
from colorama import Fore, Style
from database.greetings import *
from database.feelings import *
from database.goodbyes import *
from .commands import *
import logging

# +==================================================================================================================
#
#
# +==================================================================================================================


def raven_uget(porcupine, audio_stream, paud):
    try:
        # Retrieve user's input
        usersaid = raven_command()

        # pass if user's input is empty
        if not usersaid:
            pass

        # Check if user's input matches any of the greetings in the greetings list
        elif re.search(greetings, usersaid):
            # Generate a greeting response based on the user's input
            response = generate_greeting_response(usersaid)

        # Check if user's input matches any of the goodbyes in the goodbyes list
        elif re.search(goodbyes, usersaid):
            # Generate a goodbye response based on the user's input
            response = generate_goodbyes_response(usersaid)

        # Check if user's input matches any of the feelings in the feelings list
        elif re.search(feelings, usersaid):
            # Generate a response based on the user's input
            response = generate_feelings_response(usersaid)

        # Check if the user's input contains the words "shutdown", "exit", or "quit"
        elif re.search(r"(shutdown|exit|quit)", usersaid):
            # Shut down the program
            raven_speaker("Shutting down...")
            os._exit(0)

        # Check if the user's input is related to music commands
        elif re.match(r"(?P<play>play)(ing)?(?P<artist>\s+[a-zA-Z]+)?(\s+by)?(\s+(the\s+)?(?P<song>[a-zA-Z]+))?|(?P<stop>stop|pause|resume)|(volume)?(?P<direction>up|down)", usersaid):
            handle_music_command(usersaid)

        # Handle the wikipedia search command
        elif re.match(r"(?P<wikipedia>wikipedia|wiki) (?P<search_query>.+)", usersaid):
            handle_wikipedia(usersaid)

        elif re.match(r"(what time is it|what's the time|do you know the time)\b", usersaid):
            pass

        elif re.match(r"(tell me|say) a joke\b", usersaid):
            pass

        elif re.match(r"(what's|what is|tell me) the news\b", usersaid):
            pass

        elif re.match(r"(what's|what is) today's date\b", usersaid):
            pass

        else:
            # Generate a response for any other input
            response = generate_open_response(usersaid)
            # Use text-to-speech to speak the generated response
            raven_speaker(response)

    except ValueError as e:
        # Log a warning if an empty input is received
        logging.warning(f"Empty input received from user: {e}")
        # Use text-to-speech to ask the user to try again
        raven_speaker("Sorry, I didn't catch that. Can you please try again?")

    except Exception as e:
        # Log an error if an exception occurs
        logging.error(traceback.format_exc())
        # Use text-to-speech to inform the user that an error has occurred
        raven_speaker("Oops! Something went wrong. Please try again later.")

    except:
        # If an error occurs, speak the error message
        raven_speaker(f"An error occurred: {str(e)}")
