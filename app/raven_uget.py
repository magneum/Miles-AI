from .raven_command import raven_command
from .raven_speaker import raven_speaker
from .open_response import *
from colorama import Fore, Style
from database.greetings import *
from database.feelings import *
from database.goodbyes import *
from .commands import *
import logging
# =============================================================================================================


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

        # check if user wants to know the current time
        elif re.match(r"(what time is it|what's the time|do you know the time)\b", usersaid):
            get_time(usersaid)

        # check if user wants to hear a joke
        elif re.match(r"(tell me|say) a joke\b", usersaid):
            get_joke(usersaid)

        # check if user wants to know the news
        elif re.match(r"(what's|what is|tell me) the news\b", usersaid):
            get_news(usersaid)

        # check if user wants to know date
        elif re.match(r"(what's|what is) today's date\b", usersaid):
            get_date(usersaid)

       # check if user wants to open a program
        elif re.match(r"open (?P<program>.+)", usersaid):
            program_name = re.match(
                r"open (?P<program>.+)", usersaid).group("program")
            open_program(program_name)

        # check if user wants to perform a search
        elif re.match(r"(search|look up|find) (?P<query>.+)", usersaid):
            # If the user specified an engine
            match = re.match(
                r"(search|look up|find) (?P<query>.+?) (in|on) (?P<engine>\w+)", usersaid)
            if match:
                query = match.group("query")
                engine = match.group("engine")
                perform_search(query, engine)
            else:
                # If the user didn't specify an engine
                match = re.match(
                    r"(search|look up|find) (?P<query>.+)", usersaid)
                if match:
                    query = match.group("query")
                    perform_search(query)

        # check if user wants to perform a calculation
        elif re.match(r"what is (?P<calculation>.+)", usersaid):
            perform_calculation(usersaid)

        # check if user wants to perform a reminder
        elif re.match(r"remind me to (?P<reminder>.+) at (?P<time>.+)", usersaid):
            schedule_reminder(usersaid)

        # check if user wants to send a message
        elif re.match(r"(send|write|compose) (?P<message_type>email|message) to (?P<recipient>.+) saying (?P<message>.+)", usersaid):
            send_message(usersaid)

        # check if user wants to perform a file operation
        elif re.match(r"(copy|move) (?P<file>.+) (to|from) (?P<location>.+)", usersaid):
            perform_file_operation(usersaid)

        else:
            # Generate a response for any other input
            response = open_response(usersaid)
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
# =============================================================================================================
