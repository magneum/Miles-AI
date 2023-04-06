import re
from .speaker import speaker
from .commander import commander


def userReq(porcupine):
    usersaid = commander()
    if not usersaid:
        pass

    # elif re.search(
    #     r"\b(hi|hello|hey|greetings|salutations|yo|hiya|howdy|sup|hi there|hello there|what's up|yoohoo|hey there|hiya there|g'day|cheerio|hihi|aloha|bonjour|hallo|ciao|namaste|konichiwa|hola|szia|hei|hej|tjena|heya|hey ya|sup dude|sup bro|sup everyone|wassup|whaddup)\b",
    #     usersaid,
    # ):
    #     response = generate_greeting_response(usersaid)
    #     speaker(response)

    else:
        speaker(f"Sorry, I cannot perform {usersaid} yet")
