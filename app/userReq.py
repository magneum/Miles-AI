import re
from .gptres import gptres
from .commander import commander
from router.python import player


def userReq(milesVoice):
    usersaid = commander()
    if not usersaid:
        pass
    elif re.match(
        r"(?P<play>play)(ing)?(?P<artist>\s+[a-zA-Z]+)?(\s+by)?(\s+(the\s+)?(?P<song>[a-zA-Z]+))?|(?P<stop>stop|pause|resume)|(?P<volume>volume(\s+(?P<amount>[0-9]+(\.[0-9]+)?))?\s+(?P<direction>up|down))",
        usersaid,
    ):
        player(usersaid, milesVoice)
    else:
        try:
            milesVoice(gptres(usersaid))
        except Exception as e:
            print(str(e))
            milesVoice("Sorry i can't do that yet!")


# small_talk_model = load_model("models/final/small_talk_model.h5")
# genre_recognition_model = load_model("models/final/genre_recognition_model.h5")
# sentiment_recognition_model = load_model("models/final/sentiment_recognition_model.h5")
# with open("views/genre/intents.json", "r") as f:
#     intents = json.load(f)
#     genre_list = intents["genres"]


# def userReq():
#     usersaid = commander()
#     if not usersaid:
#         pass
#     elif usersaid:
#         # Make predictions using all three models
#         small_talk_output = small_talk_model.predict(usersaid)
#         genre_recognition_output = genre_recognition_model.predict(usersaid)
#         sentiment_recognition_output = sentiment_recognition_model.predict(usersaid)
#         # Process combined outputs
#         small_talk_prediction = small_talk_output.argmax(axis=-1)
#         genre_recognition_prediction = genre_recognition_output.argmax(axis=-1)
#         sentiment_recognition_prediction = sentiment_recognition_output.mean(axis=-1)

#         if genre_recognition_prediction == get_desired_genre(
#             genre_recognition_prediction
#         ):
#             print(genre_recognition_prediction)
#             milesVoice(genre_recognition_prediction)
#         elif sentiment_recognition_prediction >= 0.5:
#             print(sentiment_recognition_prediction)
#             milesVoice(sentiment_recognition_prediction)
#         else:
#             print(small_talk_prediction)
#             milesVoice(small_talk_prediction)
#     else:
#         milesVoice(f"Sorry, I cannot perform {usersaid} yet")
