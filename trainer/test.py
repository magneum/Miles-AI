import pickle
import random
import json
import nltk
import numpy as np
from tensorflow import keras
from colorama import Fore, Style, Back
from sklearn.preprocessing import LabelEncoder
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import colorama
colorama.init()
nltk.download('vader_lexicon')

with open("intents.json") as file:
    data = json.load(file)

# Load pre-trained sentiment analysis model
sid = SentimentIntensityAnalyzer()


def chat():
    # load trained model
    model = keras.models.load_model("chatbot_model")

    # load tokenizer object
    with open("chatbot_model/tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open("chatbot_model/label_encoder.pickle", "rb") as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20

    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        # Sentiment analysis
        score = sid.polarity_scores(inp)
        if score['compound'] >= 0.05:
            sentiment = 'positive'
        elif score['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        print(Fore.LIGHTYELLOW_EX +
              f"Sentiment: {sentiment.capitalize()}" + Style.RESET_ALL)

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                                                          truncating="post", maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data["intents"]:
            if i["tag"] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,
                      np.random.choice(i["responses"]))

        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))


print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()
