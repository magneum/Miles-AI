import nltk
import json
import pickle
import numpy as np
from nltk.corpus import wordnet
from colorama import Fore, Style
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

model = load_model("views/genre/genre_model.h5")
words = pickle.load(open("views/genre/words.pkl", "rb"))
intents = json.loads(open("views/genre/index.json").read())
classes = pickle.load(open("views/genre/classes.pkl", "rb"))


def preprocess_sentiment(sentiment):
    lemmatizer = WordNetLemmatizer()
    words = nltk.tokenize.word_tokenize(sentiment)
    pos_tags = nltk.pos_tag(words)
    words = [
        lemmatizer.lemmatize(word, pos=wordnet.ADJ)
        if tag[1][0].lower() == "j"
        else lemmatizer.lemmatize(word, pos=wordnet.VERB)
        if tag[1][0].lower() == "v"
        else lemmatizer.lemmatize(word, pos=wordnet.NOUN)
        if tag[1][0].lower() == "n"
        else lemmatizer.lemmatize(word, pos=wordnet.ADV)
        if tag[1][0].lower() == "r"
        else lemmatizer.lemmatize(word)
        for word, tag in zip(words, pos_tags)
    ]
    return words


def predict_mood(sentiment):
    sentiment = preprocess_sentiment(sentiment)
    bag_of_words = [0] * len(words)
    for word in sentiment:
        if word in words:
            bag_of_words[words.index(word)] = 1
    bag_of_words = [bag_of_words]
    bag_of_words = np.array(bag_of_words)
    prediction = model.predict(bag_of_words)
    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]
    return predicted_class


def get_genre(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = i["genre"]
            break
    return result


# while True:
#     userInput = input(Fore.BLUE + "Enter or ('q' to quit): " + Style.RESET_ALL)
#     if userInput.lower() == "q":
#         break
#     else:
#         predicted_mood = predict_mood(userInput)
#         predicted_genre = get_genre([{"intent": predicted_mood}], intents)
#         print(Fore.GREEN + f"Predicted mood: {predicted_mood}" + Style.RESET_ALL)
#         print(Fore.GREEN + f"Predicted genre: {predicted_genre}" + Style.RESET_ALL)
