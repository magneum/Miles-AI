import json
import nltk
import pickle
import random
import numpy as np
from tensorflow import keras
from colorama import Fore, Style
from nltk.stem import WordNetLemmatizer


words_path = "youtube/words.pkl"
model_path = "youtube/youtube.h5"
classes_path = "youtube/classes.pkl"
words = pickle.load(open(words_path, "rb"))
model = keras.models.load_model(model_path)
classes = pickle.load(open(classes_path, "rb"))
intents = json.loads(open("youtube/youtube.json").read())


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [
        WordNetLemmatizer().lemmatize(word.lower()) for word in sentence_words
    ]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {word}")
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [(i, r) for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def searchVideo(song_name):
    print(song_name)
    pass


def get_response(intents_list, intents_json, song_name):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            if "function" in i:
                function_name = i["function"]
                if function_name in globals():
                    globals()[function_name](song_name)
                else:
                    print("Error: Function not found")
                    break
            result = random.choice(i["responses"])
            break
    return result


print(Fore.YELLOW + "Ask me something!" + Style.RESET_ALL)
while True:
    message = input(Fore.BLUE + "USER: " + Style.RESET_ALL)
    if message.lower() == "quit":
        break
    else:
        ints = predict_class(message, model)
        res = get_response(ints, intents, message)
        print(Fore.YELLOW + "CLASS:" + str(ints) + Style.RESET_ALL)
        print(Fore.GREEN + "MILES:" + str(res) + Style.RESET_ALL)
