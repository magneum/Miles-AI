import json
import nltk
import pickle
import random
import numpy as np
from tensorflow import keras
from colorama import Fore, Style
from nltk.stem import WordNetLemmatizer


words_path = "app/miles/words.pkl"
classes_path = "app/miles/classes.pkl"
model_path = "app/miles/chatbot_model.h5"
intents = json.loads(open("app/miles/intents.json").read())
words = pickle.load(open(words_path, "rb"))
classes = pickle.load(open(classes_path, "rb"))
model = keras.models.load_model(model_path)


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


def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
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
        res = get_response(ints, intents)
        print(Fore.YELLOW + "CLASS:" + str(ints) + Style.RESET_ALL)
        print(Fore.GREEN + "MILES:" + str(res) + Style.RESET_ALL)
