import json
import nltk
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer as lemmatizer

# load the model
model = load_model("models/model/model")
words = pickle.load(open("models/neural-net/model/words.pkl", "rb"))
classes = pickle.load(open("models/neural-net/model/classes.pkl", "rb"))
lemmatizer = lemmatizer()

with open("corpdata/intents.json") as file:
    intents = json.load(file)


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0][0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = np.random.choice(i["responses"])
            break
    return result


while True:
    message = input("Enter your message: ")
    if message == "exit":
        break
    class_results = predict_class(message)
    print(class_results)
    response = get_response(class_results, intents)
    print(response)
