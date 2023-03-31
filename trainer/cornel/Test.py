import pickle
import nltk
import numpy as np
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer as lemmatizer

model = load_model("trainer/cornel/cornel.h5")

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
lemmatizer = lemmatizer()


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


while True:
    message = input("Enter your message: ")
    if message == "exit":
        break
    results = predict_class(message)
    print(results)


# chmod +x cuda_11.6.0_510.39.01_linux.run
# sudo ./cuda_11.6.0_510.39.01_linux.run
# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# source ~/.bashrc
# nvcc --version
