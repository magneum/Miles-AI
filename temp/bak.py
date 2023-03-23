import os
import re
import json
import random
import datetime
import requests
from flask import Flask
from colorama import Fore, Style
# os.system("python3 -m app")


# api_key = "860b08c7c72841e6a8a53a3c3cfa8ec6"
# response = requests.get(
#     f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}")
# data = response.json()
# articles = data["articles"]
# for article in articles:
#     print(article["title"])


# api_url = f"http://localhost:3000/news"
# response = requests.get(api_url)
# data = response.json()
# print(f"> {Fore.GREEN}TITLE: {Style.RESET_ALL}" + data["title"])
# print(f"> {Fore.GREEN}AUTHOR: {Style.RESET_ALL}" + data["author"])
# print(f"> {Fore.GREEN}DESCRIPTION: {Style.RESET_ALL}" + data["description"])
# print(f"> {Fore.GREEN}PUBLISHEDAT: {Style.RESET_ALL}" + data["publishedAt"])
# print(f"> {Fore.GREEN}CONTENT: {Style.RESET_ALL}" + data["content"])
# print(f"> {Fore.GREEN}URL: {Style.RESET_ALL}" + data["url"])
# print(f"> {Fore.GREEN}IMAGE: {Style.RESET_ALL}" + data["urlToImage"])


# api_url = f"http://localhost:3000/weather"
# response = requests.get(api_url)
# data = response.json()

import tensorflow as tf
import numpy as np
import random

# Define the training data
training_data = [["hi", "hello"],
                 ["hello", "hi"],
                 ["what's up?", "not much, you?"],
                 ["how are you?", "I'm doing well, thank you."],
                 ["bye", "goodbye"],
                 ]

# Define the vocabulary
vocab = set([word for pair in training_data for word in pair])

# Create a mapping of words to indices
word2idx = {word: idx for idx, word in enumerate(sorted(list(vocab)))}

# Convert the training data into numerical input/output pairs
train_input = np.array(
    [[word2idx[word] for word in pair[0].split()] for pair in training_data])
train_output = np.array(
    [[word2idx[word] for word in pair[1].split()] for pair in training_data])

# Define the model architecture
model = tf.keras.Sequential([tf.keras.layers.Embedding(len(vocab), 32, input_length=train_input.shape[1]),
                             tf.keras.layers.LSTM(32),
                             tf.keras.layers.Dense(
                                 len(vocab), activation="softmax"),
                             ])

# Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_input, train_output, epochs=100)

# Define a function to generate a response given an input string


def generate_response(input_str):
    input_idx = np.array([[word2idx[word] for word in input_str.split()]])
    output_idx = model.predict(input_idx).argmax(axis=1)
    output_str = " ".join([list(word2idx.keys())[list(
        word2idx.values()).index(idx)] for idx in output_idx])
    return output_str


# Start the conversation
print("Hi, I'm a chatbot. Ask me anything!")
while True:
    input_str = input("You: ")
    response_str = generate_response(input_str)
    print("Chatbot:", response_str)
