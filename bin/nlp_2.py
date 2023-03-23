import random
from chatterbot import ChatBot
from chatterbot.trainers import Trainer
from chatterbot.trainers import ListTrainer
from chatterbot.conversation import Statement
from flask import Flask, render_template, request
from chatterbot.trainers import ChatterBotCorpusTrainer

# Certainly! Here's an example of how to train a ChatterBot model using a custom dataset.

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

# Create a new chat bot
bot = ChatBot('MyBot')

# Define a list of conversational data to train the bot
conversational_data = [
    "Hello!",
    "Hi there!",
    "How are you doing?",
    "I'm doing great, thanks for asking!",
    "What's your favorite color?",
    "My favorite color is blue.",
    "What do you like to do for fun?",
    "I like to read books and go for walks in nature.",
    "Do you have any pets?",
    "No, I don't have any pets. How about you?",
    "I have a dog named Charlie.",
    "That's cool. Dogs are great pets!",
    "Goodbye!",
    "Take care!",
]

# Create a new trainer for the chat bot using the ListTrainer
trainer = ListTrainer(bot)

# Train the bot using the conversational data
trainer.train(conversational_data)

"""
In this example, we create a new ChatBot instance called "MyBot". We then define a list of conversational data to train the bot on. This data can be any type of conversational text, such as chat logs or forum posts.

We create a new ListTrainer instance and pass it the chat bot instance. We then use the train() method of the ListTrainer to train the bot on the conversational data.

Once the training is complete, the bot can be used to generate responses to user input.
"""
