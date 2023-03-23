# from colorama import Fore, Style
# from termcolor import cprint


# cprint("ҠΛI: ", "green", "on_grey", attrs=["bold"])
# cprint(": ", "white", "on_grey", attrs=[])


# cprint("ҠΛI: ", "red", "on_grey", attrs=["bold"])
# cprint(": ", "white", "on_grey", attrs=[])


# cprint("ҠΛI: ", "yellow", "on_grey", attrs=["bold"])
# cprint(": ", "white", "on_grey", attrs=[])


# cprint("ҠΛI: ", "blue", "on_grey", attrs=["bold"])
# cprint(": ", "white", "on_grey", attrs=[])

# print(f"{Fore.BLUE}Hello, {Style.RESET_ALL} guys. {Fore.RED} I should be red.")


"""
In this example, the ChatBot class from ChatterBot is used to create a new chat bot called "MyBot". The ChatterBotCorpusTrainer class is used to train the chat bot using the English corpus that comes with ChatterBot.

The while loop continuously prompts the user for input and generates a response from the chat bot using the get_response method. The loop will break when the user enters a KeyboardInterrupt (Ctrl+C).

This is a simple example, but ChatterBot is highly customizable and can be trained on your own data to create a more specific and personalized chat bot.
"""
import random
from chatterbot import ChatBot
from chatterbot.trainers import Trainer
from chatterbot.trainers import ListTrainer
from chatterbot.conversation import Statement
from flask import Flask, render_template, request
from chatterbot.trainers import ChatterBotCorpusTrainer

# Create a new chat bot
bot = ChatBot("MyBot")

# Create a new trainer for the chat bot
trainer = ChatterBotCorpusTrainer(bot)

# Train the chat bot using the english corpus
trainer.train("chatterbot.corpus.english")

# Get a response from the chat bot
while True:
    try:
        user_input = input("You: ")
        bot_response = bot.get_response(user_input)
        print("Bot: " + str(bot_response))

    except KeyboardInterrupt:
        break
# ======================================================================================

"""
This code uses the Flask framework to create a web-based chat bot. The ChatBot class is used to create a new chat bot called "MyBot" and the ChatterBotCorpusTrainer class is used to train the chat bot using the English corpus.

The home() function defines a route for the home page of the web app. The get_bot_response() function defines a route for the chat bot, which takes user input from a GET
"""

# Create a Flask app
app = Flask(__name__)

# Create a new chat bot
bot = ChatBot("MyBot")

# Create a new trainer for the chat bot
trainer = ChatterBotCorpusTrainer(bot)

# Train the chat bot using the english corpus
trainer.train("chatterbot.corpus.english")

# Define a route for the home page


@app.route("/")
def home():
    return render_template("index.html")

# Define a route for the chat bot


@app.route("/get")
def get_bot_response():
    user_input = request.args.get("msg")
    bot_response = bot.get_response(user_input)
    return str(bot_response)


if __name__ == "__main__":
    app.run()

# ======================================================================================

"""
In this example, we create a new ChatBot instance called "MyBot". We then define a list of conversational data to train the bot on. This data can be any type of conversational text, such as chat logs or forum posts.

We create a new ListTrainer instance and pass it the chat bot instance. We then use the train() method of the ListTrainer to train the bot on the conversational data.

Once the training is complete, the bot can be used to generate responses to user input.
"""

# Create a new chat bot
bot = ChatBot("MyBot")

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
In this example, we create a new custom trainer by inheriting from the Trainer class. We then define our own implementation of the train() method to implement our custom training logic.

We also define our own implementation of the get_or_create() method to get or create a statement for the input text. This method is used internally by ChatterBot to add new statements to the bot"s knowledge base.

Once we"ve defined our custom trainer, we can create a new instance of it and pass it to our ChatBot instance using the train() method. This will train the bot using our custom trainer.
"""


class MyTrainer(Trainer):
    def __init__(self, chatbot):
        super().__init__(chatbot)

    def train(self, *args, **kwargs):
        # Implement your own custom training logic here
        pass

    def get_or_create(self, statement_text):
        """
        Get or create a statement for the input text.
        """
        statement, created = self.chatbot.storage.find_or_create(
            Statement(text=statement_text))

        if created:
            self.chatbot.storage.add_to_database(statement)

        return statement


# Create a new chat bot
bot = ChatBot("MyBot")

# Create a new instance of your custom trainer
my_trainer = MyTrainer(bot)

# Train the bot using your custom trainer
my_trainer.train()

"""
In this example, we create a new custom trainer by inheriting from the ListTrainer class. We then define our own implementation of the train() method to implement our custom training logic.

In this case, our custom training logic updates the bot"s knowledge base with new statements and responses from the conversational data. We create a new Statement instance for each statement and response, and use the update() method of the bot"s storage adapter to add them to the knowledge base.

Once we"ve defined our custom trainer, we can create a new instance of it and pass it to our ChatBot instance using the train() method. We also define our own conversational data to train the bot, which is passed to the train() method of our custom trainer.
"""


class MyTrainer(ListTrainer):
    def __init__(self, chatbot):
        super().__init__(chatbot)

    def train(self, conversation):
        for statement_text, response_text in conversation:
            statement = Statement(text=statement_text)
            response = Statement(text=response_text)
            self.chatbot.storage.update(statement)
            self.chatbot.storage.update(response)


# Create a new chat bot
bot = ChatBot("MyBot")

# Create a new instance of your custom trainer
my_trainer = MyTrainer(bot)

# Define your own conversational data to train the bot
conversational_data = [
    ("Hello!", "Hi there!"),
    ("How are you doing?", "I'm doing great, thanks for asking!"),
    ("What's your favorite color?", "My favorite color is blue."),
    ("What do you like to do for fun?",
     "I like to read books and go for walks in nature."),
    ("Do you have any pets?", "No, I don't have any pets. How about you?"),
    ("I have a dog named Charlie.", "That's cool. Dogs are great pets!"),
    ("Goodbye!", "Take care!"),
]

# Train the bot using your custom trainer and conversational data
my_trainer.train(conversational_data)


"""
In this example, we define a new function train_chat_bot() to train the chat bot. We define a list of conversational data and loop over it, updating the chat bot"s knowledge base with each statement and response using the update() method.

We then call train_chat_bot() before starting the chat bot to train it with our custom data.

Note that in this example we"re using the StorageAdapter interface of the ChatBot class to update the knowledge base. This allows us to use any storage backend that implements this interface, such as a SQL database, MongoDB, or Redis.
"""


# Define a function to generate a response to user input


def generate_response(user_input):
    # Define a list of possible responses
    responses = [
        "I'm sorry, I didn't understand your question.",
        "Can you please provide more context?",
        "That's an interesting question.",
        "I'm not sure, let me look it up.",
        "I'm afraid I cannot answer that.",
    ]

    # Select a random response from the list
    response = random.choice(responses)

    return response

# Define a function to train the chat bot


def train_chat_bot():
    # Define a list of conversational data to train the chat bot
    conversational_data = [
        ("Hi", "Hello!"),
        ("How are you?", "I'm doing well, thank you. How about you?"),
        ("What is your name?", "My name is Chat Bot."),
        ("What is the weather like today?", "I'm not sure, let me check."),
        ("What is your favorite color?", "My favorite color is blue."),
        ("Goodbye", "Goodbye!"),
    ]

    # Loop over the conversational data and update the chat bot"s knowledge base
    for statement, response in conversational_data:
        bot.storage.update(bot.storage.create(text=statement),
                           bot.storage.create(text=response))

# Define a function to start the chat bot


def start_chat_bot():
    # Train the chat bot
    train_chat_bot()

    print("Hello! I'm a chat bot. How can I help you?")

    # Loop to continue the conversation until the user exits
    while True:
        # Get user input
        user_input = input("User: ")

        # Generate a response to the user input
        response = generate_response(user_input)

        # Print the response
        print("Bot: " + response)

        # Check if the user wants to exit the chat bot
        if user_input.lower() == "bye":
            print("Goodbye!")
            break


# Create a new chat bot
bot = ChatBot("My Chat Bot")

# Start the chat bot
start_chat_bot()
