import random
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
# Sure, here's an example of how to add custom training logic to your chat bot:


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

# Define a function to start the chat bot


def start_chat_bot():
    # Create a new chat bot
    bot = ChatBot("My Chat Bot")

    # Train the chat bot on a corpus of conversational data
    trainer = ChatterBotCorpusTrainer(bot)
    trainer.train("chatterbot.corpus.english")

    print("Hello! I'm a chat bot. How can I help you?")

    # Loop to continue the conversation until the user exits
    while True:
        # Get user input
        user_input = input("User: ")

        # Generate a response to the user input
        response = bot.get_response(user_input)

        # Print the response
        print("Bot: " + str(response))

        # Check if the user wants to exit the chat bot
        if user_input.lower() == 'bye':
            print("Goodbye!")
            break


# Start the chat bot
start_chat_bot()
