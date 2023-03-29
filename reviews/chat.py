import json
import torch
import random
import nltk
from model import NeuralNet
from utils import bag_of_words, tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load intents from a JSON file
try:
    with open("corpdata/intents.json", "r") as f:
        intents = json.load(f)
except FileNotFoundError:
    print("intents.json file not found")
    exit()

# Load model data from a file
try:
    data = torch.load("model.pth")
except FileNotFoundError:
    print("model.pth file not found")
    exit()

# Extract model data and set default values
all_words = data.get("all_words", [])
tags = data.get("tags", [])
input_size = data.get("input_size", len(all_words))
hidden_size = data.get("hidden_size", 8)
output_size = data.get("output_size", len(tags))
model_state = data.get("model_state")

# Initialize the neural network model
model = NeuralNet(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
).to(device)

# Load the model state if available and set the model to evaluation mode
if model_state:
    model.load_state_dict(model_state)
    model.eval()

# Set up the chatbot name and greeting message
bot_name = "Miles"
greeting = "Hi there! How can I assist you today?"

# Set up a list of fallback responses if the model cannot understand the user's input
fallback_responses = [
    "I'm sorry, could you please rephrase that?",
    "I'm not sure I understand. Can you provide more detail?",
    "I didn't catch that. Could you please repeat?",
]

# Set up a dictionary of goodbye messages and their corresponding tags
goodbye_tags = {
    "goodbye": [
        "Goodbye!",
        "Bye!",
        "See you later!",
        "Have a nice day!",
    ],
    "thanks": [
        "You're welcome!",
        "No problem!",
        "My pleasure!",
    ],
}

# Main loop for chatting with the user
while True:
    try:
        # Get input from the user and process it
        sentence = input("You: ").strip()
        if not sentence:
            continue

        # Check if the user wants to quit the chatbot
        if sentence.lower() == "quit":
            print(f"{bot_name}: {random.choice(goodbye_tags['goodbye'])}")
            break

        # Tokenize the input sentence and perform part-of-speech tagging
        sentence_tokens = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(sentence_tokens)

        # Extract only the part-of-speech tags
        pos_tags = [tag[1] for tag in pos_tags]

        # Convert the input sentence to a bag of words
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        # Feed the bag of words to the model and get a prediction
        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        # Check if the model is confident enough in its prediction
        if prob.item() > 0.75:
            for intent in intents.get("intents", []):
                if tag == intent.get("tag"):
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
                    break
        else:
            print(f"{bot_name}: I don't understand.")
    except KeyboardInterrupt:
        print("Exiting...")
        break
    except KeyError:
        print("Invalid model or data file")
        exit()
    finally:
        torch.cuda.empty_cache()
