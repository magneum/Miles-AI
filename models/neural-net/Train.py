import json
import nltk
import pickle
import random
import logging
import numpy as np
import tensorflow as tf
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
from colorama import Fore, Style
from keras.models import Sequential
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer as lemmatizer
from keras.layers import Dense, BatchNormalization, Activation, Dropout

# ======================================================================================================================
# nltk.download("wordnet")
# print(f"{Fore.CYAN}Devices for ML: {device_lib.list_local_devices()}{Style.RESET_ALL}")

# Load the intents from the intents.json file and parse them as a dictionary
intents = json.loads(open("corpdata/intents.json").read())

# Initialize empty lists for words, classes, and documents
words = []
classes = []
documents = []


learning_rate = 0.001  # Define the learning rate schedule for the Adam optimizer
num_epochs = 10000  # Set the number of epochs to train the model for
batch_size = 32  # Set the batch size to use for training
verbose = 1  # Set the level of verbosity for training progress output

# Define a list of characters to ignore in the patterns
ignore_letters = ["?", ".", "!", ","]


# ======================================================================================================================

# Loop through each intent in the intents dictionary
print(f"{Fore.BLUE}Processing intents...{Style.RESET_ALL}")
for intent in intents["intents"]:
    # Loop through each pattern in the intent
    for pattern in intent["patterns"]:
        # Tokenize the pattern into a list of words and add it to the words list
        word_list = nltk.tokenize.word_tokenize(pattern)
        words.extend(word_list)
        # Add the word list and the intent tag as a tuple to the documents list
        documents.append((word_list, intent["tag"]))
        # If the intent tag is not already in the classes list, add it
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
print(f"{Fore.GREEN}Done processing intents.{Style.RESET_ALL}")


pos_tags = nltk.pos_tag(words)  # Use NLTK to tag parts of speech in the words list
lemmatizer = lemmatizer()  # Create a new instance of the lemmatizer

# Iterate over each word and its POS tag, lemmatizing the word according to its POS tag
print(f"{Fore.BLUE}Lemmatizing words...{Style.RESET_ALL}")
words = [
    lemmatizer.lemmatize(word, pos=wordnet.ADJ)  # lemmatize adjectives
    if tag[1][0].lower() == "j"
    else lemmatizer.lemmatize(word, pos=wordnet.VERB)  # lemmatize verbs
    if tag[1][0].lower() == "v"
    else lemmatizer.lemmatize(word, pos=wordnet.NOUN)  # lemmatize nouns
    if tag[1][0].lower() == "n"
    else lemmatizer.lemmatize(word, pos=wordnet.ADV)  # lemmatize adverbs
    if tag[1][0].lower() == "r"
    else lemmatizer.lemmatize(
        word
    )  # if the POS tag is not one of the above, lemmatize the word as-is
    for word, tag in zip(
        words, pos_tags
    )  # iterate over each word and its corresponding POS tag
]
print(f"{Fore.GREEN}Done lemmatizing words.{Style.RESET_ALL}")

# Sort and remove duplicates from the list of words and classes
print(f"{Fore.BLUE}Sorting and removing duplicates...{Style.RESET_ALL}")
words = sorted(set(words))
classes = sorted(set(classes))

# Save the processed words and classes lists to pickle files for later use
pickle.dump(words, open("models/neural-net/model/words.pkl", "wb"))
pickle.dump(classes, open("models/neural-net/model/classes.pkl", "wb"))

# ======================================================================================================================
training = []  # Initialize empty list to store training data
output_empty = [0] * len(
    classes
)  # Create a list of 0s with length equal to the number of classes

# Loop through each document in the dataset
for document in documents:
    # Initialize an empty bag of words for the document
    bag = []
    # Get the list of words from the document
    word_pattern = document[0]
    # Lemmatize each word and convert to lowercase
    word_pattern = [lemmatizer.lemmatize(word.lower()) for word in word_pattern]
    # Loop through each word in the vocabulary
    for word in words:
        # If the word is in the document, set the bag of words value to 1, otherwise 0
        bag.append(1) if word in word_pattern else bag.append(0)
    # Initialize an empty output row
    output_row = list(output_empty)
    # Set the index of the output row for the class to 1
    output_row[classes.index(document[1])] = 1
    # Add the bag of words and output row to the training list
    training.append([bag, output_row])


random.shuffle(training)  # Shuffle the training data randomly
training = np.array(
    training, dtype=object
)  # Convert the training data to a numpy array
# Get the bag of words and output data separately
training_x = list(training[:, 0])
training_y = list(training[:, 1])

# ======================================================================================================================
# Set up neural network architecture and hyperparameters
model = Sequential()
units_per_layer = [
    2048,
    1024,
    512,
    256,
    128,
    64,
]

activation_functions = [
    "relu",
    "relu",
    "relu",
    "sigmoid",
    "sigmoid",
    "sigmoid",
]

# units_per_layer = [2048,1024,512,256,128,64,32,16,8,4,]
# activation_functions = ["relu","sigmoid","softmax","elu","selu","softplus","softsign","tanh","hard_sigmoid","exponential",]


# Calculate the number of hidden layers based on the number of units and activation functions provided
num_layers = len(units_per_layer)

if num_layers != len(activation_functions):
    raise ValueError("Number of units and activation functions must be equal")

# Add dense layers with corresponding activation functions
for i in range(num_layers):
    # Add a dense layer with specified number of units and input shape
    model.add(Dense(units=units_per_layer[i], input_shape=(len(training_x[0]),)))
    # Add batch normalization to normalize the inputs of the previous layer
    model.add(BatchNormalization())
    # Add the activation function specified for the layer
    activation_function = activation_functions[i % len(activation_functions)]
    if activation_function == "relu":
        model.add(Activation("relu"))
    elif activation_function == "sigmoid":
        model.add(Activation("sigmoid"))
    elif activation_function == "softmax":
        model.add(Activation("softmax"))
    elif activation_function == "elu":
        model.add(Activation("elu"))
    elif activation_function == "selu":
        model.add(Activation("selu"))
    elif activation_function == "softplus":
        model.add(Activation("softplus"))
    elif activation_function == "softsign":
        model.add(Activation("softsign"))
    elif activation_function == "tanh":
        model.add(Activation("tanh"))
    elif activation_function == "hard_sigmoid":
        model.add(Activation("hard_sigmoid"))
    elif activation_function == "exponential":
        model.add(Activation("exponential"))
    else:
        raise ValueError(f"Invalid activation function: {activation_function}")
    # Add random dropout for regularization
    if i < num_layers - 1:
        dropout_rate = np.random.uniform(0.4, 0.6)  # set range of dropout rate
        model.add(Dropout(dropout_rate))


# Add the final output layer with softmax activation
model.add(Dense(len(training_y[0]), activation="softmax"))

# ======================================================================================================================


def lr_schedule(epoch):
    learning_rates = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    lr = learning_rates[min(epoch // 10, len(learning_rates) - 1)]
    print(
        f"{Fore.GREEN}{Style.BRIGHT}Epoch {epoch + 1}:{Style.RESET_ALL} {Fore.BLUE}{Style.BRIGHT}Learning Rate:{Style.RESET_ALL} {Fore.BLUE}{lr}{Style.RESET_ALL}"
    )
    return lr


# Define the Adam optimizer with a specific learning rate
adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compile the model with the categorical crossentropy loss function, Adam optimizer, and accuracy metric
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

# Define the learning rate scheduler callback
lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)


# Split the training data into training and validation sets
training_x, validation_x, training_y, validation_y = train_test_split(
    np.array(training_x), np.array(training_y), test_size=0.2, random_state=42
)


# Save the training history (loss and accuracy values) in a variable
training_history = model.fit(
    training_x,
    training_y,
    epochs=num_epochs,
    batch_size=batch_size,
    verbose=verbose,
    validation_data=(validation_x, validation_y),
    callbacks=[lr_scheduler_callback],
)

# ======================================================================================================================
# This should give you a plot of the training and validation accuracy over each epoch.
def plot_accuracy(training_history):
    fig, ax = plt.subplots(figsize=(10, 8))
    epochs = len(training_history.history["accuracy"])
    x = range(1, epochs + 1)
    y1 = training_history.history["accuracy"]
    y2 = training_history.history["val_accuracy"]
    cmap = plt.get_cmap("plasma")
    colors = [cmap(i) for i in np.linspace(0, 1, epochs)]
    (line1,) = ax.plot(x, y1, label="Training Accuracy", color=colors[0], linewidth=3)
    (line2,) = ax.plot(
        x, y2, label="Validation Accuracy", color=colors[-1], linewidth=3
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Model Accuracy", fontsize=20)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["left"].set_position(("outward", 10))
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.plot([0, epochs], [0.9, 0.9], "--", color="#4d4d4d", linewidth=2)
    ax.plot([0, epochs], [0.8, 0.8], "--", color="#4d4d4d", linewidth=2)
    ax.annotate(
        "Accuracy Threshold 0.9",
        xy=(epochs, 0.9),
        xytext=(epochs, 0.92),
        ha="right",
        va="bottom",
        fontsize=12,
        color="#4d4d4d",
    )
    ax.annotate(
        "Accuracy Threshold 0.8",
        xy=(epochs, 0.8),
        xytext=(epochs, 0.82),
        ha="right",
        va="bottom",
        fontsize=12,
        color="#4d4d4d",
    )
    plt.tight_layout()
    plt.show()


# ======================================================================================================================
# saves the trained model.
model.save("models/neural-net/model")
# plot the model accuracy
plot_accuracy(training_history)
