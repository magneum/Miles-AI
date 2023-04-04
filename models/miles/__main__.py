# JSON: This module provides functions for encoding and decoding JSON data.
# NLTK: Natural Language Toolkit (nltk) is a library for natural language processing tasks, such as tokenization, stemming, tagging, and parsing.
# PICKLE: This module implements binary protocols for serializing and de-serializing a Python object structure.
# RANDOM: This module implements pseudo-random number generators for various purposes.
# TENSORFLOW: TensorFlow is an open-source machine learning framework for building, training, and deploying machine learning models.
# NUMPY: NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
# KERAS: Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.
# WORDNET: WordNet is a lexical database for the English language, which is used extensively in natural language processing and computational linguistics.
# COLORAMA: This module provides a simple interface to colorize the output of terminal commands.
# HYPERMODEL: This is an abstract class that defines the interface for creating a hypermodel in Keras Tuner.
# WORDNETLEMMATIZER: This class provides a method to lemmatize words using WordNet's built-in morphy function.
# RANDOMSEARCH: This class performs random search over a hypermodel space in Keras Tuner.
# TRAIN_TEST_SPLIT: This function splits arrays or matrices into random train and test subsets.

import json
import nltk
import pickle
import random
import numpy as np
from nltk.corpus import wordnet
from colorama import Fore, Style
from nltk.stem import WordNetLemmatizer
from keras_tuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split

import tensorflow
from tensorflow import keras
from keras_tuner import HyperModel


class hyperModel(HyperModel):
    def __init__(
        self,
        input_shape,
        num_classes,
        use_early_stopping=False,
        embeddings_index=None,
        words=None,
        hp=None,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_early_stopping = use_early_stopping
        self.embeddings_index = embeddings_index
        self.words = words
        self.hp = hp

    def build(self, hp):
        model = keras.Sequential()

        # Embedding layer
        embedding_dim = 300
        embedding_matrix = np.zeros((len(self.words), embedding_dim))
        for i, word in enumerate(self.words):
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        embedding_layer = keras.layers.Embedding(
            len(self.words),
            embedding_dim,
            weights=[embedding_matrix],
            input_length=self.input_shape[0],
            trainable=False,
        )

        # LSTM layer
        lstm_units = hp.Int("lstm_units", min_value=32, max_value=512, step=32)
        model.add(embedding_layer)
        model.add(
            keras.layers.LSTM(units=lstm_units, dropout=0.2, recurrent_dropout=0.2)
        )

        # Dense layers
        num_layers = hp.Int("num_layers", 1, 8)
        for i in range(num_layers):
            units = hp.Int(f"dense_{i+1}_units", min_value=128, max_value=512, step=32)
            activation = hp.Choice(
                f"activation_{i+1}", values=["relu", "sigmoid", "tanh"]
            )
            model.add(keras.layers.Dense(units=units, activation=activation))

            dropout_rate = hp.Float(
                f"dropout_{i+1}", min_value=0.0, max_value=0.5, step=0.1
            )
            model.add(keras.layers.Dropout(rate=dropout_rate))

        optimizer = hp.Choice("optimizer", values=["adam", "rmsprop", "sgd"])
        hp_learning_rate = hp.Float(
            "learning_rate", min_value=1e-5, max_value=1e-2, sampling="LOG"
        )
        if optimizer == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)
        elif optimizer == "rmsprop":
            optimizer = keras.optimizers.RMSprop(learning_rate=hp_learning_rate)
        else:
            optimizer = keras.optimizers.SGD(learning_rate=hp_learning_rate)

        if hp.Boolean("use_L1_regularization", default=True):
            L1_rate = hp.Float(
                "L1_rate", min_value=1e-5, max_value=1e-2, sampling="LOG", default=1e-5
            )
            model.add(
                keras.layers.Dense(
                    units=units,
                    activation=activation,
                    kernel_regularizer=keras.regularizers.L1(L1_rate),
                )
            )

        if hp.Boolean("use_L2_regularization", default=True):
            L2_rate = hp.Float(
                "L2_rate", min_value=1e-5, max_value=1e-2, sampling="LOG"
            )
            model.add(
                keras.layers.Dense(
                    units=units,
                    activation=activation,
                    kernel_regularizer=keras.regularizers.L2(L2_rate),
                )
            )

        model.add(keras.layers.Dense(self.num_classes, activation="softmax"))

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def get_callbacks(self):
        # initialize an empty list of callbacks
        callbacks = []
        # check if the hyperparameters dictionary is not empty
        if self.hp:
            # sample a value for early stopping patience from the search space
            early_stopping_patience = self.hp.Int(
                "early_stopping_patience", min_value=1, max_value=10, default=5
            )
            # create an instance of the EarlyStopping callback and append it to the list of callbacks
            callbacks.append(
                tensorflow.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=early_stopping_patience
                )
            )
        # return the list of callbacks
        return callbacks


# Initialize empty lists and variables
words = []
classes = []
documents = []
ignore_letters = ["?", ".", "!", ","]
glove_file = "corpdata/glove/glove.6B.300d.txt"
words_path = "models/miles/hyperModelwords.pkl"
classes_path = "models/miles/hyperModelclasses.pkl"
model_path = "models/miles/hyperModelchatbot_model.h5"
intents = json.loads(open("database/intents/index.json").read())

# Looping through each intent in the intents dictionary
for intent in intents["intents"]:
    # Looping through each pattern in the current intent
    for pattern in intent["patterns"]:
        # Tokenizing the pattern into a list of words using NLTK
        word_list = nltk.tokenize.word_tokenize(pattern)
        # Extending the words list with the new word list
        words.extend(word_list)
        # Appending a tuple of the word list and the intent tag to the documents list
        documents.append((word_list, intent["tag"]))
        # Adding the intent tag to the classes list if it's not already there
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Printing a message indicating that intent processing is complete
print(Fore.GREEN + "Done processing intents." + Style.RESET_ALL)


# Performing part-of-speech tagging on the words list using NLTK
pos_tags = nltk.pos_tag(words)
# Initializing a WordNetLemmatizer object to perform lemmatization
lemmatizer = WordNetLemmatizer()
# Printing a message indicating that lemmatization is starting
print(Fore.BLUE + "Lemmatizing words..." + Style.RESET_ALL)

# Using a list comprehension to lemmatize each word in the words list
# The lemmatization process uses the part-of-speech tags to determine the correct lemma
# If the part-of-speech tag starts with "j", the word is lemmatized as an adjective
# If the part-of-speech tag starts with "v", the word is lemmatized as a verb
# If the part-of-speech tag starts with "n", the word is lemmatized as a noun
# If the part-of-speech tag starts with "r", the word is lemmatized as an adverb
# If the part-of-speech tag does not fall into any of the above categories, the word is simply lemmatized
words = [
    lemmatizer.lemmatize(word, pos=wordnet.ADJ)
    if tag[1][0].lower() == "j"
    else lemmatizer.lemmatize(word, pos=wordnet.VERB)
    if tag[1][0].lower() == "v"
    else lemmatizer.lemmatize(word, pos=wordnet.NOUN)
    if tag[1][0].lower() == "n"
    else lemmatizer.lemmatize(word, pos=wordnet.ADV)
    if tag[1][0].lower() == "r"
    else lemmatizer.lemmatize(word)
    for word, tag in zip(words, pos_tags)
]

# Printing a message indicating that lemmatization is complete
print(Fore.GREEN + "Done lemmatizing words." + Style.RESET_ALL)
# Printing a message indicating that sorting and duplicate removal is starting
print(Fore.BLUE + "Sorting and removing duplicates..." + Style.RESET_ALL)
# Sorting the words list and removing duplicates using the built-in sorted() and set() functions
words = sorted(set(words))
classes = sorted(set(classes))
# Saving the words and classes lists to disk using the pickle module
pickle.dump(words, open(words_path, "wb"))
pickle.dump(classes, open(classes_path, "wb"))


# Initializing empty lists for the training data and output
training = []
output_empty = [0] * len(classes)
# Looping through each document in the documents list
for document in documents:
    bag = []
    word_patterns = document[0]
    # Lemmatizing and converting each word to lowercase, and removing ignored letters
    word_patterns = [
        lemmatizer.lemmatize(word.lower())
        for word in word_patterns
        if word not in ignore_letters
    ]
    # Creating a bag of words by checking if each word in the words list is present in the document
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    # Creating the output row by setting the index of the correct class to 1 and the rest to 0
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    # Appending the bag of words and output row to the training list
    training.append([bag, output_row])

# Shuffling the training data and converting it to a NumPy array
random.shuffle(training)
training = np.array(training, dtype=object)
# Splitting the training and validation data
train_x = list(training[:, 0])
train_y = list(training[:, 1])
train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42
)
# Setting the input shape and number of output classes
input_shape = (len(train_x[0]),)
num_classes = len(classes)


# Function to load GloVe embeddings from a file
def load_glove_embeddings(file):
    embeddings_index = {}
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    return embeddings_index


# Loading GloVe embeddings from a file
print(Fore.GREEN + "Loading GloVe embeddings..." + Style.RESET_ALL)
with open(glove_file, encoding="utf-8") as f:
    embeddings_index = load_glove_embeddings(f)
print(Fore.GREEN + "GloVe embeddings loaded." + Style.RESET_ALL)

# Sorting the list of words in the embeddings index
words = sorted(list(embeddings_index.keys()))

# Creating an instance of the hyperModel class
my_hyper_model = hyperModel(
    input_shape,
    num_classes,
    use_early_stopping=True,
    embeddings_index=embeddings_index,
    words=words,
)

# Creating an instance of the RandomSearch tuner to search for the best hyperparameters
print(Fore.GREEN + "Creating tuner..." + Style.RESET_ALL)
tuner = RandomSearch(
    my_hyper_model,
    objective="val_accuracy",
    max_trials=20,
    executions_per_trial=4,
    directory="models/miles/hyperModel",
    project_name="hyperModel",
)
print(Fore.GREEN + "Tuner created." + Style.RESET_ALL)

# Getting the callbacks for the tuner
callbacks = my_hyper_model.get_callbacks()
print(Fore.GREEN + "Callbacks created." + Style.RESET_ALL)

# Start tuning the hyperparameters using the tuner
tuner.search(
    x=train_x,
    y=train_y,
    epochs=10,
    batch_size=8,
    validation_data=(val_x, val_y),
    verbose=1,
    callbacks=callbacks,
)
print(Fore.GREEN + "Tuning completed." + Style.RESET_ALL)

# Get the best model and evaluate its performance on the validation set
best_model = tuner.get_best_models(num_models=1)[0]
_, val_acc = best_model.evaluate(val_x, val_y)
print(f"\nBest validation accuracy: {val_acc*100:.2f}%")

# Save the best model, words, and classes to disk
best_model.save(model_path)
pickle.dump(words, open(words_path, "wb"))
pickle.dump(classes, open(classes_path, "wb"))
print(Fore.GREEN + "Model saved successfully!" + Style.RESET_ALL)
