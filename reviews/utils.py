import os
import json
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

# Create a PorterStemmer object to use for stemming words
stemmer = PorterStemmer()

# Define a function to tokenize a sentence into individual words
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# Define a function to stem a single word using the PorterStemmer algorithm
def stem(word):
    # Convert the word to lowercase and stem it
    return stemmer.stem(word.lower())


# Define a function to create a bag-of-words vector from a tokenized sentence
def bag_of_words(tokenized_sentence, all_words):
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "i", "you", "bye", "thanks", "cool"]
    bag = [  "0",   "1",    "0",  "1",   "0",   "0",      "0"   ]
    """
    # Stem each word in the tokenized sentence using the PorterStemmer
    stemmed_sentence = [stem(w) for w in tokenized_sentence]

    # Create a binary vector with one element for each word in the vocabulary
    bag = np.zeros(len(all_words), dtype=np.float32)

    # For each word in the vocabulary, set the corresponding element in the bag vector to 1 if the word appears in the stemmed sentence
    for index, word in enumerate(all_words):
        if word in stemmed_sentence:
            bag[index] = 1.0

    # Return the bag-of-words vector
    return bag


try:
    with open("corpdata/intents.json", "r") as f:
        intents = json.load(f)
except FileNotFoundError:
    print("intents.json file not found")
    exit()

try:
    patterns = intents.get("patterns", [])
    tags = [pattern[1] for pattern in patterns]
    all_words = [stem(word) for pattern in patterns for word in tokenize(pattern[0])]
except (TypeError, KeyError):
    print("Invalid data format in intents.json file")
    exit()

# Remove duplicates from the list of all words and sort it alphabetically
all_words = sorted(list(set(all_words)))

# Sort the list of all tags alphabetically
tags = sorted(tags)

# Create a list of output vectors for each pattern
output_vectors = []
for pattern in patterns:
    tag = pattern[1]
    try:
        output_vector = [0] * len(tags)
        output_vector[tags.index(tag)] = 1
        output_vectors.append(output_vector)
    except ValueError:
        print(f"Invalid tag '{tag}' in intents.json file")
        exit()
