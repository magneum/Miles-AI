import json
import nltk
import pickle
import random
import numpy as np
import tensorflow as tf
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from nltk.stem import WordNetLemmatizer as lemmatizer
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization

# nltk.download("wordnet")

intents = json.loads(open("corpdata/intents.json").read())
words = []
classes = []
documents = []
ignore_letters = ["?", ".", "!", ","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.tokenize.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

pos_tags = nltk.pos_tag(words)
lemmatizer = lemmatizer()
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
words = sorted(set(words))
classes = sorted(set(classes))
pickle.dump(words, open("trainer/cornel/words.pkl", "wb"))
pickle.dump(classes, open("trainer/cornel/classes.pkl", "wb"))

training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_pattern = document[0]
    word_pattern = [lemmatizer.lemmatize(word.lower()) for word in word_pattern]
    for word in words:
        bag.append(1) if word in word_pattern else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
training_x = list(training[:, 0])
training_y = list(training[:, 1])


def create_model(
    input_shape,
    output_shape,
    num_layers=6,
    units_per_layer=[1024, 512, 256, 128, 64, 32],
    activation_function="relu",
    dropout_rate=0.5,
    use_batch_norm=True,
):
    model = Sequential()
    for i in range(num_layers):
        model.add(Dense(units_per_layer[i], input_shape=(input_shape,)))
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(Activation(activation_function))
        if i < num_layers - 1:
            model.add(Dropout(dropout_rate))
    model.add(Dense(output_shape, activation="softmax"))
    return model


model = create_model(
    input_shape=len(training_x[0]),
    output_shape=len(training_y[0]),
    num_layers=6,
    units_per_layer=[1024, 512, 256, 128, 64, 32],
    activation_function="relu",
    dropout_rate=0.5,
    use_batch_norm=True,
)

adam_learning_rate = 0.001
epochs = 500
batch_size = 32
verbose = 1

adam = tf.keras.optimizers.Adam(learning_rate=adam_learning_rate)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

training_history = model.fit(
    np.array(training_x),
    np.array(training_y),
    epochs=epochs,
    batch_size=batch_size,
    verbose=verbose,
)

model.save("trainer/cornel/cornel.h5")
plt.plot(range(1, epochs + 1), training_history.history["accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
