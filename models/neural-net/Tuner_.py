# # =================================================================================== ||
import json
import nltk
import pickle
import random
import numpy as np
from tensorflow import keras
from nltk.corpus import wordnet
from colorama import Fore, Style
from kerastuner import HyperModel
from nltk.stem import WordNetLemmatizer
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split

intents = json.loads(open("corpdata/intents.json").read())
with open("models/neural-net/model/words.pkl", "rb") as f:
    words = pickle.load(f)
with open("models/neural-net/model/classes.pkl", "rb") as f:
    classes = pickle.load(f)

documents = []
ignore_letters = ["?", ".", "!", ","]
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.tokenize.word_tokenize(pattern)
        documents.append((word_list, intent["tag"]))

lemmatizer = WordNetLemmatizer()
pos_tags = nltk.pos_tag(words)
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

with open("models/neural-net/model/words.pkl", "wb") as f:
    pickle.dump(words, f)
with open("models/neural-net/model/classes.pkl", "wb") as f:
    pickle.dump(classes, f)


training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [
        lemmatizer.lemmatize(word.lower())
        for word in word_patterns
        if word not in ignore_letters
    ]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
random.shuffle(training)
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42
)
input_shape = (len(train_x[0]),)
num_classes = len(classes)


class MyHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = keras.Sequential()

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

        if hp.Boolean("use_L1_regularization", default=False):
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

        if hp.Boolean("use_L2_regularization", default=False):
            L2_rate = hp.Float(
                "L2_rate", min_value=1e-5, max_value=1e-2, sampling="LOG"
            )
            model.add(keras.regularizers.L2(L2_rate))

        if hp.Boolean("use_early_stopping", default=False):
            early_stopping_patience = hp.Int(
                "early_stopping_patience", min_value=1, max_value=10
            )
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=early_stopping_patience
                )
            ]
        else:
            callbacks = []

        model.add(keras.layers.Dense(self.num_classes, activation="softmax"))
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model


my_hyper_model = MyHyperModel(input_shape, num_classes)
tuner = RandomSearch(
    my_hyper_model,
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=3,
    directory="my_dir",
    project_name="helloworld",
)


tuner.search(
    x=train_x,
    y=train_y,
    epochs=50,
    batch_size=8,
    validation_data=(val_x, val_y),
    verbose=1,
)


best_model = tuner.get_best_models(num_models=1)[0]
_, val_acc = best_model.evaluate(val_x, val_y)
print(f"\nBest validation accuracy: {val_acc*100:.2f}%")
best_model.save("models/neural-net/model/chatbot_model.h5")
pickle.dump(words, open("models/neural-net/model/words.pkl", "wb"))
pickle.dump(classes, open("models/neural-net/model/classes.pkl", "wb"))
print(Fore.GREEN + "Model saved successfully!" + Style.RESET_ALL)
# =================================================================================== ||
