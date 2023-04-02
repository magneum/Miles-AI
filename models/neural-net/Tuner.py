import json
import nltk
import pickle
import random
import numpy as np
from tensorflow import keras
from nltk.corpus import wordnet
from colorama import Fore, Style
from keras_tuner import HyperModel
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer as lemmatizer

# =================================================================================== ||
intents = json.loads(open("corpdata/intents.json").read())
words = pickle.load(open("models/neural-net/model/words.pkl", "rb"))
classes = pickle.load(open("models/neural-net/model/classes.pkl", "rb"))

documents = []
ignore_letters = ["?", ".", "!", ","]
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.tokenize.word_tokenize(pattern)
        documents.append((word_list, intent["tag"]))

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
# =================================================================================== ||
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open("models/neural-net/model/words.pkl", "wb"))
pickle.dump(classes, open("models/neural-net/model/classes.pkl", "wb"))

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
training = np.array(training, dtype=object)
training_x = list(training[:, 0])
training_y = list(training[:, 1])

training_x = np.array(training_x)
training_y = np.array(training_y)
x_train, x_val, y_train, y_val = train_test_split(training_x, training_y, test_size=0.2)

input_shape = (len(x_train[0]),)
num_classes = len(classes)
# =================================================================================== ||
class MyHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = keras.Sequential()

        for i in range(hp.Int("num_layers", 1, 4)):
            model.add(
                keras.layers.Dense(
                    units=hp.Int(
                        f"dense_{i+2}_units", min_value=128, max_value=512, step=32
                    ),
                    activation="relu",
                )
            )
            model.add(
                keras.layers.Dropout(
                    rate=hp.Float(
                        f"dropout_{i+1}", min_value=0.0, max_value=0.5, step=0.1
                    )
                )
            )

        model.add(keras.layers.Dense(self.num_classes, activation="softmax"))
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


# =================================================================================== ||
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
    x=x_train,
    y=y_train,
    epochs=50,
    batch_size=8,
    validation_data=(x_val, y_val),
    verbose=1,
)
# =================================================================================== ||
best_model = tuner.get_best_models(num_models=1)[0]
_, val_acc = best_model.evaluate(x_val, y_val)
print(f"\nBest validation accuracy: {val_acc*100:.2f}%")
best_model.save("models/neural-net/model/chatbot_model.h5")
pickle.dump(words, open("models/neural-net/model/words.pkl", "wb"))
pickle.dump(classes, open("models/neural-net/model/classes.pkl", "wb"))
print(Fore.GREEN + "Model saved successfully!" + Style.RESET_ALL)
