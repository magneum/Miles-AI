import json
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

with open("youtube/genre.json") as file:
    data = json.load(file)

intents = data["intents"]
patterns = []
genres = []
for intent in intents:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        genres.append(intent["genre"])

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(patterns)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for pattern in patterns:
    token_list = tokenizer.texts_to_sequences([pattern])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[: i + 1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(
    keras.preprocessing.sequence.pad_sequences(
        input_sequences, maxlen=max_sequence_len, padding="pre"
    )
)
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = keras.utils.to_categorical(y, num_classes=total_words)


def create_model(hp):
    model = keras.Sequential()
    model.add(layers.Embedding(total_words, 10, input_length=max_sequence_len - 1))
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                units=hp.Int("units", min_value=32, max_value=512, step=32),
                return_sequences=True,
            )
        )
    )
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(units=hp.Int("units", min_value=32, max_value=512, step=32)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(total_words, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


tuner = RandomSearch(
    create_model,
    max_trials=12,
    objective="val_accuracy",
    directory="my_model_directory",
    project_name="music_genre_model",
)

tuner.search(X, y, epochs=50, validation_split=0.2)
best_model = tuner.get_best_models(num_models=1)[0]
best_model.save("sentiment_analysis_model.h5")
