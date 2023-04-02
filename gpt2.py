import keras
import joblib
import pandas as pd
from keras import layers
from keras_tuner import RandomSearch
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

train_data = pd.read_csv("corpdata/gpt/large-762M-k40.train.csv")
valid_data = pd.read_csv("corpdata/gpt/large-762M-k40.valid.csv")
test_data = pd.read_csv("corpdata/gpt/large-762M-k40.test.csv")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data["text"].values)

train_sequences = tokenizer.texts_to_sequences(train_data["text"].values)
valid_sequences = tokenizer.texts_to_sequences(valid_data["text"].values)
test_sequences = tokenizer.texts_to_sequences(test_data["text"].values)

max_length = 128
train_inputs = pad_sequences(train_sequences, maxlen=max_length, padding="post")
train_outputs = train_data["ended"].values
valid_inputs = pad_sequences(valid_sequences, maxlen=max_length, padding="post")
valid_outputs = valid_data["ended"].values
test_inputs = pad_sequences(test_sequences, maxlen=max_length, padding="post")
test_outputs = test_data["ended"].values


def build_model(hp):
    model = keras.Sequential()
    model.add(
        layers.Dense(
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation=hp.Choice("activation", values=["relu", "sigmoid"]),
            input_shape=[max_length],
        )
    )
    model.add(layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model


tuner = RandomSearch(
    build_model, objective="val_mae", max_trials=5, executions_per_trial=3
)

tuner.search(
    train_inputs, train_outputs, epochs=5, validation_data=(valid_inputs, valid_outputs)
)

best_model = tuner.get_best_models(num_models=1)[0]
test_loss, test_mae = best_model.evaluate(test_inputs, test_outputs)

print("Best Model Summary:")
print(best_model.summary())
print("Best Hyperparameters:")
print(tuner.get_best_hyperparameters(num_trials=1)[0].values)
best_model.save("got2_model.h5")


def generate_response(input_str, model):
    input_data = pd.Series([input_str])
    prediction = model.predict(input_data)[0][0]
    return prediction


got2_model = joblib.load("got2_model.joblib")
print("Hello! I am a chatbot. How can I help you today?")
while True:
    user_input = input("> ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    response = generate_response(user_input, got2_model)
    print("Model response:", response)
