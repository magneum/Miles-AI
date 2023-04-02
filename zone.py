import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner import RandomSearch
from spellchecker import SpellChecker

# Initialize the spell checker
spell = SpellChecker()

train_data = pd.read_csv("corpdata/gpt/large-762M-k40.train.csv")
valid_data = pd.read_csv("corpdata/gpt/large-762M-k40.valid.csv")
test_data = pd.read_csv("corpdata/gpt/large-762M-k40.test.csv")

train_inputs = train_data["input"].values
train_outputs = train_data["output"].values

valid_inputs = valid_data["input"].values
valid_outputs = valid_data["output"].values

test_inputs = test_data["input"].values
test_outputs = test_data["output"].values


def build_model(hp):
    model = keras.Sequential()
    model.add(
        layers.Dense(
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation=hp.Choice("activation", values=["relu", "sigmoid"]),
            input_shape=[len(train_inputs[0])],
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

print("Test Mean Absolute Error:", test_mae)


def generate_response(input_str):
    # Spell check the user input
    words = input_str.split()
    misspelled_words = spell.unknown(words)
    for word in misspelled_words:
        corrected_word = spell.correction(word)
        input_str = input_str.replace(word, corrected_word)

    # Generate response using the best model
    input_data = pd.Series([input_str])
    prediction = best_model.predict(input_data)[0][0]
    return prediction


print("Hello! I am a chatbot. How can I help you today?")
while True:
    user_input = input("> ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    # Spell check the user input
    words = user_input.split()
    misspelled_words = spell.unknown(words)
    for word in misspelled_words:
        corrected_word = spell.correction(word)
        user_input = user_input.replace(word, corrected_word)

    response = generate_response(user_input)
    print(response)
