import keras
import pandas as pd
from keras import layers
from keras_tuner import RandomSearch

train_data = pd.read_csv("corpdata/gpt/large-762M-k40.train.csv")
print("Train data shape:", train_data.shape)
print("Train data head:", train_data.head())

valid_data = pd.read_csv("corpdata/gpt/large-762M-k40.valid.csv")
print("Validation data shape:", valid_data.shape)
print("Validation data head:", valid_data.head())

test_data = pd.read_csv("corpdata/gpt/large-762M-k40.test.csv")
print("Test data shape:", test_data.shape)
print("Test data head:", test_data.head())

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

print("Best Model Summary:")
print(best_model.summary())
print("Best Hyperparameters:")
print(tuner.get_best_hyperparameters(num_trials=1)[0].values)
best_model.save("gpt2_model.h5")


def generate_response(input_str):
    input_data = pd.Series([input_str])
    prediction = best_model.predict(input_data)[0][0]
    return prediction


print("Hello! I am a chatbot. How can I help you today?")
while True:
    user_input = input("> ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    # Generate a response using the best model
    response = generate_response(user_input)
    print("Model response:", response)
