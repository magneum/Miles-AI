import pandas as pd
from tensorflow import keras
from tensorflow import keras
from colorama import Fore as Fr
from colorama import Style as Sr
from keras_tuner import Hyperband
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional


for dataset in [
    "large-762M",
    "large-762M-k40",
    "medium-345M-k40",
    "medium-345M",
    "small-117M-k40",
    "small-117M",
    "xl-1542M-k40",
    "xl-1542M",
]:
    print(f"{Fr.GREEN}DATASET INFO: {dataset} {Sr.RESET_ALL}read_csv()")
    train_data = pd.read_csv(f"corpdata/gpt/{dataset}.train.csv")
    valid_data = pd.read_csv(f"corpdata/gpt/{dataset}.valid.csv")
    test_data = pd.read_csv(f"corpdata/gpt/{dataset}.test.csv")

    print(f"{Fr.GREEN}DATASET INFO: {dataset} {Sr.RESET_ALL}astype(str)")
    train_data["text"] = train_data["text"].astype(str)
    valid_data["text"] = valid_data["text"].astype(str)
    test_data["text"] = test_data["text"].astype(str)

    print(f"{Fr.GREEN}DATASET INFO: {dataset} {Sr.RESET_ALL}Tokenizer()")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data["text"].values)

    print(f"{Fr.GREEN}DATASET INFO: {dataset} {Sr.RESET_ALL}texts_to_sequences()")
    train_sequences = tokenizer.texts_to_sequences(train_data["text"].values)
    valid_sequences = tokenizer.texts_to_sequences(valid_data["text"].values)
    test_sequences = tokenizer.texts_to_sequences(test_data["text"].values)

    print(f"{Fr.GREEN}DATASET INFO: {dataset} {Sr.RESET_ALL}pad_sequences()")
    max_length = 128
    train_inputs = pad_sequences(train_sequences, maxlen=max_length, padding="post")
    valid_inputs = pad_sequences(valid_sequences, maxlen=max_length, padding="post")
    test_inputs = pad_sequences(test_sequences, maxlen=max_length, padding="post")

    train_outputs = train_data["ended"].values
    valid_outputs = valid_data["ended"].values
    test_outputs = test_data["ended"].values

    print(f"{Fr.BLUE}{dataset} TRAIN INPUTS SHAPE:{Sr.RESET_ALL}", train_inputs.shape)
    print(f"{Fr.BLUE}{dataset} VALID INPUTS SHAPE:{Sr.RESET_ALL}", valid_inputs.shape)
    print(f"{Fr.BLUE}{dataset} TEST INPUTS SHAPE:{Sr.RESET_ALL}", test_inputs.shape)
    print(f"{Fr.BLUE}{dataset} TRAIN OUTPUTS SHAPE:{Sr.RESET_ALL}", train_outputs.shape)
    print(f"{Fr.BLUE}{dataset} VALID OUTPUTS SHAPE:{Sr.RESET_ALL}", valid_outputs.shape)
    print(f"{Fr.BLUE}{dataset} TEST OUTPUTS SHAPE:{Sr.RESET_ALL}", test_outputs.shape)


def build_model(hp):
    model = keras.Sequential()

    # Tune the number of LSTM layers
    for i in range(hp.Int("num_lstm_layers", 1, 3)):
        model.add(
            Bidirectional(
                LSTM(
                    units=hp.Int(
                        f"lstm_{i}_units", min_value=32, max_value=512, step=32
                    )
                )
            )
        )
        # Add dropout regularization to the LSTM layers
        model.add(
            Dropout(
                rate=hp.Float(
                    f"lstm_{i}_dropout", min_value=0.1, max_value=0.5, step=0.1
                )
            )
        )

    # Add a dense layer to reduce the dimensionality before the output layer
    model.add(
        Dense(
            units=hp.Int("dense_units", min_value=32, max_value=512, step=32),
            activation=hp.Choice("dense_activation", values=["relu", "tanh"]),
        )
    )

    # Add dropout regularization to the dense layer
    model.add(
        Dropout(rate=hp.Float("dense_dropout", min_value=0.1, max_value=0.5, step=0.1))
    )

    # Add the output layer
    model.add(Dense(units=1, activation="sigmoid"))

    # Compile the model with the specified optimizer and learning rate
    model.compile(
        optimizer=Adam(
            learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


tuner = Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=10,
    factor=3,
    directory="corpdata/gpt",
    project_name="gpt_tuning",
)

tuner.search(
    x=train_inputs,
    y=train_outputs,
    epochs=10,
    validation_data=(valid_inputs, valid_outputs),
)

best_model = tuner.get_best_models(num_models=1)[0]
test_loss, test_acc = best_model.evaluate(test_inputs, test_outputs, verbose=0)
print("Test accuracy:", test_acc)
print("Best Model Summary:")
print(best_model.summary())
print("Best Hyperparameters:")
print(tuner.get_best_hyperparameters(num_trials=1)[0].values)
best_model.save("gpt2_model.h5")
