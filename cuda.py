import os
import torch
import numpy as np
from tensorflow import keras
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print("Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
    print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")


class GPT2HyperModel(HyperModel):
    # Initialize the class with max_length and vocab_size
    def __init__(self, max_length, vocab_size):
        self.max_length = max_length
        self.vocab_size = vocab_size

    # Build the model with hyperparameters using Keras API
    def build(self, hp):
        # Define the input layer
        inputs = keras.layers.Input(shape=(self.max_length,))
        # Define the embedding layer
        x = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=hp.Int("embedding_dim", min_value=32, max_value=512, step=32),
        )(inputs)
        # Add multiple LSTM layers with different number of units and dropouts
        for i in range(hp.Int("num_layers", 2, 12)):
            x = keras.layers.LSTM(
                units=hp.Int(
                    "num_units_" + str(i), min_value=32, max_value=1024, step=32
                ),
                return_sequences=True,
            )(x)
            x = keras.layers.Dropout(
                hp.Float("dropout_" + str(i), min_value=0.1, max_value=0.5, step=0.1)
            )(x)
        # Define the output layer with softmax activation
        outputs = keras.layers.Dense(units=self.vocab_size, activation="softmax")(x)
        # Define the optimizer with learning rate
        optimizer = keras.optimizers.Adam(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-5, max_value=1e-2, sampling="log"
            )
        )
        # Compile the model with sparse_categorical_crossentropy loss and accuracy metrics
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        # Return the compiled model
        return model


max_length = 1024
vocab_size = 50257

# Define datasets to use
datasets = [
    {
        "train_file": "corpdata/gpt/large-762M-k40.train.csv",
        "valid_file": "corpdata/gpt/large-762M-k40.valid.csv",
        "test_file": "corpdata/gpt/large-762M-k40.test.csv",
        "name": "large-762M-k40",
    },
    {
        "train_file": "corpdata/gpt/medium-345M-k40.train.csv",
        "valid_file": "corpdata/gpt/medium-345M-k40.valid.csv",
        "test_file": "corpdata/gpt/medium-345M-k40.test.csv",
        "name": "medium-345M-k40",
    },
]

# Loop through datasets and train models
for dataset in datasets:
    # Get file names and dataset name
    train_file = dataset["train_file"]
    valid_file = dataset["valid_file"]
    test_file = dataset["test_file"]
    name = dataset["name"]

    # Load and preprocess data
    with open(train_file, "r", encoding="utf-8") as f:
        train_data = f.read().splitlines()

    with open(valid_file, "r", encoding="utf-8") as f:
        valid_data = f.read().splitlines()

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = f.read().splitlines()

    train_data = np.loadtxt(
        train_file,
        delimiter=",",
        dtype=np.int,
        skiprows=1,
        converters={2: lambda x: int(float(x))},
    )
    valid_data = np.loadtxt(valid_file, delimiter=" ", dtype=np.int, skiprows=1)
    test_data = np.loadtxt(test_file, delimiter=" ", dtype=np.int, skiprows=1)

    train_labels = train_data[:, 1:]
    train_data = train_data[:, :-1]

    valid_labels = valid_data[:, 1:]
    valid_data = valid_data[:, :-1]

    test_labels = test_data[:, 1:]
    test_data = test_data[:, :-1]

    # Define and train hypermodel
    hypermodel = GPT2HyperModel(max_length=max_length, vocab_size=vocab_size)

    tuner = RandomSearch(
        hypermodel,
        objective="val_accuracy",
        max_trials=10,
        directory="my_dir",
        project_name=name,
    )

    tuner.search(
        x=train_data,
        y=train_labels,
        validation_data=(valid_data, valid_labels),
        epochs=10,
    )

    # Evaluate test set and print results
    test_loss, test_acc = best_model.evaluate(test_data, test_labels, verbose=2)
    print(f"Test accuracy for {name}: {test_acc}")

    # Save best model and hyperparameters
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(f"{dataset['name']}_best_model.h5")
    best_hyperparameters = tuner.get_best_hyperparameters()
    print(best_hyperparameters)

    # Save all models tried during search
    trial_num = 1
    for trial in tuner.oracle.get_trials():
        model = tuner.hypermodel.build(trial.hyperparameters)
        model.set_weights(trial.get_best_weights()[0])
        model.save(os.path.join(f"{dataset['name']}_my_dir", f"trial_{trial_num}.h5"))
        trial_num += 1
