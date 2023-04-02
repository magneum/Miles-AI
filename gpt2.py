import keras
import pandas as pd
from colorama import Fore as Fr
from colorama import Style as Sr
from keras.preprocessing.text import Tokenizer
from tensorflow.python.keras import layers, models
from keras_tuner import HyperParameters, RandomSearch
from tensorflow.python.keras.regularizers import l1, l2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers.schedules import ExponentialDecay


datasets = [
    "large-762M",
    "large-762M-k40",
    "medium-345M-k40",
    "medium-345M",
    "small-117M-k40",
    "small-117M",
    "xl-1542M-k40",
    "xl-1542M",
]

for dataset in datasets:
    print(f"{Fr.YELLOW}{dataset} read_csv(){Sr.RESET_ALL}")
    train_data = pd.read_csv(f"corpdata/gpt/{dataset}.train.csv")
    valid_data = pd.read_csv(f"corpdata/gpt/{dataset}.valid.csv")
    test_data = pd.read_csv(f"corpdata/gpt/{dataset}.test.csv")

    print(f"{Fr.YELLOW}{dataset} astype(str){Sr.RESET_ALL}")
    train_data["text"] = train_data["text"].astype(str)
    valid_data["text"] = valid_data["text"].astype(str)
    test_data["text"] = test_data["text"].astype(str)

    print(f"{Fr.YELLOW}{dataset} Tokenizer(){Sr.RESET_ALL}")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data["text"].values)

    print(f"{Fr.YELLOW}{dataset} texts_to_sequences(){Sr.RESET_ALL}")
    train_sequences = tokenizer.texts_to_sequences(train_data["text"].values)
    valid_sequences = tokenizer.texts_to_sequences(valid_data["text"].values)
    test_sequences = tokenizer.texts_to_sequences(test_data["text"].values)

    print(f"{Fr.YELLOW}{dataset} pad_sequences(){Sr.RESET_ALL}")
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


class CustomModelBuilder:
    def __init__(
        self, input_shape, use_L1_regularization=False, use_L2_regularization=False
    ):
        self.input_shape = input_shape
        self.use_L1_regularization = use_L1_regularization
        self.use_L2_regularization = use_L2_regularization

    def build_model(self, hp):
        model = models.Sequential()
        model.add(
            layers.Dense(
                units=hp.Int("units_1", min_value=32, max_value=512, step=32),
                activation=hp.Choice(
                    "activation_1", values=["relu", "sigmoid", "tanh"]
                ),
                kernel_regularizer=l1(
                    hp.Choice("l1_regularization", values=[0.0, 1e-5, 1e-4])
                )
                if self.use_L1_regularization
                else None,
                bias_regularizer=l2(
                    hp.Choice("l2_regularization", values=[0.0, 1e-5, 1e-4])
                )
                if self.use_L2_regularization
                else None,
                input_shape=self.input_shape,
            )
        )

        for i in range(hp.Int("num_layers", 1, 4)):
            model.add(
                layers.Dense(
                    units=hp.Int(f"units_{i+2}", min_value=32, max_value=512, step=32),
                    activation=hp.Choice(
                        f"activation_{i+2}", values=["relu", "sigmoid", "tanh"]
                    ),
                    kernel_regularizer=l1(
                        hp.Choice(f"l1_regularization_{i+1}", values=[0.0, 1e-5, 1e-4])
                    )
                    if self.use_L1_regularization
                    else None,
                    bias_regularizer=l2(
                        hp.Choice(f"l2_regularization_{i+1}", values=[0.0, 1e-5, 1e-4])
                    )
                    if self.use_L2_regularization
                    else None,
                )
            )

        model.add(layers.Dense(1, activation="sigmoid"))

        learning_rate = ExponentialDecay(
            initial_learning_rate=hp.Float(
                "initial_learning_rate", min_value=1e-5, max_value=1e-3, sampling="log"
            ),
            decay_steps=10000,
            decay_rate=hp.Float(
                "decay_rate", min_value=0.1, max_value=1, sampling="log"
            ),
        )
        optimizer = keras.optimizers.Adam(learning_rate)

        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy", "mae"],
        )

        return model


tuner = RandomSearch(
    CustomModelBuilder,
    objective="val_accuracy",
    max_trials=5,
    executions_per_trial=3,
    directory="my_dir",
    project_name="gpt2_tuning",
    hyperparameters=HyperParameters(),
    overwrite=True,
)

tuner.search(
    train_inputs, train_outputs, epochs=5, validation_data=(valid_inputs, valid_outputs)
)

best_model = tuner.get_best_models(num_models=1)[0]
test_loss, test_acc, test_mae = best_model.evaluate(test_inputs, test_outputs)

print("Best Model Summary:")
print(best_model.summary())
print("Best Hyperparameters:")
print(tuner.get_best_hyperparameters(num_trials=1)[0].values)
best_model.save("gpt2_model.h5")
