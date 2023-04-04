import tensorflow
import numpy as np
from tensorflow import keras
from keras_tuner import HyperModel


class hyperModel(HyperModel):
    def __init__(
        self,
        input_shape,
        num_classes,
        use_early_stopping=False,
        embeddings_index=None,
        words=None,
        hp=None,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_early_stopping = use_early_stopping
        self.embeddings_index = embeddings_index
        self.words = words
        self.hp = hp

    def build(self, hp):
        model = keras.Sequential()

        # Embedding layer
        embedding_dim = 300
        embedding_matrix = np.zeros((len(self.words), embedding_dim))
        for i, word in enumerate(self.words):
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        embedding_layer = keras.layers.Embedding(
            len(self.words),
            embedding_dim,
            weights=[embedding_matrix],
            input_length=self.input_shape[0],
            trainable=False,
        )

        # LSTM layer
        lstm_units = hp.Int("lstm_units", min_value=32, max_value=512, step=32)
        model.add(embedding_layer)
        model.add(
            keras.layers.LSTM(units=lstm_units, dropout=0.2, recurrent_dropout=0.2)
        )

        # Dense layers
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

        if hp.Boolean("use_L1_regularization", default=True):
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

        if hp.Boolean("use_L2_regularization", default=True):
            L2_rate = hp.Float(
                "L2_rate", min_value=1e-5, max_value=1e-2, sampling="LOG"
            )
            model.add(
                keras.layers.Dense(
                    units=units,
                    activation=activation,
                    kernel_regularizer=keras.regularizers.L2(L2_rate),
                )
            )

        model.add(keras.layers.Dense(self.num_classes, activation="softmax"))

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def get_callbacks(self):
        # initialize an empty list of callbacks
        callbacks = []
        # check if the hyperparameters dictionary is not empty
        if self.hp:
            # sample a value for early stopping patience from the search space
            early_stopping_patience = self.hp.Int(
                "early_stopping_patience", min_value=1, max_value=10, default=5
            )
            # create an instance of the EarlyStopping callback and append it to the list of callbacks
            callbacks.append(
                tensorflow.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=early_stopping_patience
                )
            )
        # return the list of callbacks
        return callbacks
