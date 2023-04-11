from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from kerastuner import HyperModel, Hyperband, RandomSearch
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1, l2
from keras.models import Sequential
from keras.datasets import mnist
import numpy as np


class TextProcessingHyperModel(HyperModel):
    def __init__(self, vocab_size, max_length, embedding_matrix, class_weights=None):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_matrix = embedding_matrix
        self.class_weights = class_weights

    def build(self, hp):
        model = Sequential()

        # Add pre-trained GloVe embedding layer
        embedding_layer = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_matrix.shape[1],
            weights=[self.embedding_matrix],
            input_length=self.max_length,
            trainable=hp.Boolean("trainable_embedding", default=False),
        )
        model.add(embedding_layer)

        model.add(
            LSTM(
                units=hp.Int("lstm_units", min_value=32, max_value=256, step=32),
                dropout=hp.Float(
                    "lstm_dropout", min_value=0.1, max_value=0.5, step=0.1
                ),
                recurrent_dropout=hp.Float(
                    "lstm_recurrent_dropout", min_value=0.1, max_value=0.5, step=0.1
                ),
                kernel_regularizer=l1(
                    hp.Float("l1_reg", min_value=0.0, max_value=0.01, step=0.001)
                ),
                recurrent_regularizer=l2(
                    hp.Float("l2_reg", min_value=0.0, max_value=0.01, step=0.001)
                ),
            )
        )
        model.add(Dropout(hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)))
        model.add(
            Dense(
                units=hp.Int("dense_units", min_value=32, max_value=256, step=32),
                activation="relu",
                kernel_regularizer=l1(
                    hp.Float("dense_l1_reg", min_value=0.0, max_value=0.01, step=0.001)
                ),
                bias_regularizer=l2(
                    hp.Float("dense_l2_reg", min_value=0.0, max_value=0.01, step=0.001)
                ),
            )
        )
        model.add(
            Dropout(hp.Float("dense_dropout", min_value=0.1, max_value=0.5, step=0.1))
        )
        model.add(
            Dense(
                units=1,
                activation="sigmoid",
                kernel_regularizer=l1(
                    hp.Float("output_l1_reg", min_value=0.0, max_value=0.01, step=0.001)
                ),
                bias_regularizer=l2(
                    hp.Float("output_l2_reg", min_value=0.0, max_value=0.01, step=0.001)
                ),
            )
        )

        # Compile the model
        model.compile(
            optimizer=hp.Choice(
                "optimizer", values=["adam", "rmsprop", "sgd"], default="adam"
            ),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        return model

    @staticmethod
    def compute_class_weight(y_train):
        class_weights = compute_class_weight("balanced", np.unique(y_train), y_train)
        class_weights = dict(zip(np.unique(y_train), class_weights))
        return class_weights

    def get_callbacks(self):
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, verbose=1),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=3,
                verbose=1,
                class_weight=self.class_weights,
            ),
        ]
        return callbacks


(X_train, y_train), (X_test, y_test) = mnist.load_data()

vocab_size = 256
max_length = 100
X_train = X_train.astype("str").tolist()
X_test = X_test.astype("str").tolist()

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

class_weights = TextProcessingHyperModel.compute_class_weight(y_train)
hypermodel = TextProcessingHyperModel(
    vocab_size, max_length, class_weights=class_weights
)

tuner = Hyperband(
    hypermodel,
    objective="val_accuracy",
    max_epochs=10,
    hyperband_iterations=2,
    directory="hyperband",
    project_name="mnist_small_talk",
)

tuner = RandomSearch(
    hypermodel,
    objective="val_accuracy",
    max_trials=10,
    directory="random_search",
    project_name="mnist_small_talk",
)


checkpoint_callback = ModelCheckpoint(
    filepath="best_model.h5", monitor="val_accuracy", save_best_only=True
)

tuner.search(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint_callback, *hypermodel.get_callbacks()],
)

best_model = tuner.get_best_models(1)[0]
best_model.evaluate(X_test, y_test)


best_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=hypermodel.get_callbacks(),
    class_weight=class_weights,
)

test_loss, test_acc = best_model.evaluate(X_test, y_test, batch_size=64)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)
