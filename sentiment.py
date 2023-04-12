import os
import pandas as pd
import tensorflow as tf
from keras import layers
from colorama import Fore, Style
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
from keras.callbacks import EarlyStopping

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
sentiment_data = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url, extract=True)
train_data = pd.read_csv(
    os.path.join(os.path.dirname(sentiment_data), "aclImdb", "train.csv")
)
test_data = pd.read_csv(
    os.path.join(os.path.dirname(sentiment_data), "aclImdb", "test.csv")
)

x_train = train_data["review"].values
y_train = train_data["sentiment"].values
x_test = test_data["review"].values
y_test = test_data["sentiment"].values


def build_sentiment_analysis_model(hp):
    model = tf.keras.Sequential()
    model.add(layers.Embedding(input_dim=10000, output_dim=128))
    model.add(
        layers.SpatialDropout1D(
            rate=hp.Float(
                "spatial_dropout_rate", min_value=0.2, max_value=0.5, step=0.1
            )
        )
    )
    model.add(
        layers.LSTM(
            units=hp.Int("lstm_units", min_value=64, max_value=256, step=64),
            return_sequences=True,
        )
    )
    model.add(
        layers.GRU(
            units=hp.Int("gru_units", min_value=64, max_value=256, step=64),
            return_sequences=True,
        )
    )
    model.add(
        layers.Bidirectional(
            layers.GRU(
                units=hp.Int(
                    "bidirectional_gru_units", min_value=64, max_value=256, step=64
                ),
                return_sequences=True,
            )
        )
    )
    model.add(layers.GlobalMaxPooling1D())
    model.add(
        layers.Dense(
            units=hp.Int("dense_units", min_value=64, max_value=512, step=64),
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(
                hp.Float("l2_regularizer", min_value=0.001, max_value=0.01, step=0.001)
            ),
        )
    )
    model.add(
        layers.Dropout(
            rate=hp.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1)
        )
    )
    model.add(layers.Dense(units=1, activation="sigmoid"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


hypermodel = HyperModel(build_sentiment_analysis_model)
x_train = tf.keras.preprocessing.sequence.pad_sequences(
    x_train, maxlen=256, padding="pre", truncating="pre"
)
x_test = tf.keras.preprocessing.sequence.pad_sequences(
    x_test, maxlen=256, padding="pre", truncating="pre"
)

early_stopping = EarlyStopping(monitor="val_loss", patience=3)
tuner = RandomSearch(
    hypermodel, objective="val_accuracy", seed=42, max_trials=10, overwrite=True
)

tuner.search(
    x_train, y_train, epochs=10, validation_split=0.2, callbacks=[early_stopping]
)

best_model = tuner.get_best_models(1)[0]
best_model.save("best_sentiment_analysis_model.h5")
print(f"{Fore.GREEN}Best model saved successfully as best_sentiment_analysis_model.h5")
print(Style.RESET_ALL)
