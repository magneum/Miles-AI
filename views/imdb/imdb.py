import tensorflow as tf
from keras_tuner import HyperModel
from tensorflow.keras.datasets import imdb
from keras_tuner.tuners import RandomSearch
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.regularizers import L1, L2
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization


class MyHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential()
        num_layers = hp.Int("num_layers", 1, 8)
        for i in range(num_layers):
            units = hp.Int(f"dense_{i+1}_units", min_value=128, max_value=512, step=32)
            activation = hp.Choice(
                f"activation_{i+1}", values=["relu", "sigmoid", "tanh"]
            )
            model.add(Dense(units=units, activation=activation))

            dropout_rate = hp.Float(
                f"dropout_{i+1}", min_value=0.0, max_value=0.5, step=0.1
            )
            model.add(Dropout(rate=dropout_rate))

        optimizer = hp.Choice("optimizer", values=["adam", "rmsprop", "sgd"])
        hp_learning_rate = hp.Float(
            "learning_rate", min_value=1e-5, max_value=1e-2, sampling="LOG"
        )
        if optimizer == "adam":
            optimizer = Adam(learning_rate=hp_learning_rate)
        elif optimizer == "rmsprop":
            optimizer = RMSprop(learning_rate=hp_learning_rate)
        else:
            optimizer = SGD(learning_rate=hp_learning_rate)

        if hp.Boolean("use_L1_regularization", default=True):
            L1_rate = hp.Float(
                "L1_rate", min_value=1e-5, max_value=1e-2, sampling="LOG", default=1e-5
            )
            model.add(
                Dense(
                    units=units,
                    activation=activation,
                    kernel_regularizer=L1(L1_rate),
                )
            )

        if hp.Boolean("use_L2_regularization", default=True):
            L2_rate = hp.Float(
                "L2_rate", min_value=1e-5, max_value=1e-2, sampling="LOG"
            )
            model.add(
                Dense(
                    units=units,
                    activation=activation,
                    kernel_regularizer=L2(L2_rate),
                )
            )

        if hp.Boolean("use_batch_normalization", default=True):
            model.add(BatchNormalization())

        if hp.Boolean("use_dropout", default=True):
            dropout_rate = hp.Float(
                "final_dropout_rate", min_value=0.0, max_value=0.5, step=0.1
            )
            model.add(Dropout(rate=dropout_rate))

        model.add(Dense(self.num_classes, activation="softmax"))
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model


hypermodel = MyHyperModel(input_shape=(5000,), num_classes=2)

tuner = RandomSearch(
    hypermodel,
    objective="val_accuracy",
    max_trials=10,
    directory="views/imdb/hyperModel",
    project_name="imdb_sentiment_analysis",
)

best_hyperparameters = tuner.get_best_hyperparameters()[0]
best_model = hypermodel.build(best_hyperparameters)

max_len = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

x_train, x_val = x_train[:20000], x_train[20000:]
y_train, y_val = y_train[:20000], y_train[20000:]


best_model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_val, y_val),
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],
)

best_model.evaluate(x_test, y_test)
y_pred = best_model.predict(x_test)
best_model.save("imdb_sentiment.h5")
