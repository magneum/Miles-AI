from kerastuner.tuners import RandomSearch, Hyperband
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.utils import shuffle
from keras import layers
import tensorflow as tf
import numpy as np


class MyHyperModel(tf.keras.Model):
    def __init__(self, hp):
        super(MyHyperModel, self).__init__()
        self.hp = hp

    def build(self, hp):
        model = Sequential()
        model.add(
            layers.Conv2D(
                filters=self.hp.Int(
                    "conv1_filters", min_value=32, max_value=128, step=32
                ),
                kernel_size=self.hp.Choice("conv1_kernel", values=[3, 5]),
                activation="relu",
                input_shape=(28, 28, 1),
            )
        )
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.BatchNormalization())

        model.add(
            layers.Conv2D(
                filters=self.hp.Int(
                    "conv2_filters", min_value=64, max_value=256, step=64
                ),
                kernel_size=self.hp.Choice("conv2_kernel", values=[3, 5]),
                activation="relu",
            )
        )
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.BatchNormalization())

        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                units=self.hp.Int(
                    "dense_units", min_value=128, max_value=512, step=128
                ),
                activation="relu",
            )
        )
        model.add(
            layers.Dropout(
                rate=self.hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1)
            )
        )
        model.add(layers.Dense(10, activation="softmax"))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.hp.Float(
                    "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
                )
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


def load_and_preprocess_dataset():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = load_and_preprocess_dataset()

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

tuner = Hyperband(
    MyHyperModel,
    objective="val_accuracy",
    max_epochs=10,
    hyperband_iterations=2,
    directory="Emotion_mnist",
    project_name="Emotion_mnist",
)

tuner = RandomSearch(
    MyHyperModel,
    objective="val_accuracy",
    max_trials=10,
    directory="Emotion_mnist",
    project_name="Emotion_mnist",
)

tuner.search(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_val, y_val),
)

best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hyperparameters}")
best_model.fit(X_train, y_train, epochs=20)
best_model.save("emotion_detection_model.h5")
