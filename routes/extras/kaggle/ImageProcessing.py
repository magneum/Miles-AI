from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from keras_tuner import HyperModel, Hyperband, RandomSearch
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
from keras.regularizers import l1, l2
from keras.models import Sequential
import numpy as np


class ImageProcessingHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes, class_weights=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.class_weights = class_weights

    def build(self, hp):
        model = Sequential()

        model.add(
            Conv2D(
                filters=hp.Int("filters", min_value=32, max_value=256, step=32),
                kernel_size=hp.Choice("kernel_size", values=[3, 5]),
                activation="relu",
                input_shape=self.input_shape,
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(
            Conv2D(
                filters=hp.Int("filters", min_value=64, max_value=512, step=64),
                kernel_size=hp.Choice("kernel_size", values=[3, 5]),
                activation="relu",
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(
            Dense(
                units=hp.Int("dense_units", min_value=64, max_value=512, step=64),
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
                units=self.num_classes,
                activation="softmax",
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
            loss="sparse_categorical_crossentropy",
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


# Load and preprocess the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

class_weights = ImageProcessingHyperModel.compute_class_weight(y_train)

input_shape = X_train.shape[1:]
num_classes = 10
hypermodel = ImageProcessingHyperModel(input_shape, num_classes, class_weights)

tuner = Hyperband(
    hypermodel,
    objective="val_accuracy",
    max_epochs=10,
    hyperband_iterations=2,
    directory="fashion_mnist_hyperopt",
    project_name="fashion_mnist",
)

tuner = RandomSearch(
    hypermodel,
    objective="val_accuracy",
    max_trials=10,
    directory="fashion_mnist_hyperopt",
    project_name="fashion_mnist",
)

tuner.search(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=hypermodel.get_callbacks(),
)

best_hp = tuner.get_best_hyperparameters(1)[0]
best_model = hypermodel.build(best_hp)

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
