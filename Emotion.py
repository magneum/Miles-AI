import numpy as np
import pandas as pd
from keras import layers
from tensorflow import keras
from keras_tuner.tuners import RandomSearch, Hyperband

X = []
y = []
num_epocs = 1000
num_val_split = 0.2
fer2013 = pd.read_csv("corpdata/csv/fer2013.csv")
for index, row in fer2013.iterrows():
    pixels = np.fromstring(row["pixels"], dtype="uint8", sep=" ")
    image = pixels.reshape((48, 48, 1)).astype("float32") / 255.0
    label = row["emotion"]
    X.append(image)
    y.append(label)
X = np.array(X)
y = np.array(y)


def build_model(hp):
    model = keras.Sequential()
    model.add(
        layers.Conv2D(
            filters=hp.Int("filters_1", 32, 128, step=32),
            kernel_size=hp.Choice("kernel_size_1", values=[3, 5]),
            activation="relu",
            input_shape=(48, 48, 1),
        )
    )
    model.add(layers.MaxPooling2D(pool_size=hp.Choice("pool_size_1", values=[2, 3])))
    for i in range(hp.Int("num_blocks", 1, 4)):
        model.add(
            layers.Conv2D(
                filters=hp.Int("filters_" + str(i + 2), 32, 128, step=32),
                kernel_size=hp.Choice("kernel_size_" + str(i + 2), values=[3, 5]),
                activation="relu",
            )
        )
        model.add(
            layers.MaxPooling2D(
                pool_size=hp.Choice("pool_size_" + str(i + 2), values=[2, 3])
            )
        )
    model.add(layers.Flatten())
    model.add(layers.Dense(units=hp.Int("units", 128, 512, step=32), activation="relu"))
    model.add(layers.Dense(7, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


tuner = Hyperband(
    build_model,
    max_epochs=num_epocs,
    hyperband_iterations=2,
    objective="val_accuracy",
    directory="Fer2013_Recognition",
    project_name="Fer2013_Recognition",
)
tuner = RandomSearch(
    build_model,
    max_trials=40,
    objective="val_accuracy",
    directory="Fer2013_Recognition",
    project_name="Fer2013_Recognition",
)
tuner.search(X, y, epochs=num_epocs, validation_split=num_val_split)

best_hp = tuner.get_best_hyperparameters(1)[0]
best_model = build_model(best_hp)
best_model.fit(X, y, epochs=num_epocs, validation_split=num_val_split)
best_model.save("Fer2013_Recognition.h5")
