import numpy as np
import pandas as pd
from keras import layers
from tensorflow import keras
from keras_tuner.tuners import RandomSearch

X_Index = []
Y_Index = []
nEpochs = 10
nValsplit = 0.2
fer2013 = pd.read_csv("corpdata/csv/Fer2013.csv")
for index, row in fer2013.iterrows():
    pixels = np.fromstring(row["pixels"], dtype="uint8", sep=" ")
    image = pixels.reshape((48, 48, 1)).astype("float32") / 255.0
    label = row["emotion"]
    X_Index.append(image)
    Y_Index.append(label)
X_Index = np.array(X_Index)
Y_Index = np.array(Y_Index)


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
    model.add(layers.MaxPooling2D(pool_size=hp.Choice("pool_size_1", values=[4, 5])))
    for i in range(hp.Int("nblocks", 1, 4)):
        model.add(
            layers.Conv2D(
                filters=hp.Int("filters_" + str(i + 2), 32, 128, step=32),
                kernel_size=hp.Choice("kernel_size_" + str(i + 2), values=[3, 5]),
                activation="relu",
                padding="same",
            )
        )
        model.add(
            layers.MaxPooling2D(
                pool_size=hp.Choice("pool_size_" + str(i + 2), values=[4, 5]),
                padding="same",
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


def Early_Stopping(hp):
    callbacks = []
    if hp:
        early_stopping_patience = hp.Int(
            "early_stopping_patience", min_value=1, max_value=10, default=5
        )
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=early_stopping_patience
            )
        )
    return callbacks


tuner = RandomSearch(
    build_model,
    max_trials=20,
    project_name="Emotion",
    directory="models/Face_Recog/Emotion",
    objective=["val_accuracy", "val_loss"],
)

tuner.search(
    x=X_Index,
    y=Y_Index,
    verbose=1,
    batch_size=8,
    epochs=nEpochs,
    callbacks=Early_Stopping,
    validation_split=nValsplit,
)

best_hp = tuner.get_best_hyperparameters(1)[0]
best_model = build_model(best_hp)

best_model.fit(X_Index, Y_Index, epochs=nEpochs, validation_split=0.2)

best_model.save("face_emotion_model.h5")
