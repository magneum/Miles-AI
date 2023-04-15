import numpy
import keras
import pandas
from keras.callbacks import EarlyStopping
from keras_tuner.tuners import RandomSearch


X_Index = []
Y_Index = []
nEpochs = 100
nValsplit = 0.2
hyper_directory = "models/Face_Emo/Emotion"
dataset_path = "/kaggle/input/fer2013/fer2013.csv"
model_save_path = "models/Face_Emo/Face_Emotion_Model.h5"

Fer2013 = pandas.read_csv(dataset_path)
for index, row in Fer2013.iterrows():
    pixels = numpy.fromstring(row["pixels"], dtype="uint8", sep=" ")
    image = pixels.reshape((48, 48, 1)).astype("float32") / 255.0
    label = row["emotion"]
    X_Index.append(image)
    Y_Index.append(label)
X_Index = numpy.array(X_Index)
Y_Index = numpy.array(Y_Index)


def Hyper_Builder(hp):
    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(
            filters=hp.Int("filters_1", 32, 128, step=32),
            kernel_size=hp.Choice("kernel_size_1", values=[3, 5]),
            activation="relu",
            input_shape=(48, 48, 1),
        )
    )
    model.add(
        keras.layers.MaxPooling2D(pool_size=hp.Choice("pool_size_1", values=[4, 5]))
    )
    for i in range(hp.Int("nblocks", 1, 4)):
        model.add(
            keras.layers.Conv2D(
                filters=hp.Int("filters_" + str(i + 2), 32, 128, step=32),
                kernel_size=hp.Choice("kernel_size_" + str(i + 2), values=[3, 5]),
                activation="relu",
                padding="same",
            )
        )
        model.add(
            keras.layers.MaxPooling2D(
                pool_size=hp.Choice("pool_size_" + str(i + 2), values=[4, 5]),
                padding="same",
            )
        )
    model.add(keras.layers.Flatten())
    model.add(
        keras.layers.Dense(units=hp.Int("units", 128, 512, step=32), activation="relu")
    )
    model.add(keras.layers.Dense(7, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    return model


Hyper_Tuner = RandomSearch(
    Hyper_Builder,
    max_trials=20,
    project_name="Emotion",
    objective="val_accuracy",
    directory=hyper_directory,
)

Hyper_Tuner.search(
    x=X_Index,
    y=Y_Index,
    verbose=1,
    batch_size=8,
    epochs=nEpochs,
    validation_split=nValsplit,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ],
)

BestHP = Hyper_Tuner.get_best_hyperparameters(1)[0]
Hyper_Model = Hyper_Builder(BestHP)
Hyper_Model.fit(X_Index, Y_Index, epochs=nEpochs, validation_split=0.2)
Hyper_Model.save(model_save_path)
