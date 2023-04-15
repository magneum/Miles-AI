import os
import numpy
import pandas
from keras import Sequential
from colorama import Fore, Style
from keras.optimizers import Adam
from keras_tuner.tuners import Hyperband
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# ========================================================= Magneum =========================================================
X_Index = []
Y_Index = []
nSeed = 22
verbose = 1
patience = 10
nEpochs = 200
nValsplit = 0.2
batch_size = 12
hyper_directory = "models/Face_Emo/Emotion"
dataset_path = "corpdata/csv/fer2013/fer2013.csv"
model_save_path = "models/Face_Emo/Face_Emo_Model.h5"


# ========================================================= Magneum =========================================================
file_path = os.path.abspath(__file__)
file_name = os.path.basename(file_path)
_path = "models/FaceEmo"
if not os.path.exists(_path):
    os.makedirs(_path)


# ========================================================= Magneum =========================================================
Fer2013 = pandas.read_csv(dataset_path)
for index, row in Fer2013.iterrows():
    pixels = numpy.fromstring(row["pixels"], dtype="uint8", sep=" ")
    image = pixels.reshape((48, 48, 1)).astype("float32") / 255.0
    label = row["emotion"]
    X_Index.append(image)
    Y_Index.append(label)
X_Index = numpy.array(X_Index)
Y_Index = numpy.array(Y_Index)


# ========================================================= Magneum =========================================================
def Hyper_Builder(hp):
    model = Sequential()
    model.add(
        Conv2D(
            filters=hp.Int("filters_1", 32, 128, step=32),
            kernel_size=hp.Choice("kernel_size_1", values=[3, 5]),
            activation="relu",
            input_shape=(48, 48, 1),
        )
    )
    model.add(MaxPooling2D(pool_size=hp.Choice("pool_size_1", values=[4, 5])))
    for i in range(hp.Int("nblocks", 1, 4)):
        model.add(
            Conv2D(
                filters=hp.Int("filters_" + str(i + 2), 32, 128, step=32),
                kernel_size=hp.Choice("kernel_size_" + str(i + 2), values=[3, 5]),
                activation="relu",
                padding="same",
            )
        )
        model.add(
            MaxPooling2D(
                pool_size=hp.Choice("pool_size_" + str(i + 2), values=[4, 5]),
                padding="same",
            )
        )

    model.add(Flatten())
    model.add(Dense(units=hp.Int("units", 128, 512, step=32), activation="relu"))
    model.add(Dense(7, activation="softmax"))
    model.compile(
        optimizer=Adam(hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    return model


# ========================================================= Magneum =========================================================


Hyper_Tuner = Hyperband(
    Hyper_Builder,
    seed=nSeed,
    max_epochs=nEpochs,
    project_name="Emotion",
    objective="val_accuracy",
    directory=hyper_directory,
)
Hyper_Tuner.search(
    x=X_Index,
    y=Y_Index,
    epochs=nEpochs,
    verbose=verbose,
    batch_size=batch_size,
    validation_split=nValsplit,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    ],
)


# ========================================================= Magneum =========================================================
BestHP = Hyper_Tuner.get_best_hyperparameters(1)[0]
Hyper_Model = Hyper_Builder(BestHP)
Hyper_Model.fit(X_Index, Y_Index, epochs=nEpochs, validation_split=0.2)
Hyper_Model.save(model_save_path)
