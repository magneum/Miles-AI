import os
import numpy
import pandas
from keras import Sequential
from colorama import Fore, Style
from keras.optimizers import Adam
from keras_tuner.tuners import Hyperband
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# print(f"{Fore.YELLOW}{Style.BRIGHT}Code Description: FaceProcessing.py")
# print(f"{Fore.WHITE}{Style.BRIGHT}------------------")
# print(f"{Fore.CYAN}{Style.BRIGHT}The code is a Python script that uses Keras and TensorFlow libraries to perform hyperparameter tuning for a convolutional neural network (CNN) model for facial emotion recognition.")
# print(f"{Fore.CYAN}{Style.BRIGHT}The dataset used is Fer2013, which is loaded from a CSV file.")
# print(f"\n{Fore.CYAN}{Style.BRIGHT}The script uses the keras_tuner library to perform random search hyperparameter tuning. The Hyper_Builder() function defines the architecture of the CNN model and compiles it with hyperparameters such as filters, kernel size, pool size, units, and learning rate. The hyperparameters are sampled from defined ranges using the hp object, which is passed as an argument to the function.")
# print(f"\n{Fore.CYAN}{Style.BRIGHT}The Early_Stopping() function defines early stopping as a callback for the model during training, which prevents overfitting. The patience for early stopping is also hyperparameterized using the hp object.")
# print(f"\n{Fore.CYAN}{Style.BRIGHT}The Hyperband tuner is then created with the Hyper_Builder() function as the model-building function, and the maximum number of trials, project name, and objective for tuning (in this case, 'val_accuracy') are specified. The tuner searches for the best hyperparameters using the search() function, which takes the input data (X_Index and Y_Index), number of epochs, batch size, and other parameters.")
# print(f"\n{Fore.CYAN}{Style.BRIGHT}After tuning is completed, the best hyperparameters and their corresponding model performance metrics (such as accuracy, loss) are printed. The best model is then trained with the optimal hyperparameters and evaluated on the validation set. Finally, the model is saved to disk for future use.")
# print(f"{Style.RESET_ALL}")


# Hyper Variables
X_Index = []
Y_Index = []
nSeed = 22
verbose = 1
nEpochs = 200
nValsplit = 0.2
batch_size = 12
hyper_directory = "models/Face_Emo/Emotion"
dataset_path = "corpdata/csv/fer2013/fer2013.csv"
model_save_path = "models/Face_Emo/Face_Emo_Model.h5"


file_path = os.path.abspath(__file__)
file_name = os.path.basename(file_path)
_path = "models/FaceEmo"
if not os.path.exists(_path):
    os.makedirs(_path)

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
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    return model


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
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ],
)

BestHP = Hyper_Tuner.get_best_hyperparameters(1)[0]
Hyper_Model = Hyper_Builder(BestHP)
Hyper_Model.fit(X_Index, Y_Index, epochs=nEpochs, validation_split=0.2)
Hyper_Model.save(model_save_path)
