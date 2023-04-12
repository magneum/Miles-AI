import numpy
import pandas
from keras import layers
from tensorflow import keras
from colorama import Fore, Style
from keras_tuner.tuners import RandomSearch


def print_code_description():
    print(f"{Fore.YELLOW}{Style.BRIGHT}Code Description:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{Style.BRIGHT}------------------{Style.RESET_ALL}")
    print(
        f"{Fore.CYAN}{Style.BRIGHT}The code is a Python script that uses Keras and TensorFlow libraries to perform hyperparameter tuning for a convolutional neural network (CNN) model for facial emotion recognition.{Style.RESET_ALL}"
    )
    print(
        f"{Fore.CYAN}{Style.BRIGHT}The dataset used is Fer2013, which is loaded from a CSV file.{Style.RESET_ALL}"
    )
    print(
        f"\n{Fore.CYAN}{Style.BRIGHT}The script uses the keras_tuner library to perform random search hyperparameter tuning. The build_model() function defines the architecture of the CNN model and compiles it with hyperparameters such as filters, kernel size, pool size, units, and learning rate. The hyperparameters are sampled from defined ranges using the hp object, which is passed as an argument to the function.{Style.RESET_ALL}"
    )
    print(
        f"\n{Fore.CYAN}{Style.BRIGHT}The Early_Stopping() function defines early stopping as a callback for the model during training, which prevents overfitting. The patience for early stopping is also hyperparameterized using the hp object.{Style.RESET_ALL}"
    )
    print(
        f"\n{Fore.CYAN}{Style.BRIGHT}The RandomSearch tuner is then created with the build_model() function as the model-building function, and the maximum number of trials, project name, and objective for tuning (in this case, 'val_accuracy') are specified. The tuner searches for the best hyperparameters using the search() function, which takes the input data (X_Index and Y_Index), number of epochs, batch size, and other parameters.{Style.RESET_ALL}"
    )
    print(
        f"\n{Fore.CYAN}{Style.BRIGHT}After tuning is completed, the best hyperparameters and their corresponding model performance metrics (such as accuracy, loss) are printed. The best model is then trained with the optimal hyperparameters and evaluated on the validation set. Finally, the model is saved to disk for future use.{Style.RESET_ALL}"
    )


X_Index = []
Y_Index = []
nEpochs = 10
nValsplit = 0.2


# Print loaded data information
Fer2013 = pandas.read_csv("corpdata/csv/Fer2013.csv")
print(Fore.GREEN + Style.BRIGHT + "Loaded Data Information:")
print(Fore.YELLOW + f"Data shape: {Fer2013.shape}")
print(Fore.YELLOW + f"Columns: {', '.join(Fer2013.columns)}")
print(Style.RESET_ALL)

for index, row in Fer2013.iterrows():
    pixels = numpy.fromstring(row["pixels"], dtype="uint8", sep=" ")
    image = pixels.reshape((48, 48, 1)).astype("float32") / 255.0
    label = row["emotion"]
    X_Index.append(image)
    Y_Index.append(label)
X_Index = numpy.array(X_Index)
Y_Index = numpy.array(Y_Index)

# Print each statement using colorama with reset to default text color
print(Fore.GREEN + "Data loaded successfully." + Style.RESET_ALL)
print(Fore.YELLOW + f"Total number of samples: {X_Index.shape[0]}" + Style.RESET_ALL)
print(Fore.YELLOW + f"Total number of features: {X_Index.shape[1]}" + Style.RESET_ALL)
print(Fore.YELLOW + f"Total number of labels: {Y_Index.shape[0]}" + Style.RESET_ALL)
print(Fore.CYAN + f"Number of epochs: {nEpochs}" + Style.RESET_ALL)
print(Fore.CYAN + f"Validation split: {nValsplit}" + Style.RESET_ALL)


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

    # Print hyperparameters used in the model
    print(Fore.BLUE + Style.BRIGHT + "Hyperparameters:")
    print(Fore.CYAN + f"filters_1: {hp.get('filters_1')}")
    print(Fore.CYAN + f"kernel_size_1: {hp.get('kernel_size_1')}")
    print(Fore.CYAN + f"pool_size_1: {hp.get('pool_size_1')}")
    for i in range(1, hp.get("nblocks") + 1):
        print(Fore.CYAN + f"filters_{i+1}: {hp.get('filters_' + str(i + 1))}")
        print(Fore.CYAN + f"kernel_size_{i+1}: {hp.get('kernel_size_' + str(i + 1))}")
        print(Fore.CYAN + f"pool_size_{i+1}: {hp.get('pool_size_' + str(i + 1))}")
    print(Fore.CYAN + f"units: {hp.get('units')}")
    print(Fore.CYAN + f"learning_rate: {hp.get('learning_rate')}")
    print(Style.RESET_ALL)

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

        # Print hyperparameters used in EarlyStopping
        print(Fore.BLUE + Style.BRIGHT + "Early Stopping:")
        print(Fore.CYAN + f"early_stopping_patience: {early_stopping_patience}")
        print(Style.RESET_ALL)

    return callbacks


HyperTuner = RandomSearch(
    build_model,
    max_trials=20,
    project_name="Emotion",
    objective="val_accuracy",
    directory="models/Face_Recog/Emotion",
)

HyperTuner.search(
    x=X_Index,
    y=Y_Index,
    verbose=1,
    batch_size=8,
    epochs=nEpochs,
    callbacks=Early_Stopping,
    validation_split=nValsplit,
)

BestHP = HyperTuner.get_best_hyperparameters(1)[0]
HyperModel = build_model(BestHP)
HyperModel.fit(X_Index, Y_Index, epochs=nEpochs, validation_split=0.2)
HyperModel.save("Face_Emotion_Model.h5")

# Print best hyperparameters
print(Fore.GREEN + Style.BRIGHT + "Best Hyperparameters:")
print(Fore.YELLOW + f"filters_1: {BestHP.get('filters_1')}")
print(Fore.YELLOW + f"kernel_size_1: {BestHP.get('kernel_size_1')}")
print(Fore.YELLOW + f"pool_size_1: {BestHP.get('pool_size_1')}")
print(Fore.YELLOW + f"nblocks: {BestHP.get('nblocks')}")
for i in range(BestHP.get("nblocks")):
    print(Fore.YELLOW + f"filters_{i+2}: {BestHP.get('filters_'+str(i+2))}")
    print(Fore.YELLOW + f"kernel_size_{i+2}: {BestHP.get('kernel_size_'+str(i+2))}")
    print(Fore.YELLOW + f"pool_size_{i+2}: {BestHP.get('pool_size_'+str(i+2))}")
print(Fore.YELLOW + f"units: {BestHP.get('units')}")
print(Fore.YELLOW + f"learning_rate: {BestHP.get('learning_rate')}")
print(Style.RESET_ALL)
