import os
import numpy
import pandas
from keras import Sequential
from colorama import Fore, Style
from keras.optimizers import Adam
from keras_tuner.tuners import Hyperband
from keras.callbacks import EarlyStopping
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)

# ========================================================= Magneum =========================================================
X_Index = []
Y_Index = []
verbose = 1
patience = 10
num_seeds = 22
batch_size = 12
num_epochs = 200
num_valsplit = 0.2
Hyperband_factor = 2
executions_per_trial = 1
Hyperband_overwrite = True
dir_path = "models/FacialEmotion"
Hyperband_project_name = "Emotion"
Hyperband_objective = "val_accuracy"
hyper_directory = "models/FacialEmotion/Emotion"
dataset_path = "corpdata/csv/fer2013/fer2013.csv"
best_model_save_path = "models/FacialEmotion/models"

if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"{Fore.GREEN}Directory created: {dir_path}")
else:
    print(f"{Fore.YELLOW}Directory already exists: {dir_path}")

print(Fore.BLUE + "Hyperparameters:" + Style.RESET_ALL)
print("num_seeds: " + Fore.GREEN + str(num_seeds))
print("verbose: " + Fore.GREEN + str(verbose))
print("patience: " + Fore.GREEN + str(patience))
print("num_epochs: " + Fore.GREEN + str(num_epochs))
print("num_valsplit: " + Fore.GREEN + str(num_valsplit))
print("batch_size: " + Fore.GREEN + str(batch_size))
print("hyper_directory: " + Fore.GREEN + hyper_directory)
print("dataset_path: " + Fore.GREEN + dataset_path)
print("best_model_save_path: " + Fore.GREEN + best_model_save_path)
print("Hyperband_factor: " + Fore.GREEN + str(Hyperband_factor))
print("executions_per_trial: " + Fore.GREEN + str(executions_per_trial))
print("Hyperband_overwrite: " + Fore.GREEN + str(Hyperband_overwrite))
print("dir_path: " + Fore.GREEN + dir_path + Style.RESET_ALL)
print("Hyperband_project_name: " + Fore.GREEN + Hyperband_project_name)
print("Hyperband_objective: " + Fore.GREEN + Hyperband_objective)
print("hyper_directory: " + Fore.GREEN + hyper_directory + Style.RESET_ALL)
print("dataset_path: " + Fore.GREEN + dataset_path)
print("best_model_save_path: " + Fore.GREEN + best_model_save_path)
print(Style.RESET_ALL)

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

print(Fore.YELLOW + "Information from Fer2013 dataset:")
print(Fore.CYAN + "Total number of rows: " + Style.RESET_ALL + f"{Fer2013.shape[0]}")
print(Fore.CYAN + "Number of columns: " + Style.RESET_ALL + f"{Fer2013.shape[1]}")
print(Style.RESET_ALL)

# ========================================================= Magneum =========================================================
def Hyper_Builder(hp):
    model = Sequential()
    model.add(Conv2D(filters=hp.Int("filters_1", 32, 128, step=32), kernel_size=hp.Choice("kernel_size_1", values=[3, 5]), activation="relu", input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=hp.Choice("pool_size_1", values=[4, 5])))
    nblocks = hp.Int("nblocks", 1, 8)
    for i in range(nblocks):
        model.add(Conv2D(filters=hp.Int("filters_" + str(i + 2), 32, 128, step=32), kernel_size=hp.Choice("kernel_size_" + str(i + 2), values=[3, 5]), activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=hp.Choice("pool_size_" + str(i + 2), values=[4, 5]), padding="same"))
    model.add(Flatten())
    model.add(Dense(units=hp.Int("units", 128, 512, step=32), activation="relu"))
    model.add(Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1)))
    nlayers = hp.Int("nlayers", 0, 6)
    for j in range(nlayers):
        model.add(Dense(units=hp.Int(f"units_{j + 2}", 64, 256, step=32), activation="relu"))
    model.add(Dense(7, activation="softmax"))
    model.compile(optimizer=Adam(hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ========================================================= Magneum =========================================================


print(Fore.GREEN + "Create Hyperband tuner Completed!" + Style.RESET_ALL)
Hyper_Tuner = Hyperband(
    allow_new_entries=True,  # Whether to allow new entries in hyperparameters dictionary
    directory=hyper_directory,  # Directory to store Hyperband results
    executions_per_trial=executions_per_trial,  # Number of times to train a model with the same hyperparameters
    factor=Hyperband_factor,  # Reduction factor for the number of configurations
    hyperband_iterations=1,  # Number of brackets for Successive Halving
    hypermodel=Hyper_Builder,  # Hypermodel to be tuned
    max_epochs=num_epochs,  # Maximum number of epochs to run
    objective=Hyperband_objective,  # Objective to optimize
    overwrite=Hyperband_overwrite,  # Whether to overwrite existing Hyperband results
    project_name=Hyperband_project_name,  # Name of the project
    seed=num_seeds,  # Seed for random number generation
)
Hyper_Tuner.search(
    x=X_Index,  # Input data (X) for tuning
    y=Y_Index,  # Target data (y) for tuning
    verbose=verbose,  # Verbosity level for logging (0: silent, 1: progress bar, 2: one line per epoch)
    batch_size=batch_size,  # Batch size for training
    validation_split=num_valsplit,  # Fraction of data to use for validation during training
    callbacks=[  # List of callbacks to use during training
        EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )  # Early stopping callback to stop training early if validation loss does not improve for a certain number of epochs (patience) and restore the best weights of the model
    ],
)
print(Fore.GREEN + "Defining Hyperband search parameters Completed!" + Style.RESET_ALL)


# ========================================================= Magneum =========================================================
print(Fore.GREEN + "Hyperband Search Completed!" + Style.RESET_ALL)
print(
    Fore.CYAN
    + "Best Hyperparameters: "
    + Style.RESET_ALL
    + str(Hyper_Tuner.get_best_hyperparameters()[0].values)
)
print(
    Fore.YELLOW
    + "Best Model Architecture: "
    + Style.RESET_ALL
    + Hyper_Tuner.get_best_models()[0].summary()
)
print(
    Fore.MAGENTA
    + "Best Validation Accuracy: "
    + Style.RESET_ALL
    + str(Hyper_Tuner.get_best_models()[0].evaluate(X_Index, Y_Index)[1])
)

# ========================================================= Magneum =========================================================
BestHP = Hyper_Tuner.get_best_hyperparameters(1)[0]
Hyper_Model = Hyper_Builder(BestHP)
print(Fore.GREEN + "Best Hyperparameters: " + str(BestHP) + Style.RESET_ALL)

# Save best model
Hyper_Model.save(best_model_save_path)
print(Fore.YELLOW + "Best Model saved at: " + best_model_save_path + Style.RESET_ALL)

# Train and save all models
for i, hyperparams in enumerate(Hyper_Tuner.get_best_hyperparameters()):
    model = Hyper_Builder(hyperparams)
    print(
        Fore.GREEN
        + "Hyperparameters for Model "
        + str(i + 1)
        + ": "
        + str(hyperparams)
        + Style.RESET_ALL
    )
    model.fit(X_Index, Y_Index, epochs=num_epochs, validation_split=0.2)
    model_save_path_i = best_model_save_path + "_model_" + str(i + 1)
    model.save(model_save_path_i)
    print(
        Fore.YELLOW
        + "Model "
        + str(i + 1)
        + " saved at: "
        + model_save_path_i
        + Style.RESET_ALL
    )

print(Fore.BLUE + "Training in progress..." + Style.RESET_ALL)
