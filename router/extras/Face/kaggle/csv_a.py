import os
import numpy
import pandas
from keras import Sequential
from colorama import Fore, Style
from keras.optimizers import Adam
from keras_tuner.tuners import Hyperband
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

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
model_save_path = "models/Face_Emo/Face_Emo_Model.h5"
dataset_path = "/kaggle/input/d/deadskull7/fer2013/fer2013.csv"

_path = "models/FaceEmo"
if not os.path.exists(_path):
    os.makedirs(_path)

print(f"{Fore.BLUE}Hyperparameters:{Style.RESET_ALL}")
print(f"nSeed: {Fore.GREEN}{nSeed}{Style.RESET_ALL}")
print(f"verbose: {Fore.GREEN}{verbose}{Style.RESET_ALL}")
print(f"patience: {Fore.GREEN}{patience}{Style.RESET_ALL}")
print(f"nEpochs: {Fore.GREEN}{nEpochs}{Style.RESET_ALL}")
print(f"nValsplit: {Fore.GREEN}{nValsplit}{Style.RESET_ALL}")
print(f"batch_size: {Fore.GREEN}{batch_size}{Style.RESET_ALL}")
print(f"hyper_directory: {Fore.GREEN}{hyper_directory}{Style.RESET_ALL}")
print(f"dataset_path: {Fore.GREEN}{dataset_path}{Style.RESET_ALL}")
print(f"model_save_path: {Fore.GREEN}{model_save_path}{Style.RESET_ALL}")

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
print(Style.RESET_ALL)
print(Fore.CYAN + "Total number of rows: " + Style.RESET_ALL + f"{Fer2013.shape[0]}")
print(Fore.CYAN + "Number of columns: " + Style.RESET_ALL + f"{Fer2013.shape[1]}")
print(Fore.CYAN + "Example of image data: " + Style.RESET_ALL)
print(X_Index[0]) 
print(Fore.CYAN + "Example of label: " + Style.RESET_ALL)
print(Y_Index[0]) 

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
    model.compile(optimizer=Adam(hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.add(EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))
    return model


# ========================================================= Magneum ========================================================= 
print(f"{Fore.GREEN}Create Hyperband tuner Completed!{Style.RESET_ALL}")
Hyper_Tuner = Hyperband(
    Hyper_Builder,
    seed=nSeed,
    max_epochs=nEpochs,
    project_name="Emotion",
    objective="val_accuracy",
    directory=hyper_directory,
)
# ========================================================= Magneum ========================================================= 
print(f"{Fore.GREEN}Define Hyperband search parameters Completed!{Style.RESET_ALL}")
Hyper_Tuner.search(
    x=X_Index,
    y=Y_Index,
    epochs=nEpochs,
    verbose=verbose,
    batch_size=batch_size,
    validation_split=nValsplit,
)

# ========================================================= Magneum ========================================================= 
print(f"{Fore.GREEN}Hyperband Search Completed!{Style.RESET_ALL}")
print(
    f"{Fore.CYAN}Best Hyperparameters: {Style.RESET_ALL}{Hyper_Tuner.get_best_hyperparameters()[0].values}"
)
print(
    f"{Fore.YELLOW}Best Model Architecture: {Style.RESET_ALL}{Hyper_Tuner.get_best_models()[0].summary()}"
)
print(
    f"{Fore.MAGENTA}Best Validation Accuracy: {Style.RESET_ALL}{Hyper_Tuner.get_best_models()[0].evaluate(X_Index, Y_Index)[1]}"
)

# ========================================================= Magneum ========================================================= 
BestHP = Hyper_Tuner.get_best_hyperparameters(1)[0]
Hyper_Model = Hyper_Builder(BestHP)
print(f"{Fore.GREEN}Best Hyperparameters: {BestHP}{Style.RESET_ALL}")
Hyper_Model.fit(X_Index, Y_Index, epochs=nEpochs, validation_split=0.2)
print(f"{Fore.BLUE}Training in progress...{Style.RESET_ALL}")
Hyper_Model.save(model_save_path)
print(f"{Fore.YELLOW}Model saved at: {model_save_path}{Style.RESET_ALL}")
