import os
import numpy
import keras
import random
import pandas
from colorama import Fore, Style
from keras.callbacks import EarlyStopping
from keras_tuner import Hyperband, HyperParameters


# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
def CodeSeparator(section_name):
    separator_width = 50
    separator_char = "*"
    section_label = f" {section_name} "
    section_label_width = separator_width - 2
    section_label_padding = (section_label_width - len(section_label)) // 2
    separator_line = separator_char * separator_width
    available_colors = [
        Fore.RED,
        Fore.GREEN,
        Fore.YELLOW,
        Fore.BLUE,
        Fore.MAGENTA,
        Fore.CYAN,
        Fore.WHITE,
    ]
    random_color = random.choice(available_colors)
    section_label_line = (
        separator_char
        + " " * section_label_padding
        + f"{random_color}{Style.BRIGHT}{section_name}{Style.RESET_ALL}"
        + " " * section_label_padding
        + separator_char
    )

    print(separator_line)
    print(section_label_line)
    print(separator_line)


# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
print(f"{Fore.YELLOW}{Style.BRIGHT}Code Description: FaceProcessing.py")
print(f"{Fore.WHITE}{Style.BRIGHT}------------------")
print(
    f"{Fore.CYAN}{Style.BRIGHT}The code is a Python script that uses Keras and TensorFlow libraries to perform hyperparameter tuning for a convolutional neural network (CNN) model for facial emotion recognition."
)
print(
    f"{Fore.CYAN}{Style.BRIGHT}The dataset used is Fer2013, which is loaded from a CSV file."
)
print(
    f"\n{Fore.CYAN}{Style.BRIGHT}The script uses the keras_tuner library to perform random search hyperparameter tuning. The Hyper_Builder() function defines the architecture of the CNN model and compiles it with hyperparameters such as filters, kernel size, pool size, units, and learning rate. The hyperparameters are sampled from defined ranges using the hp object, which is passed as an argument to the function."
)
print(
    f"\n{Fore.CYAN}{Style.BRIGHT}The Early_Stopping() function defines early stopping as a callback for the model during training, which prevents overfitting. The patience for early stopping is also hyperparameterized using the hp object."
)
print(
    f"\n{Fore.CYAN}{Style.BRIGHT}The HyperBand tuner is then created with the Hyper_Builder() function as the model-building function, and the maximum number of trials, project name, and objective for tuning (in this case, 'val_accuracy') are specified. The tuner searches for the best hyperparameters using the search() function, which takes the input data (X_Index and Y_Index), number of epochs, batch size, and other parameters."
)
print(
    f"\n{Fore.CYAN}{Style.BRIGHT}After tuning is completed, the best hyperparameters and their corresponding model performance metrics (such as accuracy, loss) are printed. The best model is then trained with the optimal hyperparameters and evaluated on the validation set. Finally, the model is saved to disk for future use."
)
print(f"{Style.RESET_ALL}")

# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
X_Index = []
Y_Index = []
nEpochs = 100
nValsplit = 0.2
hyper_directory = "models/Face_Emo/Emotion"
dataset_path = "/kaggle/input/fer2013/fer2013.csv"
model_save_path = "models/Face_Emo/Face_Emotion_Model.h5"

# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
CodeSeparator("# Print loaded data information")
print(f"{Style.RESET_ALL}")
Fer2013 = pandas.read_csv(dataset_path)
print(f"{Fore.GREEN}{Style.BRIGHT}Loaded Data Information:")
print(f"{Fore.YELLOW}{Style.BRIGHT}• Data shape: {str(Fer2013.shape)}")
print(f"{Fore.YELLOW}{Style.BRIGHT}• Columns: {', '.join(Fer2013.columns)}")
print(f"{Style.RESET_ALL}")

# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
for index, row in Fer2013.iterrows():
    pixels = numpy.fromstring(row["pixels"], dtype="uint8", sep=" ")
    image = pixels.reshape((48, 48, 1)).astype("float32") / 255.0
    label = row["emotion"]
    X_Index.append(image)
    Y_Index.append(label)
X_Index = numpy.array(X_Index)
Y_Index = numpy.array(Y_Index)

CodeSeparator("# Print each statement using colorama with reset to default text color")
print(f"{Style.RESET_ALL}")
print(f"{Fore.GREEN}{Style.BRIGHT}Data loaded successfully.")
print(f"{Fore.YELLOW}{Style.BRIGHT}• Total number of samples: {X_Index.shape[0]}")
print(f"{Fore.YELLOW}{Style.BRIGHT}• Total number of features: {X_Index.shape[1]}")
print(f"{Fore.YELLOW}{Style.BRIGHT}• Total number of labels: {Y_Index.shape[0]}")
print(f"{Fore.CYAN}{Style.BRIGHT}• Number of epochs: {nEpochs}")
print(f"{Fore.CYAN}{Style.BRIGHT}• Validation split: {nValsplit}")
print(f"{Style.RESET_ALL}")


# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
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
    for i in range(1, hp.get("nblocks") + 1):
        print(
            f"{Fore.CYAN}{Style.BRIGHT}• filters_{i + 1}: {hp.get('filters_' + str(i + 1))}"
        )
        print(
            f"{Fore.CYAN}{Style.BRIGHT}• kernel_size_{i + 1}: {hp.get('kernel_size_' + str(i + 1))}"
        )
        print(
            f"{Fore.CYAN}{Style.BRIGHT}• pool_size_{i + 1}: {hp.get('pool_size_' + str(i + 1))}"
        )
    print(f"{Fore.BLUE}{Style.BRIGHT}Hyperparameters:")
    print(f"{Fore.CYAN}{Style.BRIGHT}• filters_1: {hp.get('filters_1')}")
    print(f"{Fore.CYAN}{Style.BRIGHT}• kernel_size_1: {hp.get('kernel_size_1')}")
    print(f"{Fore.CYAN}{Style.BRIGHT}• pool_size_1: {hp.get('pool_size_1')}")
    print(Fore.CYAN + Style.BRIGHT + "• units: " + str(hp.get("units")))
    print(Fore.CYAN + Style.BRIGHT + "• learning_rate: " + str(hp.get("learning_rate")))
    print(Style.RESET_ALL)
    return model


# ============================================================ [ CREATED BY MAGNEUM ] ============================================================

CodeSeparator("# Create HyperBand tuner")
print(Style.RESET_ALL)
hyperparameters = HyperParameters()
Hyper_Tuner = Hyperband(
    Hyper_Builder,
    objective="val_accuracy",
    max_epochs=20,
    hyperparameters=hyperparameters,
    directory=hyper_directory,
    project_name="Emotion",
)

CodeSeparator("# Start hyperparameter search")
print(Style.RESET_ALL)
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
# ============================================================ [ CREATED BY MAGNEUM ] ============================================================

CodeSeparator("# Print best hyperparameters")
print(Style.RESET_ALL)
print(f"{Fore.GREEN}{Style.BRIGHT}Best Hyperparameters:")
print(f"{Fore.YELLOW}{Style.BRIGHT}• filters_1: {BestHP.get('filters_1')}")
print(f"{Fore.YELLOW}{Style.BRIGHT}• kernel_size_1: {BestHP.get('kernel_size_1')}")
print(f"{Fore.YELLOW}{Style.BRIGHT}• pool_size_1: {BestHP.get('pool_size_1')}")
print(f"{Fore.YELLOW}{Style.BRIGHT}• nblocks: {BestHP.get('nblocks')}")
for i in range(BestHP.get("nblocks")):
    print(
        f"{Fore.YELLOW}{Style.BRIGHT}• filters_{i+2}: {BestHP.get('filters_'+str(i+2))}"
    )
    print(
        f"{Fore.YELLOW}{Style.BRIGHT}• kernel_size_{i+2}: {BestHP.get('kernel_size_'+str(i+2))}"
    )
    print(
        f"{Fore.YELLOW}{Style.BRIGHT}• pool_size_{i+2}: {BestHP.get('pool_size_'+str(i+2))}"
    )
    print(f"{Fore.YELLOW}{Style.BRIGHT}• units: {BestHP.get('units')}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}• learning_rate: {BestHP.get('learning_rate')}")
print(Style.RESET_ALL)
