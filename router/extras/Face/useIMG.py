import os
from colorama import Fore, Style
from keras.models import Sequential
from keras_tuner.tuners import Hyperband
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (
    GlobalAveragePooling2D,
    BatchNormalization,
    MaxPooling2D,
    Dropout,
    Conv2D,
    Dense,
)

# ========================================================= Magneum
nSeed = 22
verbose = 1
patience = 10
nEpochs = 200
nValsplit = 0.2
batch_size = 12
target_size = (64, 64)
hyper_directory = "models/Face_Emo/Emotion"
Test_dir = "corpdata/Fer2013-img/Test_Images"
Train_dir = "corpdata/Fer2013-img/Train_Images"
print(Fore.GREEN + "Hyperparameters:")
print("nSeed =", nSeed)
print("verbose =", verbose)
print("patience =", patience)
print("nEpochs =", nEpochs)
print("nValsplit =", nValsplit)
print("batch_size =", batch_size)
print("target_size =", target_size)
print("hyper_directory =", hyper_directory)
print("Test_dir =", Test_dir)
print("Train_dir =", Train_dir)
print(Style.RESET_ALL)


# ========================================================= Magneum
def Hyper_Builder(hp):
    model = Sequential()
    model.add(
        Conv2D(
            filters=hp.Choice("filters", values=[32, 64], default=32),
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(64, 64, 3),
        )
    )
    model.add(BatchNormalization())
    model.add(
        Dropout(rate=hp.Float("dropout_1", min_value=0.1, max_value=0.5, default=0.25))
    )
    for i in range(hp.Int("num_blocks", 1, 4, default=2)):
        model.add(
            Conv2D(
                filters=hp.Choice(f"filters_{i}", values=[32, 64, 128], default=32),
                kernel_size=(3, 3),
                activation="relu",
            )
        )
        model.add(BatchNormalization())
        model.add(
            Dropout(
                rate=hp.Float(
                    f"dropout_{i+2}", min_value=0.1, max_value=0.5, default=0.25
                )
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(
        Dense(
            units=hp.Int("units", min_value=128, max_value=512, step=64, default=256),
            activation=hp.Choice(
                "dense_activation", values=["relu", "sigmoid"], default="relu"
            ),
        )
    )
    model.add(BatchNormalization())
    model.add(
        Dropout(rate=hp.Float("dropout_6", min_value=0.1, max_value=0.5, default=0.5))
    )
    model.add(Dense(units=7, activation="softmax"))
    optimizer = hp.Choice("optimizer", values=["adam", "rmsprop"], default="adam")
    if optimizer == "adam":
        model_optimizer = Adam(
            learning_rate=hp.Float(
                "learning_rate",
                min_value=1e-5,
                max_value=1e-3,
                default=1e-4,
                sampling="log",
            )
        )
    else:
        model_optimizer = RMSprop(
            learning_rate=hp.Float(
                "learning_rate",
                min_value=1e-5,
                max_value=1e-3,
                default=1e-4,
                sampling="log",
            )
        )
    model.compile(
        optimizer=model_optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.add(EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))
    return model


# ========================================================= Magneum
File_path = os.path.abspath(__file__)
File_name = os.path.basename(File_path)
_path = "models/FaceEmo"
if not os.path.exists(_path):
    os.makedirs(_path)
print(f"{Fore.GREEN}File Path: {File_path}")
print(f"{Fore.YELLOW}File Name: {File_name}")
print(f"{Fore.CYAN}Path: {_path}")
print(Style.RESET_ALL)


# ========================================================= Magneum
Train_Datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)
print(Fore.GREEN + "Training Data Generator:")
print(Style.RESET_ALL)
print(Train_Datagen)

Test_Datagen = ImageDataGenerator(rescale=1.0 / 255)
print(Fore.YELLOW + "Test Data Generator:")
print(Style.RESET_ALL)
print(Test_Datagen)


# ========================================================= Magneum
Train_Generator = ImageDataGenerator().flow_from_directory(
    Train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
)
print(Fore.GREEN + "Train Generator Information:")
print(f"Directory: {Train_dir}")
print(f"Target Size: {target_size}")
print(f"Batch Size: {batch_size}")
print(f"Class Mode: categorical")
print(Style.RESET_ALL)

Test_Generator = ImageDataGenerator().flow_from_directory(
    Test_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
)
print(Fore.YELLOW + "Test Generator Information:")
print(f"Directory: {Test_dir}")
print(f"Target Size: {target_size}")
print(f"Batch Size: {batch_size}")
print(f"Class Mode: categorical")
print(Style.RESET_ALL)


# ========================================================= Magneum
print(Fore.GREEN + "Starting Hyperparameter Tuning..." )
print(
    Fore.YELLOW + "Using Hyper_Builder with seed =",
    nSeed,
    "and max_epochs =",
    nEpochs ,
)
print(Fore.CYAN + "Objective: val_accuracy" )
print(Fore.MAGENTA + "Project Name: Fer_Emotion" )
print(Fore.BLUE + "Directory: " + hyper_directory )
print(Style.RESET_ALL)
Hyper_Tuner = Hyperband(
    Hyper_Builder,
    seed=nSeed,
    max_epochs=nEpochs,
    objective="val_accuracy",
    project_name="Fer_Emotion",
    directory=hyper_directory,
)

Hyper_Tuner.search(
    Train_Generator,
    epochs=nEpochs,
    verbose=verbose,
    batch_size=batch_size,
    validation_data=Test_Generator,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    ],
)
print(Fore.GREEN + "Hyperparameter tuning completed successfully!" )
print(Fore.YELLOW + "Best hyperparameters found:" )
print(Style.RESET_ALL)
print(Hyper_Tuner.get_best_hyperparameters()[0].values)


# ========================================================= Magneum
Hyper_Best = Hyper_Tuner.get_best_models(num_models=1)[0]
Hyper_Best.fit(Train_Generator, epochs=nSeed, validation_data=Test_Generator)
Evaluation = Hyper_Best.evaluate(Test_Generator)
Test_Loss, Test_Acc = Evaluation[0], Evaluation[1]
Hyper_Best.save("models/FaceEmo/Fer_model.h5")
print(Fore.GREEN + "Training completed successfully!" )
print(
    Fore.CYAN
    + f"Test Loss: {Test_Loss:.4f}, Test Accuracy: {Test_Acc:.4f}"
    
)
print(
    Fore.YELLOW + "Best model saved as 'models/FaceEmo/Fer_model.h5'" 
)
print(Style.RESET_ALL)