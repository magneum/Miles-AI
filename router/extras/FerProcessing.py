import os
import keras_tuner
from tensorflow import keras
from keras.models import Sequential
from colorama import Fore as F, Style as S
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator


# Hyper Variables
nSeed = 22
verbose = 1
nEpoch = 100
batch_size = 32
target_size = (64, 64)
Test_dir = "corpdata/Fer2013-img/Test_Images"
Train_dir = "corpdata/Fer2013-img/Train_Images"


def Hyper_Builder(hp):
    model = Sequential()
    model.add(
        keras.layers.Conv2D(
            filters=hp.Choice("filters", values=[32, 64], default=32),
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(64, 64, 3),
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(
        keras.layers.Dropout(
            rate=hp.Float("dropout_1", min_value=0.1, max_value=0.5, default=0.25)
        )
    )
    for i in range(hp.Int("num_blocks", 1, 4, default=2)):
        model.add(
            keras.layers.Conv2D(
                filters=hp.Choice(f"filters_{i}", values=[32, 64, 128], default=32),
                kernel_size=(3, 3),
                activation="relu",
            )
        )
        model.add(keras.layers.BatchNormalization())
        model.add(
            keras.layers.Dropout(
                rate=hp.Float(
                    f"dropout_{i+2}", min_value=0.1, max_value=0.5, default=0.25
                )
            )
        )
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(
        keras.layers.Dense(
            units=hp.Int("units", min_value=128, max_value=512, step=64, default=256),
            activation=hp.Choice(
                "dense_activation", values=["relu", "sigmoid"], default="relu"
            ),
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(
        keras.layers.Dropout(
            rate=hp.Float("dropout_6", min_value=0.1, max_value=0.5, default=0.5)
        )
    )
    model.add(keras.layers.Dense(units=7, activation="softmax"))
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

    return model


hp = keras_tuner.HyperParameters()
for i, layer in enumerate(Hyper_Builder(hp).layers):
    if "filters" in layer.get_config():
        print(F.YELLOW + f"Filters_{i}: {layer.get_config()['filters']}")

file_path = os.path.abspath(__file__)
file_name = os.path.basename(file_path)
print(S.RESET_ALL)
_path = "models/FaceEmo"
if not os.path.exists(_path):
    os.makedirs(_path)
    print(f"{F.GREEN}{S.BRIGHT}Folder created: {_path}{S.RESET_ALL}")
else:
    print(f"{F.YELLOW}{S.BRIGHT}Folder already exists: {_path}{S.RESET_ALL}")


Train_Datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)
Test_Datagen = ImageDataGenerator(rescale=1.0 / 255)


Train_Generator = ImageDataGenerator().flow_from_directory(
    Train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
)


Test_Generator = ImageDataGenerator().flow_from_directory(
    Test_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
)

Hyper_Tuner = keras_tuner.tuners.Hyperband(
    Hyper_Builder,
    seed=nSeed,
    max_epochs=nSeed,
    objective="val_accuracy",
    project_name="Fer_Emotion",
    directory="models/FaceEmo",
)

Hyper_Tuner.search(Train_Generator, epochs=nSeed, validation_data=Test_Generator)
Hyper_Best = Hyper_Tuner.get_best_models(num_models=1)[0]
Hyper_Best.fit(Train_Generator, epochs=nSeed, validation_data=Test_Generator)
Evaluation = Hyper_Best.evaluate(Test_Generator)
Test_Loss, Test_Acc = Evaluation[0], Evaluation[1]
Hyper_Best.save("models/FaceEmo/Fer_model.h5")
