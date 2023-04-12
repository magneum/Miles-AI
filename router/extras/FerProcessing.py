import os
from tensorflow import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import Sequential
from keras_tuner import HyperParameters
from keras_tuner.tuners import Hyperband
from colorama import Fore as F, Style as S
from keras.preprocessing.image import ImageDataGenerator


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
    model.add(keras.layers.Dropout(rate=0.25))
    for i in range(hp.Int("num_blocks", 1, 4, default=2)):
        model.add(
            keras.layers.Conv2D(
                filters=hp.Choice(f"filters_{i}", values=[32, 64, 128], default=32),
                kernel_size=(3, 3),
                activation="relu",
            )
        )
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(rate=0.25))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(
        keras.layers.Dense(
            units=hp.Int("units", min_value=128, max_value=512, step=64, default=256),
            activation="relu",
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=7, activation="softmax"))
    model.compile(
        optimizer=Adam(
            learning_rate=hp.Float(
                "learning_rate",
                min_value=1e-4,
                max_value=1e-2,
                default=1e-3,
                sampling="log",
            )
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# Define hp
hp = HyperParameters()
for i, layer in enumerate(Hyper_Builder(hp).layers):
    if "filters" in layer.get_config():
        print(F.YELLOW + f"Filters_{i}: {layer.get_config()['filters']}")

file_path = os.path.abspath(__file__)
file_name = os.path.basename(file_path)
print(S.RESET_ALL)
_path = "models/Face_Emo_Fer"
if not os.path.exists(_path):
    os.makedirs(_path)
    print(f"{F.GREEN}{S.BRIGHT}Folder created: {_path}{S.RESET_ALL}")
else:
    print(f"{F.YELLOW}{S.BRIGHT}Folder already exists: {_path}{S.RESET_ALL}")


num_epochs = 60
batch_size = 32
target_size = (64, 64)
Test_dir = "corpdata/Fer2013-img/Test_Images"
Train_dir = "corpdata/Fer2013-img/Train_Images"
Train_Datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)
Test_Datagen = ImageDataGenerator(rescale=1.0 / 255)

print(f"{F.YELLOW}{S.BRIGHT}Batch Size:{S.RESET_ALL} {batch_size}")
print(f"{F.YELLOW}{S.BRIGHT}Target Size:{S.RESET_ALL} {target_size}")
print(f"{F.YELLOW}{S.BRIGHT}Test Directory:{S.RESET_ALL} {Test_dir}")
print(f"{F.YELLOW}{S.BRIGHT}Train Directory:{S.RESET_ALL} {Train_dir}")
print(f"{F.YELLOW}{S.BRIGHT}Training Data Generator Settings:{S.RESET_ALL}")
print(f"{F.CYAN}{S.BRIGHT}Rescale:{S.RESET_ALL} {Train_Datagen.rescale}")
print(f"{F.CYAN}{S.BRIGHT}Shear Range:{S.RESET_ALL} {Train_Datagen.shear_range}")
print(f"{F.CYAN}{S.BRIGHT}Zoom Range:{S.RESET_ALL} {Train_Datagen.zoom_range}")
print(
    f"{F.CYAN}{S.BRIGHT}Horizontal Flip:{S.RESET_ALL} {Train_Datagen.horizontal_flip}"
)
print(f"{F.YELLOW}{S.BRIGHT}Test Data Generator Settings:{S.RESET_ALL}")
print(f"{F.CYAN}{S.BRIGHT}Rescale:{S.RESET_ALL} {Test_Datagen.rescale}")

# Define Train Data Generator
Train_Generator = ImageDataGenerator().flow_from_directory(
    Train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
)
print(f"{F.GREEN}{S.BRIGHT}Train Data Generator created successfully!{S.RESET_ALL}")
print(f"{F.CYAN}{S.BRIGHT}Train Directory: {Train_dir}{S.RESET_ALL}")
print(f"{F.CYAN}{S.BRIGHT}Target Size: {target_size}{S.RESET_ALL}")
print(f"{F.CYAN}{S.BRIGHT}Batch Size: {batch_size}{S.RESET_ALL}")
print(f"{F.CYAN}{S.BRIGHT}Class Mode: categorical{S.RESET_ALL}")

# Define Test Data Generator
Test_Generator = ImageDataGenerator().flow_from_directory(
    Test_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
)
print(f"{F.GREEN}{S.BRIGHT}Test Data Generator created successfully!{S.RESET_ALL}")
print(f"{F.YELLOW}{S.BRIGHT}Test Directory: {Test_dir}{S.RESET_ALL}")
print(f"{F.YELLOW}{S.BRIGHT}Target Size: {target_size}{S.RESET_ALL}")
print(f"{F.YELLOW}{S.BRIGHT}Batch Size: {batch_size}{S.RESET_ALL}")
print(f"{F.YELLOW}{S.BRIGHT}Class Mode: categorical{S.RESET_ALL}")

# Define Hyper_Tuner with Hyperband configuration
Hyper_Tuner = Hyperband(
    Hyper_Builder,
    seed=44,
    max_epochs=100,
    hyperband_iterations=22,
    project_name="Emotion",
    objective="val_accuracy",
    directory="models/Face_Emo_Fer",
)

print(f"{F.YELLOW}Starting hyperparameter search...{S.RESET_ALL}")
Hyper_Tuner.search(Train_Generator, epochs=num_epochs, validation_data=Test_Generator)
print(f"{F.GREEN}Hyperparameter search completed successfully!{S.RESET_ALL}")

Hyper_Best = Hyper_Tuner.get_best_models(num_models=1)[0]
Hyper_Best.fit(Train_Generator, epochs=num_epochs, validation_data=Test_Generator)
Evaluation = Hyper_Best.evaluate(Test_Generator)
Test_Loss, Test_Acc = Evaluation[0], Evaluation[1]
print(F.GREEN + f"Test Loss: {Test_Loss:.4f}" + S.RESET_ALL)
print(F.GREEN + f"Test Accuracy: {Test_Acc:.4f}" + S.RESET_ALL)
Hyper_Best.save("models/Face_Emo_Fer/Hyper_Best.h5")
