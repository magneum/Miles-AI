import keras_tuner
from tensorflow import keras
from keras.models import Sequential
from colorama import Fore as F, Style as S
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator


max_trials = 20
batch_size = 20
num_epochs = 200
target_size = (64, 64)
directory = "/kaggle/working/FaceEmo"
Test_dir = "/kaggle/input/fer2013/test"
Train_dir = "/kaggle/input/fer2013/train"
save_dir = "/kaggle/working/Fer_model.h5"
face_detection_model = "/kaggle/input/face-detection/tfjs/full/1"
face_landmarks_model = "/kaggle/input/face-landmarks-detection/tfjs/face-mesh/1"


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
Hyper_Tuner = keras_tuner.tuners.Hyperband(
    Hyper_Builder,
    max_epochs=50,  # set the maximum number of epochs
    seed=44,
    directory=directory,
    objective="val_accuracy",
    project_name="Fer_Emotion",
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
Hyper_Best.save(save_dir)
