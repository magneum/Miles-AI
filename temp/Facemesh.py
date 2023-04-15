import os
import cv2
import numpy as np
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from kerastuner.tuners import Hyperband, RandomSearch
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

dataset_path = "path/to/wflw/dataset"
ensemble_model_path = "ensemble_model.h5"
images_path = os.path.join(dataset_path, "images")
annotations_path = os.path.join(dataset_path, "annotations")

x_train = []
y_train = []
for image_file in os.listdir(images_path):
    image_path = os.path.join(images_path, image_file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    x_train.append(image)
    annotation_file = image_file.replace(".jpg", ".txt")
    annotation_path = os.path.join(annotations_path, annotation_file)
    keypoints = np.loadtxt(annotation_path)
    y_train.append(keypoints)

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = x_train / 255.0
y_train = y_train / 224.0
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)


def build_model(hp):
    model = Sequential()
    model.add(
        Conv2D(
            filters=hp.Int("filters", min_value=32, max_value=256, step=32),
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(224, 224, 3),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    for i in range(hp.Int("num_conv_layers", min_value=0, max_value=3)):
        model.add(
            Conv2D(
                filters=hp.Int(
                    "filters_" + str(i), min_value=32, max_value=256, step=32
                ),
                kernel_size=(3, 3),
                activation="relu",
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(
        Dense(
            units=hp.Int("units", min_value=32, max_value=256, step=32),
            activation="relu",
        )
    )
    model.add(Dense(units=136))
    model.compile(optimizer="adam", loss="mse")
    return model


hyperband_tuner = Hyperband(
    build_model,
    max_epochs=10,
    objective="val_loss",
    hyperband_iterations=2,
    directory="hyperband_tuner_dir",
    project_name="my_hyperband_tuner",
)

random_search_tuner = RandomSearch(
    build_model,
    max_trials=10,
    objective="val_loss",
    directory="random_search_tuner_dir",
    project_name="my_random_search_tuner",
)

hyperband_tuner.search_space_summary()
random_search_tuner.search_space_summary()

hyperband_tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
random_search_tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
best_hyperband_model = hyperband_tuner.get_best_models(num_models=1)[0]
best_random_model = random_search_tuner.get_best_models(num_models=1)[0]
ensemble_model = Sequential()
ensemble_model.add(best_hyperband_model)
ensemble_model.add(best_random_model)
ensemble_model.compile(optimizer="adam", loss="mse")
ensemble_model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
ensemble_model.evaluate(x_val, y_val)
ensemble_model.save(ensemble_model_path)
