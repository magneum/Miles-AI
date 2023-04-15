import numpy
import keras
import pandas
from colorama import Fore, Style
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras_tuner.tuners import RandomSearch, Hyperband


X_Index = []
Y_Index = []
nEpochs = 100
nValsplit = 0.2
hyper_directory = "models/Face_Emo/Emotion"
dataset_path = "/kaggle/input/fer2013/fer2013.csv"
ensemble_model_save_path = "models/Face_Emo/ensemble_model.h5"
model_save_path_hyperband = "models/Face_Emo/hyperband_model.h5"
model_save_path_random_search = "models/Face_Emo/random_search.h5"

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
    return model


# Instantiate Hyperband tuner
hyperband_tuner = Hyperband(
    Hyper_Builder,
    max_epochs=nEpochs,
    objective="val_accuracy",
    directory=hyper_directory,
    project_name="Emotion_Hyperband",
    seed=42,
)

# Instantiate RandomSearch tuner
random_search_tuner = RandomSearch(
    Hyper_Builder,
    max_trials=20,
    objective="val_accuracy",
    directory=hyper_directory,
    project_name="Emotion_RandomSearch",
    seed=42,
)

hyperband_tuner.search(
    x=X_Index,
    y=Y_Index,
    verbose=1,
    batch_size=8,
    epochs=nEpochs,
    validation_split=nValsplit,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ],
    factor=3,
    min_epochs=10,
)

random_search_tuner.search(
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


best_hyperparameters_hyperband = hyperband_tuner.get_best_hyperparameters(1)[0]
best_hyperparameters_random_search = random_search_tuner.get_best_hyperparameters(1)[0]

hyper_model_hyperband = Hyper_Builder(best_hyperparameters_hyperband)
hyper_model_hyperband.fit(X_Index, Y_Index, epochs=nEpochs, validation_split=0.2)
hyper_model_hyperband.save(model_save_path_hyperband)
print(Fore.GREEN + "HyperBand model saved successfully." + Style.RESET_ALL)

hyper_model_random_search = Hyper_Builder(best_hyperparameters_random_search)
hyper_model_random_search.fit(X_Index, Y_Index, epochs=nEpochs, validation_split=0.2)
hyper_model_random_search.save(model_save_path_random_search)
print(Fore.GREEN + "RandomSearch model saved successfully." + Style.RESET_ALL)

hyper_model_hyperband = load_model(model_save_path_hyperband)
hyper_model_random_search = load_model(model_save_path_random_search)

weights_hyperband = hyper_model_hyperband.get_weights()
weights_random_search = hyper_model_random_search.get_weights()

ensemble_weights = []
for weights1, weights2 in zip(weights_hyperband, weights_random_search):
    ensemble_weights.append((weights1 + weights2) / 2.0)

ensemble_model = Hyper_Builder(best_hyperparameters_hyperband)
ensemble_model.set_weights(ensemble_weights)

ensemble_model.save(ensemble_model_save_path)
print(Fore.GREEN + "Ensemble model saved successfully." + Style.RESET_ALL)
