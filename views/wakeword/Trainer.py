import os
import pickle
import tensorflow
import numpy as np
import pandas as pd
from keras import regularizers
import matplotlib.pyplot as plt
from colorama import Fore, Style
from tensorflow.keras import Sequential
from keras.optimizers import Adam, RMSprop
from keras_tuner.tuners import RandomSearch
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Paired
):
    plt.figure(figsize=(8, 6))
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized " + title

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right", fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=14,
            fontweight="bold",
        )

    plt.ylabel("True label", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted label", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def modelBuilder(hp):
    model = Sequential()

    model.add(
        Dense(
            units=hp.Int("units1", min_value=32, max_value=512, step=32),
            input_shape=X_train[0].shape,
            activation=hp.Choice("activation1", values=["relu", "tanh", "sigmoid"]),
            kernel_regularizer=regularizers.l1_l2(
                l1=hp.Float(
                    "l1_regularizer1", min_value=0.001, max_value=0.01, step=0.001
                ),
                l2=hp.Float(
                    "l2_regularizer1", min_value=0.001, max_value=0.01, step=0.001
                ),
            ),
        )
    )
    model.add(Dropout(hp.Float("dropout1", min_value=0.1, max_value=0.5, step=0.1)))
    num_layers = hp.Int("num_layers", min_value=1, max_value=4)
    for i in range(num_layers):
        model.add(
            Dense(
                units=hp.Int(
                    "units{}".format(i + 2), min_value=32, max_value=512, step=32
                ),
                activation=hp.Choice(
                    "activation{}".format(i + 2), values=["relu", "tanh", "sigmoid"]
                ),
                kernel_regularizer=regularizers.l1_l2(
                    l1=hp.Float(
                        "l1_regularizer{}".format(i + 2),
                        min_value=0.001,
                        max_value=0.01,
                        step=0.001,
                    ),
                    l2=hp.Float(
                        "l2_regularizer{}".format(i + 2),
                        min_value=0.001,
                        max_value=0.01,
                        step=0.001,
                    ),
                ),
            )
        )
        model.add(
            Dropout(
                hp.Float(
                    "dropout{}".format(i + 2), min_value=0.1, max_value=0.5, step=0.1
                )
            )
        )

    model.add(Dense(len(np.unique(df["class_label"].tolist())), activation="softmax"))
    optimizer = hp.Choice("optimizer", values=["adam", "rmsprop"])
    learning_rate = hp.Float(
        "learning_rate", min_value=0.001, max_value=0.01, step=0.001
    )
    if optimizer == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)

    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model


num_epochs = 10

df = pd.read_pickle("models/wakeword/wakeword_data.csv")
print(f"Dataframe: {df}")

X = np.concatenate(df["feature"].values, axis=0).reshape(len(df), 40)
y = to_categorical(LabelEncoder().fit_transform(df["class_label"].tolist()))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"{Fore.GREEN}Train data:{Style.RESET_ALL}")
print(f"X_train shape: {Fore.CYAN}{X_train.shape}{Style.RESET_ALL}")
print(f"y_train shape: {Fore.CYAN}{y_train.shape}{Style.RESET_ALL}")
print(f"{Fore.GREEN}Test data:{Style.RESET_ALL}")
print(f"X_test shape: {Fore.CYAN}{X_test.shape}{Style.RESET_ALL}")
print(f"y_test shape: {Fore.CYAN}{y_test.shape}{Style.RESET_ALL}")


# Create the RandomSearch tuner
hyperTuner = RandomSearch(
    modelBuilder,
    max_trials=5,
    overwrite=True,
    executions_per_trial=3,
    objective="val_accuracy",
    project_name="hyperModel",
    directory="models/wakeword",
)

# Search for best hyperparameters
hyperTuner.search(
    X_train,
    y_train,
    verbose=1,
    epochs=num_epochs,
    validation_split=0.2,
)

# Get the best model from the tuner
hyperModel = hyperTuner.get_best_models(num_models=1)[0]

# Fit the best model
hyperModel.fit(
    X_train,
    y_train,
    epochs=num_epochs,
    validation_split=0.2,
    verbose=1,
)


hyperHistory = hyperModel.fit(
    X_train, y_train, epochs=num_epochs, validation_split=0.2, verbose=1
)

print(f"{Fore.CYAN}HyperModel Summary:{Style.RESET_ALL}")
hyperModel.summary()
np.save("models/wakeword/hyperModel/WakeWordHistory.npy", hyperHistory.history)
hyperModel.save("models/wakeword/hyperModel")
score = hyperModel.evaluate(X_test, y_test)
print(f"{Fore.YELLOW}Score: {score}{Style.RESET_ALL}")
print(f"{Fore.GREEN}Model Classification Report:{Style.RESET_ALL}")
y_pred = np.argmax(hyperModel.predict(X_test), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(classification_report(np.argmax(y_test, axis=1), y_pred))
plot_confusion_matrix(cm, classes=["No Wake Word", "Has Wake Word"])
