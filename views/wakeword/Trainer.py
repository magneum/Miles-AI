import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.regularizers import l1, l2
from tensorflow.keras import Sequential
from keras.optimizers import Adam, RMSprop
from keras_tuner.tuners import RandomSearch
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized " + title

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


def build_model(hp):
    model = Sequential()

    model.add(
        Dense(
            units=hp.Int("units1", min_value=32, max_value=512, step=32),
            input_shape=X_train[0].shape,
            activation=hp.Choice("activation1", values=["relu", "tanh", "sigmoid"]),
        )
    )
    model.add(Dropout(hp.Float("dropout1", min_value=0.1, max_value=0.5, step=0.1)))

    num_layers = hp.Choice("num_layers", values=[1, 2, 3])
    for i in range(num_layers):
        model.add(
            Dense(
                units=hp.Int(
                    "units{}".format(i + 2), min_value=32, max_value=512, step=32
                ),
                activation=hp.Choice(
                    "activation{}".format(i + 2), values=["relu", "tanh", "sigmoid"]
                ),
                kernel_regularizer=l2(
                    hp.Float(
                        "l2_regularizer{}".format(i + 2),
                        min_value=0.001,
                        max_value=0.01,
                        step=0.001,
                    )
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


df = pd.read_pickle("models/wakeword/audio_data.csv")

X = np.concatenate(df["feature"].values, axis=0).reshape(len(df), 40)
y = to_categorical(LabelEncoder().fit_transform(df["class_label"].tolist()))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tuner = RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=5,
    executions_per_trial=3,
    directory="views/wakeword",
    project_name="wake_word_tuner",
)

tuner.search(X_train, y_train, epochs=1000, validation_split=0.2, verbose=2)
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

history = best_model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)

best_model.save("models/wakeword/wake_word.h5")
np.save("models/wakeword/wake_word_history.npy", history.history)

score = best_model.evaluate(X_test, y_test)
print(f"Score: {score}")

print("Model Classification Report: \n")
y_pred = np.argmax(best_model.predict(X_test), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(classification_report(np.argmax(y_test, axis=1), y_pred))
plot_confusion_matrix(cm, classes=["Does not have Wake Word", "Has Wake Word"])
