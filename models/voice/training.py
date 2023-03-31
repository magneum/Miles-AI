from plot_cm import plot_confusion_matrix
import os
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Activation, Dropout
from sklearn.metrics import confusion_matrix, classification_report
# =============================================================================================================
df = pd.read_pickle("models/voice/audio_data.csv")

# Making our data training-ready
X = df["feature"].values
X = np.concatenate(X, axis=0).reshape(len(X), 40)

y = np.array(df["class_label"].tolist())
le = LabelEncoder()
y = to_categorical(le.fit_transform(y))

num_classes = len(le.classes_)
# =============================================================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# =============================================================================================================
model = Sequential([
    Dense(256, input_shape=X_train[0].shape),
    Activation("relu"),
    Dropout(0.5),
    Dense(256),
    Activation("relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

print(model.summary())

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print("Model Score: \n")
history = model.fit(
    X_train, y_train, epochs=1000, validation_split=0.2, verbose=2
)
model.save("public/audio/wake_word.h5")
np.save("public/audio/wake_word_history.npy", history.history)
score = model.evaluate(X_test, y_test)
print(score)

#### Evaluating our model ###########
print("Model Classification Report: \n")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(classification_report(np.argmax(y_test, axis=1), y_pred))
plot_confusion_matrix(cm, classes=["Does not have Wake Word", "Has Wake Word"])
# =============================================================================================================
