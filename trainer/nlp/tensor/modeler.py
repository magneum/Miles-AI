# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical

# # Load the dataset
# data = pd.read_csv("data/csv/Tweets.csv")
# print("CSV Colums: ", data.columns)
# print("CSV Data: ", data)

# # Preprocess the data
# data["text"] = (
#     data["text"]
#     .str.lower()
#     .str.replace(r"\d+", "")
#     .str.replace(r"[^\w\s]", "")
#     .str.replace(r"\s+", " ")
# )

# # Convert the airline_sentiment labels to numeric values
# le = LabelEncoder()
# data["airline_sentiment"] = le.fit_transform(data["airline_sentiment"])

# # Split the data into training and validation sets
# train_data = data.sample(frac=0.8, random_state=42)
# val_data = data.drop(train_data.index)

# # Tokenize the text data
# tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
# tokenizer.fit_on_texts(train_data["text"])

# # Convert the text data to sequences
# train_sequences = tokenizer.texts_to_sequences(train_data["text"])
# val_sequences = tokenizer.texts_to_sequences(val_data["text"])

# # Pad the sequences
# train_padded = pad_sequences(
#     train_sequences, maxlen=128, truncating="post", padding="post"
# )
# val_padded = pad_sequences(val_sequences, maxlen=128, truncating="post", padding="post")

# # Convert the labels to one-hot encoded vectors
# num_classes = len(np.unique(data["airline_sentiment"]))
# train_labels = to_categorical(train_data["airline_sentiment"], num_classes=num_classes)
# val_labels = to_categorical(val_data["airline_sentiment"], num_classes=num_classes)

# # Define the model architecture
# model = tf.keras.Sequential(
#     [
#         tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=128),
#         tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
#         tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#         tf.keras.layers.Dense(64, activation="relu"),
#         tf.keras.layers.Dense(num_classes, activation="softmax"),
#     ]
# )

# # Compile the model
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# # Train the model
# history = model.fit(
#     train_padded,
#     train_labels,
#     validation_data=(val_padded, val_labels),
#     epochs=1000,
#     batch_size=32,
# )

# # Save the model
# model.save("my_chatbot_model.h5")
# print("Saved Model")


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = pd.read_csv("data/csv/Tweets.csv")

# Print the columns and data of the loaded dataset
print("CSV Colums: ", data.columns)
print("CSV Data: ", data)

# Preprocess the text data
data["text"] = (
    data["text"]
    .str.lower()  # Convert all characters to lowercase
    .str.replace(r"\d+", "")  # Remove all digits
    .str.replace(r"[^\w\s]", "")  # Remove all non-alphanumeric characters except spaces
    .str.replace(r"\s+", " ")  # Remove extra spaces
)

# Convert the airline_sentiment labels to numeric values
le = LabelEncoder()
data["airline_sentiment"] = le.fit_transform(data["airline_sentiment"])

# Split the data into training and validation sets
train_data = data.sample(
    frac=0.8, random_state=42
)  # Randomly sample 80% of data for training
val_data = data.drop(train_data.index)  # Use the remaining data for validation

# Tokenize the text data
tokenizer = Tokenizer(
    num_words=10000, oov_token="<OOV>"
)  # Tokenize only top 10k most frequent words, replace rare words with <OOV>
tokenizer.fit_on_texts(train_data["text"])  # Fit the tokenizer on the training data

# Convert the text data to sequences
train_sequences = tokenizer.texts_to_sequences(
    train_data["text"]
)  # Convert text data to sequences of integers
val_sequences = tokenizer.texts_to_sequences(val_data["text"])

# Pad the sequences to have uniform length
train_padded = pad_sequences(
    train_sequences, maxlen=128, truncating="post", padding="post"
)
val_padded = pad_sequences(val_sequences, maxlen=128, truncating="post", padding="post")

# Convert the labels to one-hot encoded vectors
num_classes = len(
    np.unique(data["airline_sentiment"])
)  # Get number of unique sentiment labels
train_labels = to_categorical(
    train_data["airline_sentiment"], num_classes=num_classes
)  # Convert sentiment labels to one-hot vectors
val_labels = to_categorical(val_data["airline_sentiment"], num_classes=num_classes)

# Define the model architecture
model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(
            input_dim=10000, output_dim=64, input_length=128
        ),  # Create an embedding layer to convert integer sequences to dense vectors
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        ),  # Add bidirectional LSTM layer
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32)
        ),  # Add another bidirectional LSTM layer
        tf.keras.layers.Dense(
            64, activation="relu"
        ),  # Add a dense layer with 64 neurons and ReLU activation
        tf.keras.layers.Dense(
            num_classes, activation="softmax"
        ),  # Add a dense layer with softmax activation for output
    ]
)

# Compile the model with categorical crossentropy loss and Adam optimizer
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(
    train_padded,
    train_labels,
    validation_data=(val_padded, val_labels),
    epochs=1000,
    batch_size=32,
)

# Save the trained model
model.save("my_chatbot_model.h5")
