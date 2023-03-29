import os
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load the dataset
movie_lines = pd.read_csv(
    "corpdata/tsv/movie_lines.tsv", sep="\t", error_bad_lines=False
)
movie_lines.dropna(inplace=True)

# Create the input and target sequences
input_seqs = []
target_seqs = []
for i in range(1, len(movie_lines)):
    conversation = movie_lines.iloc[i - 1, 4] + " " + movie_lines.iloc[i, 4]
    input_seq = conversation.split(" ")[
        :-1
    ]  # input sequence is the conversation up to the last token
    target_seq = conversation.split(" ")[-1]  # target sequence is the last token
    input_seqs.append(input_seq)
    target_seqs.append(target_seq)

# Tokenize the conversations
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(input_seqs)

# Pad the input and target sequences
max_len = max(
    [len(seq) for seq in input_seqs]
)  # find the maximum length of the input sequences
input_seqs = tokenizer.texts_to_sequences(input_seqs)  # tokenize the input sequences
input_seqs = pad_sequences(
    input_seqs, maxlen=max_len, padding="pre"
)  # pad the input sequences to the maximum length
target_seqs = tokenizer.texts_to_sequences(target_seqs)  # tokenize the target sequences
target_seqs = to_categorical(
    target_seqs, num_classes=5000
)  # one-hot encode the target sequences

# Define the model architecture
input_seq = Input(shape=(max_len,))
embedding = Embedding(input_dim=5000, output_dim=50)(
    input_seq
)  # add an embedding layer
encoder = LSTM(50)(embedding)  # add an LSTM layer for encoding the input sequence
decoder = Dense(50, activation="relu")(
    encoder
)  # add a dense layer with ReLU activation
output_seq = Dense(5000, activation="softmax")(
    decoder
)  # add a dense layer with softmax activation for output
model = Model(inputs=input_seq, outputs=output_seq)  # create the model

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

os.system("clear")  # clear the terminal in case of linux
n_epochs = 1000  # set number of epochs
# Train the model
model.fit(input_seqs, target_seqs, epochs=n_epochs, verbose=1)

# Save the model
model.save("trainer/corpus/corpus_model.h5")
