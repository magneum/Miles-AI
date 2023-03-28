import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model

# Load the dataset
characters = pd.read_csv(
    "trainer/data/tsv/movie_characters_metadata.tsv",
    sep="\t",
    encoding="ISO-8859-1",
    usecols=[0, 1, 4],
    names=["id", "name", "movie_id"],
    header=None,
)
conversations = pd.read_csv(
    "trainer/data/tsv/movie_conversations.tsv",
    sep="\t",
    encoding="ISO-8859-1",
    usecols=[3],
    names=["lines"],
    header=None,
)
lines = pd.read_csv(
    "trainer/data/tsv/movie_lines.tsv",
    sep="\t",
    encoding="ISO-8859-1",
    usecols=[0, 4],
    names=["id", "text"],
    header=None,
)
movies = pd.read_csv(
    "trainer/data/tsv/movie_titles_metadata.tsv",
    sep="\t",
    encoding="ISO-8859-1",
    usecols=[0, 1, 2],
    names=["id", "title", "year"],
    header=None,
)

# Create a dictionary that maps each line to its ID
id_to_line = {}
for i, row in lines.iterrows():
    id_to_line[row["id"]] = row["text"]

# Create a dictionary that maps each movie ID to its title
id_to_title = {}
for i, row in movies.iterrows():
    id_to_title[row["id"]] = row["title"]

# Create a list of all conversations
conversations_ids = []
for i, row in conversations.iterrows():
    line_ids = row["lines"][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(line_ids.split(","))

# Separate the questions and answers
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id_to_line[conversation[i]])
        answers.append(id_to_line[conversation[i + 1]])

# Preprocess the data
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(questions + answers)
question_seq = tokenizer.texts_to_sequences(questions)
answer_seq = tokenizer.texts_to_sequences(answers)
question_seq_pad = pad_sequences(question_seq, padding="post", maxlen=20)
answer_seq_pad = pad_sequences(answer_seq, padding="post", maxlen=20)

# Create a TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((question_seq_pad, answer_seq_pad))
dataset = dataset.shuffle(buffer_size=len(questions))
dataset = dataset.batch(32)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Define the model architecture
input_seq = Input(shape=(20,))
embedding = Embedding(input_dim=5000, output_dim=50)(input_seq)
encoder = LSTM(50)(embedding)
decoder = Dense(50, activation="relu")(encoder)
output_seq = Dense(5000, activation="softmax")(decoder)
model = Model(inputs=input_seq, outputs=output_seq)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Train the model
model.fit(dataset, epochs=50)

# Save the model
model.save("trainer/corpus/corpus_model.h5")
