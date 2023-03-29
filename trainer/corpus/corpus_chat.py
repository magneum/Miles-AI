import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = load_model("trainer/corpus/corpus_model.h5")

# Load the data
data = pd.read_csv("corpdata/tsv/movie_conversations.tsv", sep="\t", header=None)
conversations = data.iloc[:, 3].values.tolist()

# Load the tokenizer
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(conversations)
tokenizer.word_index["<sos>"] = 5001  # Add "<sos>" and "<eos>" to the word index
tokenizer.word_index["<eos>"] = 5002

# Define the maximum length of input sequence
maxlen = 20

# Create a function to preprocess the question
def preprocess_question(question):
    # Convert the question to a sequence of integers
    question_seq = tokenizer.texts_to_sequences([question])
    # Pad the sequence to the defined maximum length
    question_seq_pad = pad_sequences(question_seq, padding="post", maxlen=maxlen)
    return question_seq_pad


# Create a function to generate a response
def generate_response(question):
    # Preprocess the question
    question_seq_pad = preprocess_question(question)
    # Predict the answer
    prediction = model.predict(question_seq_pad)
    # Convert the predicted answer sequence to text
    predicted_answer_seq = np.argmax(prediction, axis=1)
    predicted_answer = tokenizer.sequences_to_texts([predicted_answer_seq])[0]
    return predicted_answer


# Use the function to generate responses
while True:
    question = input("You: ")
    response = generate_response(question)
    print("Bot:", response)
