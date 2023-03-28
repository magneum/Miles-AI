import json
import pickle
import spacy
import numpy as np
from colorama import init, Fore
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM

# Initialize colorama for colored print statements
init(autoreset=True)

# Load and preprocess the data
with open("app/trainer/data/intents.json") as file:
    data = json.load(file)

nlp = spacy.load("en_core_web_sm")  # python -m spacy download en_core_web_sm

training_sentences = []
training_labels = []
labels = []
responses = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        doc = nlp(pattern)
        for ent in doc.ents:
            pattern = pattern.replace(ent.text, ent.label_)
        training_sentences.append(pattern)
        training_labels.append(intent["tag"])
    responses.append(intent["responses"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

num_classes = len(labels)

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

vocab_size = 100
embedding_dim = 16
max_len = 20

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating="post", maxlen=max_len)

# Build the model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    LSTM(16, return_sequences=True),
    LSTM(16),
    Dense(num_classes, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.summary()

# Train the model
history = model.fit(
    padded_sequences,
    np.array(training_labels),
    batch_size=32,
    epochs=100,
    verbose=1,
    validation_split=0.2
)

# Save the model and preprocessors
model.save("chatbot_model")

with open("chatbot_model/tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("chatbot_model/label_encoder.pickle", "wb") as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

# Print training results
print(Fore.GREEN +
      f"Training complete. Trained on {len(training_sentences)} sentences, {num_classes} classes.")
print(Fore.GREEN + f"Model accuracy: {history.history['accuracy'][-1]:.4f}")
