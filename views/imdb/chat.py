import tensorflow as tf
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.models import load_model

best_model = load_model("views/imdb/imdb_sentiment.h5")


def preprocess_input(text):
    word_index = imdb.get_word_index()
    text = text.lower()
    words = text.split()
    seq = [
        word_index[word] if word in word_index and word_index[word] < 5000 else 0
        for word in words
    ]
    return seq


def predict_sentiment(text):
    seq = preprocess_input(text)
    seq = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=500)
    prediction = best_model.predict(seq)
    if prediction[0][0] > prediction[0][1]:
        return "Negative"
    else:
        return "Positive"


while True:
    user_input = input("Enter a text to predict sentiment (q to quit): ")
    if user_input.lower() == "q":
        break
    sentiment = predict_sentiment(user_input)
    print("Sentiment: ", sentiment)
