import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = load_model("trainer/corpus/corpus_model.h5")

# Load the tokenizer
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(["<sos>", "<eos>"])


# Create a function to generate a response
def generate_response(question):
    # Preprocess the question
    question_seq = tokenizer.texts_to_sequences([question])
    question_seq_pad = pad_sequences(
        question_seq,
        padding="post",
        maxlen=20,  # Define the maximum length of input sequence
    )

    # Predict the answer
    prediction = model.predict(question_seq_pad)
    predicted_answer_seq = np.argmax(prediction, axis=1)
    predicted_answer = tokenizer.sequences_to_texts([predicted_answer_seq])[0]

    return predicted_answer


# Use the function to generate responses
while True:
    question = input("You: ")
    response = generate_response(question)
    print("Bot:", response)
