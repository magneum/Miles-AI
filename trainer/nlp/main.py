import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import random

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Load dataset using pandas
df = pd.read_csv("NLP/IMDB_Dataset.csv")

# Preprocess text data using a custom function
def preprocess_text(text):
    # remove HTML tags
    text = re.sub("<[^>]*>", "", text)
    # remove punctuation
    text = re.sub("[^\w\s]", "", text)
    # convert to lowercase
    text = text.lower()
    # remove stop words
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    # lemmatize words
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in filtered_text]
    return " ".join(lemmatized_text)


df["review"] = df["review"].apply(preprocess_text)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42
)

# Create a pipeline to vectorize text data and train a model
pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("model", LinearSVC())])

# Define hyperparameters for grid search
params = {
    "tfidf__max_df": [0.5, 0.75, 1.0],
    "tfidf__max_features": [None, 1000, 5000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "model__C": [0.1, 1, 10],
}

# Perform grid search to find best hyperparameters
grid_search = GridSearchCV(pipeline, params, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Define a function to generate a response based on user input
def generate_response(input_text):
    # Preprocess user input
    input_text = preprocess_text(input_text)
    # Make a prediction using the trained model
    prediction = grid_search.predict([input_text])[0]
    # Choose a response based on the prediction
    if prediction == "positive":
        response = random.choice(["I agree!", "That's great to hear!", "Awesome!"])
    else:
        response = random.choice(
            [
                "I'm sorry to hear that.",
                "That's too bad.",
                "Hopefully things will get better.",
            ]
        )
    return response


# Chat with the user
print("Hi, I'm a movie review chatbot. How can I help you?")
while True:
    input_text = input("> ")
    if input_text.lower() == "bye":
        print("Goodbye!")
        break
    else:
        response = generate_response(input_text)
        print(response)
