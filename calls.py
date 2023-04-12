from sentence_transformers import SentenceTransformer
import openai
import numpy as np
from scipy.spatial.distance import cosine

openai.api_key = "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"
model_name = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"
embedder = SentenceTransformer(model_name)
vectors = []
texts = []


def generate_and_store_response(prompt):
    try:
        response = openai.Completion.create(
            prompt=prompt, max_tokens=256, n=1, stop=None, temperature=0.7
        )
        text = response["choices"][0]["text"].strip()
        embedding = embedder.encode([text])[0]
        vectors.append(embedding)
        texts.append(text)
        return text
    except Exception as e:
        print("Error generating response:", e)
        return None


def search_similar_responses(query):
    try:
        query_embedding = embedder.encode([query])[0]
        similarities = [1 - cosine(query_embedding, vector) for vector in vectors]
        sorted_indices = np.argsort(similarities)[::-1]  # Sort in descending order
        similar_responses = [texts[index] for index in sorted_indices]
        return similar_responses
    except Exception as e:
        print("Error searching for similar responses:", e)
        return None


while True:
    prompt = input("Enter a prompt or type 'quit' to exit: ")
    if prompt == "quit":
        break
    response = generate_and_store_response(prompt)
    if response:
        print("Generated response:", response)

query = input("Enter a query to search for similar responses: ")
if query:
    similar_responses = search_similar_responses(query)
    if similar_responses:
        print("Similar responses:", similar_responses)
