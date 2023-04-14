import os
import json
import time
import openai
import random
import spacy
from Agents import Agents
from dotenv import load_dotenv
from colorama import Fore, Style

load_dotenv()
objective = "Create a Python code that implements a vector database using numpy for the BabyAGI system, enabling efficient storage and retrieval of vector representations for data processing and inference."

openai.api_key = os.getenv("OPENAI_API_KEY", "")
nlp = spacy.load("en_core_web_sm")  # !python -m spacy download en_core_web_sm
num_iterations = 4
max_tokens = 2048
temperature = 0.5
wait_time = 30


def generate_response(agent, objective):
    agent_name = agent["name"]
    prompt = f"As the {agent_name}, my goal is to {agent['motive']}.\n\nTask: {agent_name}: {agent['prompt']}{objective}"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].text.strip()


def get_sentiment_score(text):
    doc = nlp(text)
    sentiment_score = {"compound": 0, "pos": 0, "neu": 0, "neg": 0}
    for sentence in doc.sents:
        sentiment = sentence.sentiment
        sentiment_score["compound"] += sentiment.compound
        sentiment_score["pos"] += sentiment.pos
        sentiment_score["neu"] += sentiment.neu
        sentiment_score["neg"] += sentiment.neg
    return sentiment_score


def get_named_entities(text):
    doc = nlp(text)
    named_entities = [(ent.label_, ent.text) for ent in doc.ents]
    return named_entities


data = []
for i in range(num_iterations):
    print(f"Iteration {i+1} of agent communication")
    for agent in Agents:
        try:
            response_text = generate_response(agent, objective)
            sentiment_score = get_sentiment_score(response_text)
            named_entities = get_named_entities(response_text)
        except Exception as e:
            response_text = str(e)
            sentiment_score = None
            named_entities = None
        agent_name = agent["name"]
        print(f"{Fore.CYAN}{agent_name}:{Style.RESET_ALL} {response_text}")
        data.append(
            {
                "agent": agent_name,
                "text": response_text,
                "sentiment_score": sentiment_score,
                "named_entities": named_entities,
            }
        )
        time.sleep(random.uniform(0.5, 1.5))
    if i < num_iterations - 1:
        print(f"Waiting for {wait_time} seconds before next iteration...")
        time.sleep(wait_time)

with open(f"{objective}_responses.json", "w") as f:
    json.dump(data, f)
print(
    f"Agent responses have been generated and saved to {objective}_responses.json file."
)

with open("final.txt", "w") as f:
    for agent in data:
        f.write(f"{agent['agent']}: {agent['text']}\n")
print("Final output has been saved to final.txt file.")
