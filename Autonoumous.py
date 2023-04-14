import os
import json
import time
import openai
import random
from Agents import Agents
from dotenv import load_dotenv
from colorama import Fore, Style

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY", "")
objective = "Create a Python code that implements a vector database using numpy for the BabyAGI system, enabling efficient storage and retrieval of vector representations for data processing and inference."
num_iterations = 8
max_tokens = 2048
temperature = 1
wait_time = 30


class AgentTask:
    def __init__(self, agent):
        self.agent = agent
        self.status = "ongoing"
        self.response_text = None
        self.response_embedding = None

    def generate_response(self, objective):
        agent_name = self.agent["name"]
        prompt = f"As the {agent_name}, my goal is to {self.agent['motive']}.\n\nTask: {agent_name}: {self.agent['prompt']}{objective}"
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.response_text = response.choices[0].text.strip()
        self.response_embedding = ada_embedding(self.response_text)
        return self.response_text

    def mark_task_as_finished(self):
        self.status = "finished"

    def mark_task_as_ended(self):
        self.status = "ended"


def ada_embedding(text):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    embedding = response["data"][0]["embedding"]
    return embedding


# Create task objects for each agent
task_list = []
for agent in Agents:
    task = AgentTask(agent)
    task_list.append(task)

data = []
for i in range(num_iterations):
    print(f"Iteration {i+1} of agent communication")
    for task in task_list:
        if task.status == "ongoing":
            try:
                response_text = task.generate_response(objective)
            except Exception as e:
                response_text = str(e)
                task.mark_task_as_ended()
            agent_name = task.agent["name"]
            print(f"{Fore.CYAN}{agent_name}:{Style.RESET_ALL} {response_text}")
            data.append(
                {
                    "agent": agent_name,
                    "text": response_text,
                    "embedding": task.response_embedding,
                }
            )
            if i == num_iterations - 1:
                task.mark_task_as_finished()
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
