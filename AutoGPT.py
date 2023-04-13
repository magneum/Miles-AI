import os
import time
import openai
import pinecone
import importlib
import subprocess
from typing import Dict, List
from collections import deque
from dotenv import load_dotenv
from colorama import Fore, Style

load_dotenv()

INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv("FIRST_TASK", ""))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
PINECONE_API = os.getenv("PINECONE_API", "")
PINECONE_ENV = os.getenv("PINECONE_ENV", "")
OPENAI_API = os.getenv("OPENAI_API", "")
TABLE_NAME = os.getenv("TABLE_NAME", "")
OBJECTIVE = os.getenv("OBJECTIVE", "")


assert OPENAI_API, "OPENAI_API environment variable is missing from .env"
assert TABLE_NAME, "TABLE_NAME environment variable is missing from .env"
assert PINECONE_ENV, "PINECONE_ENV environment variable is missing from .env"
assert PINECONE_API, "PINECONE_API environment variable is missing from .env"
assert OPENAI_MODEL, "OPENAI_MODEL environment variable is missing from .env"


def can_import(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


if "gpt-4" in OPENAI_MODEL.lower():
    print(
        f"{Fore.RED}{Style.BRIGHT}"
        + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + f"{Style.RESET_ALL}"
    )

print(f"{Fore.BLUE}{Style.BRIGHT}" + "\n*****OBJECTIVE*****\n" + f"{Style.RESET_ALL}")
print(f"{OBJECTIVE}")
print(
    f"{Fore.YELLOW}{Style.BRIGHT}"
    + "\nInitial task:"
    + f"{Style.RESET_ALL}"
    + f" {INITIAL_TASK}"
)

openai.api_key = OPENAI_API
pinecone.init(api_key=PINECONE_API, environment=PINECONE_ENV)
table_name = TABLE_NAME
dimension = 1536
metric = "cosine"
pod_type = "p1"
if table_name not in pinecone.list_indexes():
    pinecone.create_index(
        table_name, dimension=dimension, metric=metric, pod_type=pod_type
    )

index = pinecone.Index(table_name)
task_list = deque([])


def add_task(task: Dict):
    task_list.append(task)


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]


def openai_response(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.5,
    max_tokens: int = 100,
):
    while True:
        try:
            if model.startswith("llama"):
                cmd = ["llama/main", "-p", prompt]
                result = subprocess.run(
                    cmd,
                    shell=True,
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    text=True,
                )
                return result.stdout.strip()
            elif not model.startswith("gpt-"):
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return response.choices[0].text.strip()
            else:
                messages = [{"role": "system", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                )
                return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                "The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again."
            )
            time.sleep(10)


def Agent_TaskCreate(
    objective: str, result: Dict, task_description: str, task_list: List[str]
):
    prompt = f"""You are a task creation AI that uses the result of a completed task to create new tasks with the objective: {objective}. 
    The last completed task had the result: {result}. 
    The task description is: {task_description}. 
    Incomplete tasks are: {', '.join(task_list)}. 
    Generate new tasks based on the result."""
    response = openai_response(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    return [{"task_name": task_name} for task_name in new_tasks]


def prioritization_agent(this_task_id: int):
    global task_list
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    prompt = f"""You are a task prioritization AI responsible for cleaning the formatting and reprioritizing the following tasks: {task_names}. 
    Consider the ultimate objective of your team: {OBJECTIVE}. Do not remove any tasks. 
    Return the result as a numbered list, starting with task number {next_task_id}."""
    response = openai_response(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    task_list = deque()
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            task_list.append({"task_id": task_id, "task_name": task_name})


def context_agent(query: str, n: int):
    query_embedding = get_ada_embedding(query)
    results = index.query(
        query_embedding, top_k=n, include_metadata=True, namespace=OBJECTIVE
    )
    sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
    return [(str(item.metadata["task"])) for item in sorted_results]


def execution_agent(objective: str, task: str) -> str:
    context = context_agent(query=objective, n=5)
    prompt = f"""You are an execution AI tasked with performing a task based on the objective: {objective}. 
    Take into account the previously completed tasks: {context}. 
    Your task is: {task}. 
    Provide a response."""
    return openai_response(prompt, temperature=0.7, max_tokens=2000)


First_Task = {"task_id": 1, "task_name": INITIAL_TASK}
add_task(First_Task)
task_id_counter = 1
while True:
    if task_list:
        print(f"{Fore.MAGENTA}{Style.BRIGHT}\n*****TASK LIST*****\n{Style.RESET_ALL}")
        for t in task_list:
            print(f"{t['task_id']}: {t['task_name']}")
        task = task_list.popleft()
        print(f"{Fore.GREEN}{Style.BRIGHT}\n*****NEXT TASK*****\n{Style.RESET_ALL}")
        print(f"{task['task_id']}: {task['task_name']}")
        result = execution_agent(OBJECTIVE, task["task_name"])
        this_task_id = int(task["task_id"])
        print(f"{Fore.YELLOW}{Style.BRIGHT}\n*****TASK RESULT*****\n{Style.RESET_ALL}")
        print(result)
        enriched_result = {"data": result}
        result_id = f"result_{task['task_id']}"
        vector = get_ada_embedding(enriched_result["data"])
        index.upsert(
            [(result_id, vector, {"task": task["task_name"], "result": result})],
            namespace=OBJECTIVE,
        )
        new_tasks = Agent_TaskCreate(
            OBJECTIVE,
            enriched_result,
            task["task_name"],
            [t["task_name"] for t in task_list],
        )

        for new_task in new_tasks:
            task_id_counter += 1
            new_task.update({"task_id": task_id_counter})
            add_task(new_task)
        prioritization_agent(this_task_id)

    time.sleep(1)
