import os, json, time, openai, pinecone, subprocess
from typing import Dict, List
from collections import deque
from dotenv import load_dotenv
from colorama import Fore, Style


load_dotenv()

# ============================================================ [ CREATED BY MAGNEUM ]
# Get environment variables
INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv("FIRST_TASK", ""))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
PINECONE_API = os.getenv("PINECONE_API", "")
PINECONE_ENV = os.getenv("PINECONE_ENV", "")
OPENAI_API = os.getenv("OPENAI_API", "")
TABLE_NAME = os.getenv("TABLE_NAME", "")
OBJECTIVE = os.getenv("OBJECTIVE", "")

# Check if environment variables are present
if not (OPENAI_API and TABLE_NAME and PINECONE_ENV and PINECONE_API and OPENAI_MODEL):
    print(
        f"{Fore.RED}Error:{Style.RESET_ALL} One or more environment variables are missing from .env"
    )
    exit(1)
else:
    print(f"{Fore.GREEN}Environment variables are present.{Style.RESET_ALL}")


# ============================================================ [ CREATED BY MAGNEUM ]
if "gpt-4" in OPENAI_MODEL.lower():
    print(
        f"{Fore.RED}{Style.BRIGHT}"
        + "\n======[ USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS ]======\n"
        + f"{Style.RESET_ALL}"
    )

print(
    f"{Fore.BLUE}{Style.BRIGHT}"
    + "\n======[ OBJECTIVE ]======\n"
    + f"{Style.RESET_ALL}"
)
print(f"{OBJECTIVE}")
print(
    f"{Fore.YELLOW}{Style.BRIGHT}"
    + "\nInitial task:"
    + f"{Style.RESET_ALL}"
    + f" {INITIAL_TASK}"
)

# ============================================================ [ CREATED BY MAGNEUM ]
openai.api_key = OPENAI_API
pinecone.init(api_key=PINECONE_API, environment=PINECONE_ENV)
Table_Name = TABLE_NAME.lower()
dimension = 1536
metric = "cosine"
pod_type = "p1"
if Table_Name not in pinecone.list_indexes():
    pinecone.create_index(
        Table_Name, dimension=dimension, metric=metric, pod_type=pod_type
    )
    print(f"Index '{Table_Name}' created successfully!")
else:
    print(f"Index '{Table_Name}' already exists!")

index = pinecone.Index(Table_Name)
Task_List = deque([])


# ============================================================ [ CREATED BY MAGNEUM ]
def Add_Task(task: Dict):
    Task_List.append(task)
    print(f"{Fore.GREEN}Task added successfully: {task}{Style.RESET_ALL}")


def Ada_Embedding(text):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    embedding = response["data"][0]["embedding"]
    print(f"{Fore.CYAN}Text embedded successfully: {text}{Style.RESET_ALL}")
    return embedding


# ============================================================ [ CREATED BY MAGNEUM ]
def Openai_Response(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.5,
    max_tokens: int = 100,
):
    while True:
        try:
            if model.startswith("llama"):
                result = subprocess.run(
                    ["llama/main", "-p", prompt],
                    text=True,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                return result.stdout.strip()
            elif not model.startswith("gpt-"):
                response = openai.Completion.create(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0,
                    prompt=prompt,
                    engine=model,
                    top_p=1,
                )
                return response.choices[0].text.strip()
            else:
                response = openai.ChatCompletion.create(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[{"role": "system", "content": prompt}],
                    model=model,
                    stop=None,
                    n=1,
                )
                return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                Fore.YELLOW
                + "The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again."
                + Style.RESET_ALL
            )
            time.sleep(10)
        except Exception as e:
            print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)
            return None


# ============================================================ [ CREATED BY MAGNEUM ]
def Agent_TaskCreate(
    objective: str, result: Dict, task_description: str, Task_List: List[str]
):
    prompt = f"""
    You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective},
    The last completed task has the result: {result}.
    This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(Task_List)}.
    Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks.
    Return the tasks as an array."""
    response = Openai_Response(prompt)
    task_names = [
        line.split("Task Name: ")[1]
        for line in response.split("\n")
        if "Task Name:" in line
    ]
    task_descriptions = [
        line.split("Task Description: ")[1]
        for line in response.split("\n")
        if "Task Description:" in line
    ]
    new_tasks = [
        {"task_name": task_name, "task_description": task_description}
        for task_name, task_description in zip(task_names, task_descriptions)
    ]
    print(f"{Fore.GREEN}Generated New Tasks:{Style.RESET_ALL}")
    for i, task in enumerate(new_tasks, 1):
        print(f"{Fore.YELLOW}{i}. Task Name: {task['task_name']}")
        print(f"Task Description: {task['task_description']}{Style.RESET_ALL}")
    return new_tasks


# ============================================================ [ CREATED BY MAGNEUM ]
def Prioritization_Agent(this_task_id: int):
    global Task_List
    task_names = [t["task_name"] for t in Task_List]
    next_task_id = int(this_task_id) + 1
    prompt = f"""
    You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}.
    Consider the ultimate objective of your team:{OBJECTIVE}.
    Do not remove any tasks. Return the result as a numbered list, like:
    #. First task
    #. Second task
    Start the task list with number {next_task_id}."""
    response = Openai_Response(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    Task_List = deque()
    for i, task_string in enumerate(new_tasks, start=next_task_id):
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            Task_List.append({"task_id": task_id, "task_name": task_name})
            print(
                f"{Fore.CYAN}{Style.BRIGHT}[INFO] Task {i}: {task_name}{Style.RESET_ALL}"
            )
        else:
            print(
                f"{Fore.RED}{Style.BRIGHT}[ERROR] Invalid task format: {task_string}{Style.RESET_ALL}"
            )


# ============================================================ [ CREATED BY MAGNEUM ]
def Context_Agent(query: str, n: int):
    query_embedding = Ada_Embedding(query)
    results = index.query(
        query_embedding, top_k=n, include_metadata=True, namespace=OBJECTIVE
    )
    sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
    return [(str(item.metadata["task"])) for item in sorted_results]


def Execution_Agent(objective: str, task: str) -> str:
    context = Context_Agent(query=objective, n=5)
    prompt = f"""
    You are an AI who performs one task based on the following objective: {objective}\n.
    Take into account these previously completed tasks: {context}\n.
    Your task: {task}\nResponse:"""
    response = Openai_Response(prompt, temperature=0.7, max_tokens=2000)
    print(f"{Fore.MAGENTA}Execution Plan:{Style.RESET_ALL} {response}")
    return response


# ============================================================ [ CREATED BY MAGNEUM ]
# Initial task
First_Task = {"task_id": 1, "task_name": "INITIAL_TASK"}
Add_Task(First_Task)
task_id_counter = 1

while True:
    if Task_List:
        print(
            f"{Fore.MAGENTA}{Style.BRIGHT}\n======[ TASK LIST ]======\n{Style.RESET_ALL}"
        )
        for t in Task_List:
            print(f"{t['task_id']}: {t['task_name']}")
        task = Task_List.popleft()
        print(
            f"{Fore.GREEN}{Style.BRIGHT}\n======[ NEXT TASK ]======\n{Style.RESET_ALL}"
        )
        print(f"{task['task_id']}: {task['task_name']}")
        result = Execution_Agent(OBJECTIVE, task["task_name"])
        this_task_id = int(task["task_id"])
        print(
            f"{Fore.YELLOW}{Style.BRIGHT}\n======[ TASK RESULT ]======\n{Style.RESET_ALL}"
        )
        print(result)
        enriched_result = {"data": result}
        result_id = f"result_{task['task_id']}"
        vector = Ada_Embedding(enriched_result["data"])
        print(
            f"{Fore.CYAN}{Style.BRIGHT}\n======[ UPDATING INDEX ]======\n{Style.RESET_ALL}"
        )
        print(f"Updating index with result_id: {result_id}, vector: {vector}")
        new_tasks = Agent_TaskCreate(
            OBJECTIVE,
            enriched_result,
            task["task_name"],
            [t["task_name"] for t in Task_List],
        )

        for new_task in new_tasks:
            task_id_counter += 1
            new_task.update({"task_id": task_id_counter})
            Add_Task(new_task)
        Prioritization_Agent(this_task_id)

    time.sleep(1)
