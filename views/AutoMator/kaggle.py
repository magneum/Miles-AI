from typing import Dict, List
from collections import deque
from dotenv import load_dotenv
from colorama import Fore, Style
import os, json, time, openai, pinecone


load_dotenv()


# Get environment variables
INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv("FIRST_TASK", ""))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API = os.getenv("PINECONE_API", "")
PINECONE_ENV = os.getenv("PINECONE_ENV", "")
TABLE_NAME = os.getenv("TABLE_NAME", "")
OBJECTIVE = os.getenv("OBJECTIVE", "")


if not (
    OPENAI_API_KEY and TABLE_NAME and PINECONE_ENV and PINECONE_API and OPENAI_MODEL
):
    print(
        f"{Fore.RED}Error:{Style.RESET_ALL} One or more environment variables are missing from .env"
    )
    exit(1)
else:
    print(f"{Fore.GREEN}Environment variables are present.{Style.RESET_ALL}")


if "gpt-4" in OPENAI_MODEL.lower():
    print(
        f"{Fore.RED}{Style.BRIGHT}"
        + "\n=+=+=+=+=+=+=+=+=+=+=+=[ USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS ]=+=+=+=+=+=+=+=+=+=+=+\n"
        + f"{Style.RESET_ALL}"
    )

print(
    f"{Fore.BLUE}{Style.BRIGHT}"
    + "\n=+=+=+=+=+=+=+=+=+=+=+=[ OBJECTIVE ]=+=+=+=+=+=+=+=+=+=+=+\n"
    + f"{Style.RESET_ALL}"
)
print(f"{OBJECTIVE}")
print(
    f"{Fore.YELLOW}{Style.BRIGHT}"
    + "\nInitial task:"
    + f"{Style.RESET_ALL}"
    + f" {INITIAL_TASK}"
)


openai.api_key = OPENAI_API_KEY
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


def Add_Task(task: Dict):
    if not isinstance(task, dict):
        raise ValueError("Task must be a dictionary.")
    Task_List.append(task)


def Ada_Embedding(text):
    if not isinstance(text, str) or not text:
        raise ValueError("Text must be a non-empty string.")
    text = text.replace("\n", " ")
    try:
        response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
        embedding = response["data"][0]["embedding"]
        return embedding
    except Exception as e:
        raise Exception("Failed to generate embedding: {}".format(str(e)))


def openai_response(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.5,
    max_tokens: int = 100,
):
    while True:
        try:
            if not model.startswith("gpt-"):
                # For non-chat models
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
                # For chat models
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
            # Handle rate limit error by waiting and retrying
            print(
                Fore.YELLOW
                + "The OpenAI API rate limit has been exceeded. Waiting 30 seconds and trying again."
                + Style.RESET_ALL
            )
            time.sleep(30)
        except Exception as e:
            # Handle other exceptions and return None
            print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)
            return None


def Agent_TaskCreate(
    objective: str, result: Dict, task_description: str, Task_List: List[str]
):
    prompt = f"""
    You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective},
    The last completed task has the result: {result}.
    This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(Task_List)}.
    Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks.
    Return the tasks as an array."""
    if not objective:
        raise ValueError("Objective cannot be empty or None")
    if not result:
        raise ValueError("Result cannot be empty or None")
    if not task_description:
        raise ValueError("Task description cannot be empty or None")
    if not Task_List:
        raise ValueError("Task List cannot be empty or None")
    if not isinstance(objective, str):
        raise TypeError("Objective must be a string")
    if not isinstance(result, dict):
        raise TypeError("Result must be a dictionary")
    if not isinstance(task_description, str):
        raise TypeError("Task description must be a string")
    if not isinstance(Task_List, list):
        raise TypeError("Task List must be a list")

    response = openai_response(prompt)
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
    return new_tasks


def Prioritization_Agent(this_task_id: int):
    global Task_List
    if not isinstance(this_task_id, int):
        raise TypeError("this_task_id must be an integer")
    task_names = [t["task_name"] for t in Task_List]
    next_task_id = int(this_task_id) + 1
    prompt = f"""
    You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}.
    Consider the ultimate objective of your team:{OBJECTIVE}.
    Do not remove any tasks. Return the result as a numbered list, like:
    #. First task
    #. Second task
    Start the task list with number {next_task_id}."""
    response = openai_response(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    Task_List = deque()
    for i, task_string in enumerate(new_tasks, start=next_task_id):
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            Task_List.append({"task_id": task_id, "task_name": task_name})
            print(Fore.CYAN + Style.BRIGHT + "[INFO] Task {}: {}".format(i, task_name))
            print(Style.RESET_ALL)
        else:
            print(
                Fore.RED
                + Style.BRIGHT
                + "[ERROR] Invalid task format: {}".format(task_string)
            )
            print(Style.RESET_ALL)


def Context_Agent(query: str, n: int):
    if not isinstance(query, str):
        raise ValueError("Query must be a string.")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")

    try:
        query_embedding = Ada_Embedding(query)
        results = index.query(
            query_embedding, top_k=n, include_metadata=True, namespace=OBJECTIVE
        )
        sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
        return [(str(item.metadata["task"])) for item in sorted_results]
    except Exception as e:
        print("Error: ", e)
        return None


def Execution_Agent(objective: str, task: str) -> str:
    if not objective or not task:
        raise ValueError("Objective and task must not be empty or None.")
    try:
        context = Context_Agent(query=objective, n=5)
    except Exception as e:
        print(f"Failed to retrieve context: {e}")
        context = ""
    prompt = f"""
    You are an AI who performs one task based on the following objective: {objective}\n.
    Take into account these previously completed tasks: {context}\n.
    Your task: {task}\nResponse:"""
    try:
        response = openai_response(prompt, temperature=0.7, max_tokens=2000)
        print(Fore.MAGENTA + "Execution Plan:" + Style.RESET_ALL + " " + response)
    except Exception as e:
        print(f"Failed to generate response: {e}")
        response = ""

    return response


First_Task = {"task_id": 1, "task_name": INITIAL_TASK}
Add_Task(First_Task)
task_id_counter = 1


if not os.path.exists("views/AutoMator"):
    os.makedirs("views/AutoMator")

while True:
    if Task_List:
        print(
            Fore.MAGENTA
            + Style.BRIGHT
            + "\n=+=+=+=+=+=+=+=+=+=+=+=[ TASK LIST ]=+=+=+=+=+=+=+=+=+=+=+\n"
        )
        print(Style.RESET_ALL)
        for t in Task_List:
            print(str(t["task_id"]) + ": " + t["task_name"])
        task = Task_List.popleft()
        print(
            Fore.GREEN
            + Style.BRIGHT
            + "\n=+=+=+=+=+=+=+=+=+=+=+=[ NEXT TASK ]=+=+=+=+=+=+=+=+=+=+=+\n"
        )
        print(Style.RESET_ALL)
        print(str(task["task_id"]) + ": " + task["task_name"])
        result = Execution_Agent(OBJECTIVE, task["task_name"])
        this_task_id = int(task["task_id"])
        print(
            Fore.YELLOW
            + Style.BRIGHT
            + "\n=+=+=+=+=+=+=+=+=+=+=+=[ TASK RESULT ]=+=+=+=+=+=+=+=+=+=+=+\n"
        )
        print(Style.RESET_ALL)
        print(result)
        enriched_result = {"data": result}
        result_id = "result_" + str(task["task_id"])
        vector = Ada_Embedding(enriched_result["data"])
        print(
            Fore.CYAN
            + Style.BRIGHT
            + "\n=+=+=+=+=+=+=+=+=+=+=+=[ UPDATING INDEX ]=+=+=+=+=+=+=+=+=+=+=+\n"
        )
        print(Style.RESET_ALL)
        print("Updating index with result_id: " + result_id)
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
        output_list = []
        output_list.append({"title": "TASK LIST", "data": []})
        for t in Task_List:
            output_list[-1]["data"].append(str(t["task_id"]) + ": " + t["task_name"])
        output_list.append(
            {
                "title": "NEXT TASK",
                "data": str(task["task_id"]) + ": " + task["task_name"],
            }
        )
        output_list.append({"title": "TASK RESULT", "data": result})
        output_list.append(
            {
                "title": "UPDATING INDEX",
                "data": "Updating index with result_id: " + result_id,
            }
        )
        file_name = f"views/AutoMator/output_{result_id}.json"
        with open(file_name, "w") as f:
            json.dump(output_list, f)
            print(
                Fore.GREEN
                + Style.BRIGHT
                + "File updated: "
                + os.path.abspath(f.name)
                + Style.RESET_ALL
            )
    time.sleep(1)
