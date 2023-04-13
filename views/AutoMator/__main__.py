import os, json, time, openai, pinecone, subprocess
from typing import Dict, List
from collections import deque
from dotenv import load_dotenv
from colorama import Fore, Style


load_dotenv()

# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv("FIRST_TASK", ""))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
PINECONE_API = os.getenv("PINECONE_API", "")
PINECONE_ENV = os.getenv("PINECONE_ENV", "")
OPENAI_API = os.getenv("OPENAI_API", "")
TABLE_NAME = os.getenv("TABLE_NAME", "")
OBJECTIVE = os.getenv("OBJECTIVE", "")


assert (
    OPENAI_API and TABLE_NAME and PINECONE_ENV and PINECONE_API and OPENAI_MODEL
), "One or more environment variables are missing from .env"


# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
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

# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
all_responses = []
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

index = pinecone.Index(Table_Name)
Task_List = deque([])


# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
def Add_Task(task: Dict):
    Task_List.append(task)


def Ada_Embedding(text):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    embedding = response["data"][0]["embedding"]
    return embedding


# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
def Openai_Response(
    prompt: str,
    model: str = OPENAI_MODEL,
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
                all_responses.append(response.choices[0].text.strip())
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
                all_responses.append(response.choices[0].text.strip())
                return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                "The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again."
            )
            time.sleep(10)


# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
def Agent_TaskCreate(
    objective: str, result: Dict, task_description: str, Task_List: List[str]
):
    prompt = f"""You are an advanced task creation AI that uses the result of a completed task to generate new tasks with the objective: "{objective}". 
The last completed task had the result: "{result}". 
The task description is: "{task_description}". 
Incomplete tasks are: {', '.join(Task_List)}. 

Based on the result of the completed task, generate at least 3 new tasks that align with the given objective. Each new task should have a unique name and a brief description. Make sure the new tasks are meaningful and relevant to the objective. Be creative and think outside the box!

New Tasks:
1. Task Name: [Task Name 1]
   Task Description: [Task Description 1]

2. Task Name: [Task Name 2]
   Task Description: [Task Description 2]

3. Task Name: [Task Name 3]
   Task Description: [Task Description 3]
"""
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
    with open("views/AutoMator/AutoMator.json", "w") as f:
        json.dump(all_responses, f, indent=4)
    return new_tasks


# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
def Prioritization_Agent(this_task_id: int):
    global Task_List
    task_names = [t["task_name"] for t in Task_List]
    next_task_id = int(this_task_id) + 1
    prompt = f"""As the advanced task prioritization AI, you are tasked with optimizing the order of tasks by cleaning up the formatting and strategically re-prioritizing the following list of tasks: {task_names}. Your objective is to align with the overall goal of your team: {OBJECTIVE}, while considering various factors such as deadlines, dependencies, and resources. You are also expected to consider the urgency and importance of each task, and balance short-term and long-term objectives.

    In addition, you are required to take into account the skills and expertise of team members, workload distribution, and potential bottlenecks in order to ensure efficient task allocation. Your goal is to create an optimal task sequence that maximizes productivity and effectiveness, while minimizing delays and conflicts.

    It is essential to note that no tasks should be removed from the list, and all tasks need to be included in the final prioritized sequence. Please provide the revised task list as a numbered and well-organized list, starting with task number {next_task_id}, to help your team achieve its objectives efficiently and effectively."""
    response = Openai_Response(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    Task_List = deque()
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            Task_List.append({"task_id": task_id, "task_name": task_name})


# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
def Context_Agent(query: str, n: int):
    query_embedding = Ada_Embedding(query)
    results = index.query(
        query_embedding, top_k=n, include_metadata=True, namespace=OBJECTIVE
    )
    sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
    return [(str(item.metadata["task"])) for item in sorted_results]


def Execution_Agent(objective: str, task: str) -> str:
    context = Context_Agent(query=objective, n=5)
    prompt = f"""You are an advanced execution AI tasked with performing a task based on the objective: "{objective}". 
Take into account the context of the previously completed tasks: "{context}". 
Your task is: "{task}". 

Consider the information provided about the completed tasks and the task at hand. Generate a detailed response outlining the steps and actions you will take to successfully complete the task. Provide a comprehensive plan or description of how you will approach the task, including any relevant information, resources, or strategies that you will utilize.

Response:
[Your detailed response here]
"""
    return Openai_Response(prompt, temperature=0.7, max_tokens=2000)


# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
First_Task = {"task_id": 1, "task_name": INITIAL_TASK}
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
        index.upsert(
            [(result_id, vector, {"task": task["task_name"], "result": result})],
            namespace=OBJECTIVE,
        )
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
