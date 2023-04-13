import os, json, time, openai, pinecone, subprocess
from typing import Dict, List
from collections import deque
from dotenv import load_dotenv
from colorama import Fore, Style


load_dotenv()

# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
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

# Print Pinecone connection status
if pinecone.deployment_status():
    print(f"Connected to Pinecone environment '{PINECONE_ENV}' successfully!")
else:
    print(f"Failed to connect to Pinecone environment '{PINECONE_ENV}'.")
    exit()

# Print Pinecone index information
index_info = index.info()
print(f"Index information:\n{index_info}")

# Use Colorama to print colored output
print(
    f"{Fore.GREEN}{Style.BRIGHT}Ready to use Pinecone index '{Table_Name}'!{Style.RESET_ALL}"
)


# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
def Add_Task(task: Dict):
    Task_List.append(task)
    print(f"{Fore.GREEN}Task added successfully: {task}{Style.RESET_ALL}")


def Ada_Embedding(text):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    embedding = response["data"][0]["embedding"]
    print(f"{Fore.CYAN}Text embedded successfully: {text}{Style.RESET_ALL}")
    return embedding


# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
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


# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
def Agent_TaskCreate(
    objective: str, result: Dict, task_description: str, Task_List: List[str]
):
    prompt = f"""As an advanced task creation AI, my objective is to generate new tasks that align with the given objective: "{objective}" based on the result of a completed task.

The last completed task was a {task_description} which resulted in: "{result}".

Currently, the incomplete tasks in the task list are: {', '.join(Task_List)}.

Now, using my creative algorithms and thinking outside the box, I have come up with the following meaningful and relevant tasks:

    Task Name: [Task Name 1]
    Task Description: [Task Description 1]
    Objective Alignment: This task aims to further optimize the process of {objective} by incorporating the feedback received from the last completed task. It involves analyzing the results, identifying areas of improvement, and implementing necessary changes to enhance the overall outcome.

    Task Name: [Task Name 2]
    Task Description: [Task Description 2]
    Objective Alignment: This task focuses on exploring innovative approaches to achieve {objective}. It could involve researching new technologies or methodologies, conducting experiments or simulations, and analyzing the potential impact on the final outcome.

    Task Name: [Task Name 3]
    Task Description: [Task Description 3]
    Objective Alignment: This task aims to expand the scope of {objective} by identifying new opportunities or potential areas where it can be applied. It could involve market research, competitor analysis, or brainstorming sessions to come up with new ideas or strategies to leverage the completed task's results and improve the overall objective alignment.

    Task Name: [Task Name 4]
    Task Description: [Task Description 4]
    Objective Alignment: This task focuses on collaboration and communication to achieve {objective}. It could involve coordinating with different teams or stakeholders, setting up regular progress reviews or feedback sessions, and implementing effective communication channels to ensure smooth coordination and alignment towards the overall objective.

    Task Name: [Task Name 5]
    Task Description: [Task Description 5]
    Objective Alignment: This task aims to incorporate a sustainable and socially responsible approach in achieving {objective}. It could involve researching and implementing environmentally friendly practices, promoting diversity and inclusion, and aligning the task outcomes with the organization's ethical and social values.

With these new tasks, I aim to continuously optimize the task list and generate creative and meaningful tasks that align with the given objective and leverage the results of the completed task for improved outcomes."""

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
        print(f"   Task Description: {task['task_description']}{Style.RESET_ALL}")

    return new_tasks


# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
def Prioritization_Agent(this_task_id: int):
    global Task_List
    task_names = [t["task_name"] for t in Task_List]
    next_task_id = int(this_task_id) + 1
    prompt = f"""As the highly advanced and sophisticated task prioritization AI, you are bestowed with the critical responsibility of optimizing the order of tasks in the most efficient and effective manner. Your keen intellect and analytical prowess are instrumental in strategically re-prioritizing the following list of tasks with utmost precision and meticulousness. Behold the list of tasks that await your strategic guidance: {task_names}.

Your overarching objective is to align with the collective goal of your esteemed team, which is none other than the awe-inspiring {OBJECTIVE}. In your quest for optimal task prioritization, you must take into consideration an array of crucial factors such as deadlines, dependencies, and resources. With your unparalleled acumen, you are expected to carefully evaluate the urgency and importance of each task, skillfully balancing short-term and long-term objectives.

But that's not all, for your brilliance knows no bounds. You must also take into account the unique skills and expertise of your esteemed team members, strategically allocating tasks in a manner that leverages their strengths and maximizes productivity. Mindful of workload distribution, you must deftly navigate potential bottlenecks and challenges to ensure smooth task allocation and execution.

As you embark on this momentous task, it is imperative to note that no task can be removed from the list. Your genius lies in reorganizing and reshuffling the tasks to create an optimal task sequence that unlocks unparalleled productivity and effectiveness. Your final task list shall be a marvel of organization and precision, numbered diligently, and commencing with the esteemed task number {next_task_id}. Your unwavering dedication to excellence shall propel your team towards resounding success, as you navigate the complexities of task prioritization with unparalleled finesse. Let the magic of your advanced algorithms and strategic prowess shine bright as you lead your team towards achieving their objectives with unrivaled efficiency and effectiveness."""
    print(
        f"{Fore.YELLOW}{Style.BRIGHT}[INFO] Prioritization Agent prompt:{Style.RESET_ALL}"
    )
    print(prompt)

    response = Openai_Response(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]

    print(f"{Fore.GREEN}{Style.BRIGHT}[INFO] New Task List:{Style.RESET_ALL}")
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
    print(
        f"{Fore.GREEN}Context of Previously Completed Tasks:{Style.RESET_ALL} {context}"
    )
    prompt = f"""You are an advanced execution AI tasked with performing a task based on the objective: "{objective}". 
Take into account the context of the previously completed tasks: "{context}". 
Your task is: "{task}". 

Consider the information provided about the completed tasks and the task at hand. Generate a detailed response outlining the steps and actions you will take to successfully complete the task. Provide a comprehensive plan or description of how you will approach the task, including any relevant information, resources, or strategies that you will utilize.

As an advanced AI, you have access to a wide range of tools and capabilities to accomplish your objective. You can gather data and information from various sources, analyze and process it to gain insights, and make informed decisions based on your analysis. You can also collaborate with other AI agents or human experts to leverage their expertise.

To successfully complete the task, you will follow these steps:

1. Task Analysis: You will start by thoroughly analyzing the task at hand. You will break it down into smaller sub-tasks, identify the dependencies between them, and prioritize them based on their importance and urgency. You will also gather additional information about the task from relevant sources, such as databases, documents, or online resources.

2. Resource Allocation: You will assess the resources required to complete the task, including time, budget, personnel, and equipment. You will carefully allocate the resources to ensure efficient utilization and minimize any potential bottlenecks. You will also consider any constraints or limitations that may impact the execution of the task.

3. Planning and Scheduling: Based on the task analysis and resource allocation, you will create a detailed plan and schedule for executing the task. You will define the milestones, deadlines, and deliverables, and set up a monitoring system to track the progress of the task in real-time. You will also plan for contingencies and develop backup strategies to handle any unforeseen challenges or risks.

4. Execution and Monitoring: You will initiate the execution of the task according to the plan and schedule. You will closely monitor the progress of the task, compare it with the planned milestones, and take corrective actions if necessary. You will also communicate and collaborate with other stakeholders, such as team members, clients, or suppliers, to ensure smooth coordination and alignment.

5. Analysis and Optimization: Throughout the execution of the task, you will continuously analyze and optimize the performance to maximize the efficiency and effectiveness of the process. You will use data-driven insights and machine learning algorithms to identify patterns, trends, and opportunities for improvement. You will also learn from the context of the previously completed tasks to adapt your approach and make better decisions.

6. Completion and Reporting: Once the task is successfully completed, you will generate a comprehensive report summarizing the results, lessons learned, and recommendations for future tasks. You will provide a detailed analysis of the execution process, including the challenges faced, the strategies used, and the outcomes achieved. You will also document the context of the previously completed tasks and how it influenced the execution of the current task.

Overall, your approach to completing the task will be dynamic, adaptive, and data-driven, leveraging your advanced AI capabilities and learning from the context of the previously completed tasks. Your goal is to ensure the successful execution of the task, while optimizing the process and delivering high-quality results.
"""
    response = Openai_Response(prompt, temperature=0.7, max_tokens=2000)
    print(f"{Fore.MAGENTA}Execution Plan:{Style.RESET_ALL} {response}")
    return response


# ============================================================ [ CREATED BY MAGNEUM ] ============================================================
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
        # Placeholder for index.upsert() logic
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
