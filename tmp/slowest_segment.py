
from pathlib import Path
from mteb import MTEB
import json

path = Path("results/results/paraphrase-multilingual-MiniLM-L12-v2/")

# all json files
json_files = list(path.glob("*.json"))

# load all results
results = []
for file in json_files:
    with open(file, "r") as f:
        results.append(json.load(f))

results[0]

def get_task_type(task: str):
    tmp_bench = MTEB(tasks=[task])
    if len(tmp_bench.tasks) == 0:
        return None
    return tmp_bench.tasks[0].metadata_dict["type"]
    
for result in results:
    result["task_type"] = get_task_type(result["mteb_dataset_name"])

task_types = set([result["task_type"] for result in results])

# get mean "evaluation_time" for each task type
scores = {}
for task_type in task_types:
    task_times = []
    for result in results:
        if result["task_type"] == task_type:
            if "evaluation_time" in result:
                task_times.append(result["evaluation_time"])
            elif "test" in result and "evaluation_time" in result["test"]:
                task_times.append(result["test"]["evaluation_time"])
            else:
                print(f"No evaluation time for {result['mteb_dataset_name']}")
    
    # get name of max and min
    name_of_max = None
    name_of_min = None
    for result in results:
        if result["task_type"] == task_type:
            if "evaluation_time" in result and result["evaluation_time"] == max(task_times):
                name_of_max = result["mteb_dataset_name"]
            if "evaluation_time" in result and result["evaluation_time"] == min(task_times):
                name_of_min = result["mteb_dataset_name"]
            if "test" in result and "evaluation_time" in result["test"] and result["test"]["evaluation_time"] == max(task_times):
                name_of_max = result["mteb_dataset_name"]
            if "test" in result and "evaluation_time" in result["test"] and result["test"]["evaluation_time"] == min(task_times):
                name_of_min = result["mteb_dataset_name"]
                
    scores[task_type] = {
        "mean": sum(task_times) / len(task_times),
        "n": len(task_times),
        "total": sum(task_times),
        "median": sorted(task_times)[len(task_times) // 2],
        "min": min(task_times),
        "max": max(task_times),
        "name_of_max": name_of_max,
        "name_of_min": name_of_min,
    }