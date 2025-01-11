from __future__ import annotations

import json
import logging
from pathlib import Path

from mteb import MTEB_ENG_CLASSIC, load_results

logging.basicConfig(level=logging.INFO)

models = [
    "dunzhang/stella_en_1.5B_v5",
    "dunzhang/stella_en_400M_v5",
    # Add other models here
]

# in same folder as mteb repo
# git clone https://github.com/embeddings-benchmark/leaderboard
data_tasks_path = Path("../../leaderboard/boards_data/en/data_tasks/")

results = []

for model_name_to_search in models:
    model_results = load_results(
        models=[model_name_to_search],
        tasks=MTEB_ENG_CLASSIC.tasks,
        only_main_score=True,
    )

    cur_model = {}
    for model_res in model_results:
        for task_res in model_res.task_results:
            task_name = task_res.task.metadata.name
            split = "test" if task_name != "MSMARCO" else "dev"
            scores = [score["main_score"] for score in task_res.scores[split]]
            # this tmp solution, because some tasks have multiple results
            cur_model[task_name] = {"new": round((sum(scores) / len(scores)) * 100, 2)}

    for task_dir in data_tasks_path.iterdir():
        if task_dir.is_dir():
            results_file_path = task_dir / "default.jsonl"
            if results_file_path.exists():
                with open(results_file_path) as file:
                    for line in file:
                        data = json.loads(line)
                        model_name = data.get("Model", "")
                        if model_name_to_search in model_name:
                            for key, value in data.items():
                                if key in [
                                    "index",
                                    "Rank",
                                    "Model",
                                    "Model Size (Million Parameters)",
                                    "Memory Usage (GB, fp32)",
                                    "Embedding Dimensions",
                                    "Max Tokens",
                                    "Average",
                                ]:
                                    continue
                                for benchmark_task in MTEB_ENG_CLASSIC.tasks:
                                    if benchmark_task.metadata.name in key:
                                        cur_model[benchmark_task.metadata.name][
                                            "old"
                                        ] = value

    sorted_cur_model = {
        task.metadata.name: cur_model[task.metadata.name]
        for task in MTEB_ENG_CLASSIC.tasks
        if task.metadata.name in cur_model
    }
    results.append({"model": model_name_to_search, "results": sorted_cur_model})

# Write results to JSONL file
with open("results.jsonl", "w") as file:
    for result in results:
        file.write(json.dumps(result) + "\n")

# Write results to Markdown file
with open("results.md", "w") as file:
    for result in results:
        file.write(f"## Model: {result['model']}\n\n")
        file.write("| Task Name | Old Leaderboard | New Leaderboard |\n")
        file.write("|-----------|-----------------|-----------------|\n")
        for task_name, scores in result["results"].items():
            old_score = scores.get("old", "N/A")
            new_score = scores.get("new", "N/A")
            file.write(f"| {task_name} | {old_score} | {new_score} |\n")
        file.write("\n")
