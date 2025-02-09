from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

import mteb
from mteb import TaskResult


def generate_readme(results_folder: Path, from_existing: Path | None = None) -> str:
    task_results = get_task_results(results_folder)
    yaml_results = []
    for task_result in task_results:
        yaml_results.extend(process_task_result(task_result))

    model_name = load_model_name(results_folder)
    yaml_dict = {
        "tags": ["mteb"],
        "model-index": [
            {
                "name": model_name,
                "results": yaml_results,
            }
        ],
    }

    if from_existing:
        yaml_dict, readme_end = merge_yamls(yaml_dict, from_existing)
    else:
        readme_end = ""

    yaml_str = yaml.dump(yaml_dict)
    return f"---\n{yaml_str}---\n{readme_end}"


def load_model_name(results_folder: Path) -> str:
    model_meta_path = results_folder / "model_meta.json"
    if model_meta_path.exists():
        with model_meta_path.open("r") as f:
            return json.load(f)["name"]
    return "PLACEHOLDER"


def process_task_result(task_result: TaskResult) -> list[dict[str, Any]]:
    task = mteb.get_task(task_result.task_name)
    yaml_results = []

    for split, hf_subset_scores in task_result.scores.items():
        for hf_subset_score in hf_subset_scores:
            metrics = [
                {"type": k, "value": v * 100}
                # convert to percentage (for consistency with the leaderboard and to make it more readable)
                for k, v in hf_subset_score.items()
                if isinstance(v, (int, float))
            ]

            if task.metadata.main_score not in hf_subset_score:
                raise ValueError(
                    f"Main score {task.metadata.main_score} not found in metrics or is not a number."
                )

            yaml_result = {
                "task": {"type": task.metadata.type},
                "dataset": {
                    "type": task.metadata.dataset["path"],
                    "name": f"MTEB {task.metadata.name} ({hf_subset_score['hf_subset']})",
                    "config": hf_subset_score["hf_subset"],
                    "split": split,
                    "revision": task_result.dataset_revision,
                },
                "metrics": metrics,
            }
            yaml_results.append(yaml_result)

    return yaml_results


def get_task_results(results_folder: Path) -> list[TaskResult]:
    json_files = [
        r
        for r in results_folder.glob("*.json")
        if r.is_file() and r.name != "model_meta.json" and "predictions" not in r.name
    ]
    task_results = [TaskResult.from_disk(path) for path in json_files]
    task_results = [
        results
        for results in task_results
        if results.task_name not in ["GPUSpeedTask", "CPUSpeedTask"]
    ]
    # We should ideally find better way in the future to aggregate scores for tasks like CQADupstack
    task_results = potentially_add_cqadupstack_to_results(task_results)
    return sorted(task_results, key=lambda x: x.task_name)


def potentially_add_cqadupstack_to_results(
    results: list[TaskResult],
) -> list[TaskResult]:
    task_list_cqa = {
        "CQADupstackAndroidRetrieval",
        "CQADupstackEnglishRetrieval",
        "CQADupstackGamingRetrieval",
        "CQADupstackGisRetrieval",
        "CQADupstackMathematicaRetrieval",
        "CQADupstackPhysicsRetrieval",
        "CQADupstackProgrammersRetrieval",
        "CQADupstackStatsRetrieval",
        "CQADupstackTexRetrieval",
        "CQADupstackUnixRetrieval",
        "CQADupstackWebmastersRetrieval",
        "CQADupstackWordpressRetrieval",
    }

    task_names = {result.task_name for result in results}
    if not task_list_cqa.issubset(task_names):
        return results

    cqa_results = [result for result in results if result.task_name in task_list_cqa]
    evaluation_time = sum(result.evaluation_time for result in cqa_results)
    main_scores = [r.get_score(splits=["test"]) for r in cqa_results]
    main_score = float(sum(main_scores) / len(main_scores))

    combined_result = TaskResult(
        task_name="CQADupstackRetrieval",
        dataset_revision="CQADupstackRetrieval_is_a_combined_dataset",
        mteb_version="NA",
        scores={
            "test": [
                {
                    "main_score": main_score,
                    "ndcg_at_10": main_score,
                    "hf_subset": "default",
                    "languages": ["eng_Latn"],
                }
            ]
        },
        evaluation_time=evaluation_time,
        kg_co2_emissions=None,
    )
    results.append(combined_result)
    return results


def merge_yamls(
    yaml_dict: dict[str, Any], existing_readme: Path
) -> tuple[dict[str, Any], str]:
    if not existing_readme.name.lower().endswith(".md"):
        raise ValueError("Readme file should be markdown and end with '.md'")

    with open(existing_readme) as f:
        existing_file = f.read()

    existing_yaml_dict, readme_end = extract_yaml_and_content(existing_file)
    existing_yaml_dict = update_yaml_dict(existing_yaml_dict, yaml_dict)

    return existing_yaml_dict, readme_end


def extract_yaml_and_content(file_content: str) -> tuple[dict[str, Any], str]:
    yaml_start_sep = "---"
    yaml_end_sep = "\n---\n"  # newline to avoid matching "---" in the content
    if file_content.startswith(yaml_start_sep) and yaml_end_sep in file_content:
        start_yaml_index = file_content.index(yaml_start_sep) + len(yaml_start_sep)
        end_yaml_index = file_content.index(yaml_end_sep, start_yaml_index)
        existing_yaml = file_content[start_yaml_index:end_yaml_index]
        readme_end = file_content[end_yaml_index + len(yaml_end_sep) :]
        return yaml.safe_load(existing_yaml), readme_end
    return {}, file_content


def update_yaml_dict(
    existing_yaml_dict: dict[str, Any], new_yaml_dict: dict[str, Any]
) -> dict[str, Any]:
    if "tags" not in existing_yaml_dict:
        existing_yaml_dict["tags"] = []
    if "mteb" not in existing_yaml_dict["tags"]:
        existing_yaml_dict["tags"].append("mteb")

    if "model-index" not in existing_yaml_dict:
        existing_yaml_dict["model-index"] = new_yaml_dict.get("model-index", [])
        return existing_yaml_dict

    # model index always array with one element
    existing_model = existing_yaml_dict["model-index"][0]
    new_model = new_yaml_dict.get("model-index", [{}])[0]

    if existing_model["name"] != new_model["name"]:
        raise ValueError("Model names do not match")

    existing_results = {
        (r["dataset"]["name"], r["dataset"].get("config")): r
        for r in existing_model["results"]
    }
    for new_result in new_model["results"]:
        key = (new_result["dataset"]["name"], new_result["dataset"].get("config"))
        if key in existing_results:
            existing_results[key]["metrics"] = new_result["metrics"]
        else:
            existing_results[key] = new_result

    existing_model["results"] = list(existing_results.values())
    return existing_yaml_dict
