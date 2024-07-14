from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

import mteb

REVISION = str
MODEL = str


def filter_results(
    results,
    tasks: Iterable[mteb.AbsTask] | None = None,
    models: list[mteb.ModelMeta] | None = None,
):
    _results = {}

    iter_models = models if models is not None else results.keys()
    if tasks is not None:
        task_names = set(t.metadata.name for t in tasks)

    for mdl in iter_models:
        if isinstance(mdl, mteb.ModelMeta):
            model_name = mdl.name
            revisions = (
                [mdl.revision] if mdl.revision is not None else results[model_name]
            )
        else:
            model_name = mdl
            revisions = results[model_name]

        _results[model_name] = {}

        if model_name not in results:
            continue

        for rev in revisions:
            _results[model_name][rev] = []

            if rev not in results[model_name]:
                continue

            tasks_results = results[model_name][rev]

            if tasks is not None:
                task_res = [r for r in tasks_results if r.task_name in task_names]
            else:
                task_res = tasks_results
            _results[model_name][rev] = task_res

    return _results


def results_to_dataframe(
    mteb_results: dict[MODEL, dict[REVISION, list[mteb.MTEBResults]]],
):
    data = []
    for model_name, revisions in mteb_results.items():
        for rev, tasks_results in revisions.items():
            for task_result in tasks_results:
                for split, scores in task_result.scores.items():
                    for score in scores:
                        data.append(
                            {
                                "model": model_name,
                                "revision": rev,
                                "task": task_result.task_name,
                                "split": split,
                                "hf_subset": score["hf_subset"],
                                "main_score": score["main_score"],
                            }
                        )
    return pd.DataFrame(data)


def create_results_tables(
    results: dict[str, dict[str, list[mteb.MTEBResults]]],
    task_types: list[str],
):
    tasks = mteb.get_tasks(task_types=task_types)  # type: ignore
    results = filter_results(results, tasks, models)
    results_df = results_to_dataframe(results)

    # create a task correlation matrix base on correlation between model performance on different tasks
    # a task is task_name + split + hf_subset
    # model is model_name + revision

    wide_table = results_df.pivot_table(
        index=["model", "revision"],
        columns=["task", "split", "hf_subset"],
        values="main_score",
    )

    md_table = wide_table.reset_index()
    md_table["Model"] = md_table["model"] + "(" + md_table["revision"] + ")"
    md_table = md_table[md_table.columns.drop(["revision", "model"])]
    md_table = md_table.set_index("Model")

    # to csv
    l_table = md_table.T
    l_table = (l_table * 100).round(1)

    save_path = (
        Path(__file__).parent
        / "results_tables"
        / ("_".join([t.lower() for t in task_types]) + ".csv")
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    l_table.to_csv(save_path)


model_names = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/LaBSE",
    "intfloat/multilingual-e5-large-instruct",
    "intfloat/e5-mistral-7b-instruct",
    "GritLM/GritLM-7B",
    "GritLM/GritLM-8x7B",
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
]

if __name__ == "__main__":
    results = mteb.load_results()

    # only include bitext tasks
    models: list[mteb.ModelMeta] = [mteb.get_model_meta(name) for name in model_names]

    tasks_types = [
        "Classification",
        "MultilabelClassification",
        "Reranking",
        "BitextMining",
        "PairClassification",
        "STS",
        "Summarization",
        # "Clustering",
        # "Retrieval",
        # "InstructionRetrieval",
    ]

    for task_type in tasks_types:
        create_results_tables(results, [task_type])
