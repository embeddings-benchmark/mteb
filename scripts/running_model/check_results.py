"""This script checks that all result have been run and writes a file: missing_tasks.txt with the missing tasks for each model"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

import mteb

MODEL = str
REVISION = str


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


def filter_results(
    results,
    tasks: Iterable[mteb.AbsTask] | None = None,
    models: list[mteb.ModelMeta] | None = None,
):
    _results = {}

    iter_models = models if models is not None else results.keys()
    if tasks is not None:
        task_names = {t.metadata.name for t in tasks}

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


def normalize_results(results):
    for model_name, revisions in results.items():
        for rev, tasks_results in revisions.items():
            for task_result in tasks_results:
                try:
                    task_result.validate_and_filter_scores()
                except Exception:
                    print(
                        f"Error validating and filtering scores for {model_name} {rev} {task_result.task_name}. Some splits are missing"
                    )
                    # print(e)
    return results


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


tasks_which_should_be_there = mteb.get_tasks(
    task_types=[
        "BitextMining",
        "Classification",
        "MultilabelClassification",
        "PairClassification",
        "Reranking",
        "STS",
        "Summarization",
        "Clustering",
        "InstructionRetrieval",
        "Retrieval",
    ]
)

retrieval_to_be_downsampled = [
    "TopiOCQA",
    "MSMARCO-PL",
    "ClimateFEVER",
    "FEVER",
    "HotpotQA",
    "HotpotQA-PL",
    "DBPedia",
    "DBPedia-PL",
    "NeuCLIR2022Retrieval",
    "NeuCLIR2023Retrieval",
    "NeuCLIR2022Retrieval",
    "NeuCLIR2023Retrieval",
    "NQ",
    "NQ-PL",
    "NeuCLIR2022Retrieval",
    "NeuCLIR2023Retrieval",
    "MIRACLRetrieval",
    "RiaNewsRetrieval",
    "Quora-PL",
    "QuoraRetrieval",
]
not_include = ["MSMARCOv2"]

tasks_which_should_be_there = [
    t
    for t in tasks_which_should_be_there
    if t.metadata.name not in (retrieval_to_be_downsampled + not_include)
]

mteb_results = mteb.load_results()
models: list[mteb.ModelMeta] = [mteb.get_model_meta(name) for name in model_names]

# get missing revisions - Assuming we are using the latest revision
for model in models:
    if model.revision is None:
        print(f"Getting revision for {model.name}")
        encoder = model.load_model()
        model.revision = encoder.model_card_data.base_model_revision


mteb_results = filter_results(
    mteb_results, tasks=tasks_which_should_be_there, models=models
)

_mteb_results = normalize_results(mteb_results)
mteb_results = _mteb_results

# [t.task_name for t in mteb_results['GritLM/GritLM-7B']["13f00a0e36500c80ce12870ea513846a066004af"] if t.task_name == "SemRel24STS"]
# it is there

assert [
    len(revisions.keys()) == 1 for model, revisions in mteb_results.items()
], "Some models have more than one revision"

results_df = results_to_dataframe(mteb_results)

# results_df[results_df["task"] == "SemRel24STS"]  # still there

wide_table = results_df.pivot_table(
    index=["model", "revision"],
    columns=["task", "split", "hf_subset"],
    values="main_score",
)

# Find models which has NaN values
nans = wide_table[wide_table.isna().any(axis=1)]

# create list of model names x task names which is missing
t_names = {t.metadata.name for t in tasks_which_should_be_there}


sav_str = ""
for model in nans.index:
    print(f"Mising values for {model}:")
    sav_str += f"{model}:\n"
    nan_tasks = ""
    for task, split, lang in nans.columns:
        if task not in t_names:
            continue
        if task == nan_tasks:
            continue
        value = nans.loc[model, (task, split, lang)]
        if pd.isna(value):
            print(f"\t{task}")
            nan_tasks = task

            sav_str += f"\t{task}\n"


with open("missing_tasks.txt", "w") as f:
    f.write(sav_str)


# import mteb

# # running a model to ensure that the code works as expected
# task = mteb.get_task("FloresBitextMining")
# mdl = models[0].load_model()

# eval = mteb.MTEB(tasks = [task])
# results = eval.run(mdl)
# result = results[0]
# result.validate_and_filter_scores() # sucess


# len(set([s for s in task.metadata.hf_subsets_to_langscripts])) # 506

# # fetch from existing results
# grit_lm = [t for t in mteb_results["GritLM/GritLM-7B"]["13f00a0e36500c80ce12870ea513846a066004af"] if t.task_name == "FloresBitextMining"][0]

# grit_lm.validate_and_filter_scores() # fails
# # ValueError: Missing subsets {'ben_Beng-urd_Arab', 'sat_Olck-mal_Mlym', 'mar_Deva-snd_Deva', 'mar_Deva-san_Deva', 'mai_Deva-pan_Guru', 'sat_Olck-mni_Mtei', ...
# # KCE: have checked the results the keys are not present in the scores

# grit_lm.scores.keys() # ["test"]
# result.scores.keys() # ["test"]

# len(grit_lm.scores["devtest"]) # 20706
# len(result.scores["devtest"]) # 41412
