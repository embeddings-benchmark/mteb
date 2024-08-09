from __future__ import annotations

from importlib import reload
from typing import Iterable

import task_selector

import mteb

reload(task_selector)


eu_languages = [
    # official EU languages (56) - we could include the whole economic area e.g. Norway - additioanlly we could include minority languages (probably a good idea?)
    # germanic
    "dan",
    "eng",
    "deu",
    "nld",
    "swe",
    # romance
    "fra",
    "ita",
    "por",
    "spa",
    "ron",
    # slavic
    "bul",
    "hrv",
    "ces",
    "pol",
    "slk",
    "slv",
    # baltic
    "lav",
    "lit",
    "est",
    # finno-ugric
    "fin",
    "hun",
    # other indo european
    "ell",
    # non-indo european
    "mlt",
    "gle",
    # Shengen Area
    "nno",
    "nob",
    "isl",
    "ron",
    "eus",  # Basque - recognized minority language
    "ron",  # Romanian - recognized minority language
]


def get_eu_tasks():
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
        ],
        languages=eu_languages,
    )

    retrieval_to_be_downsampled = [  # TODO: Removing this list when tasks are ready
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

    return tasks_which_should_be_there


def get_models():
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
    models: list[mteb.ModelMeta] = [mteb.get_model_meta(name) for name in model_names]

    # get missing revisions - Assuming we are using the latest revision
    for model in models:
        if model.revision is None:
            print(f"Getting revision for {model.name}")
            encoder = model.load_model()
            model.revision = encoder.model_card_data.base_model_revision  # type: ignore

    return models


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
                except Exception:  # expected for older results
                    print(
                        f"Error validating and filtering scores for {model_name} {rev} {task_result.task_name}. Some splits are missing"
                    )
    return results


def is_candidate_valid_removal(current_tasks: list[str], task_to_remove: str):
    _current_tasks = current_tasks.copy()
    if task_to_remove in _current_tasks:
        _current_tasks.remove(task_to_remove)
    task = mteb.get_task(task_to_remove)
    ctasks = mteb.get_tasks(tasks=_current_tasks)

    # don't remove a unique task type
    task_types = {t.metadata.type for t in ctasks}
    if task.metadata.type not in task_types:
        return False
    return True


# Load data
tasks = get_eu_tasks()
models: list[ModelMeta] = get_models()
mteb_results = mteb.load_results()
mteb_results = filter_results(mteb_results, tasks=tasks, models=models)
mteb_results = normalize_results(mteb_results)

# Filter down to a sample language Danish
language = "dan"
lang_tasks = [t for t in tasks if language in t.metadata.languages]
lang_tasks_names = [t.metadata.name for t in lang_tasks]
mteb_results = filter_results(mteb_results, tasks=lang_tasks, models=models)

results_df = task_selector.results_to_dataframe(mteb_results, languages=["dan"])
lang_table = results_df.dropna(axis=1)

n_tasks = len(lang_table.columns)
if n_tasks != len(lang_table.columns):
    lang_tasks = [t for t in lang_tasks if t.metadata.name in lang_table.columns]
    missing_tasks = [t for t in lang_tasks if t.metadata.name not in lang_table.columns]
    print(f"Missing tasks for {language}: {missing_tasks}")  # should be empty


# Predict performance
most_pred_tasks = task_selector.most_predictable_task(lang_table)

