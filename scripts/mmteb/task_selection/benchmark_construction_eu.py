"""This script is the outline for construction the european benchmark.

It takes the following steps:

1. From all tasks select only the tasks that are in specified list (e.g. EU languages)
2. For each language
    2.1 select the tasks that are in the specified list
    2.2 for each model x task estimate the performance on the task by training on all other tasks x models (leave one task out). For this we use a linear regression model.
    2.3 calculate the correlation between the estimated performance and the actual performance
    2.4 remove the task with the highest (spearman) correlation (i.e. the most predictable task) - Additionally we impose a contraint that the task must be valid to remove (i.e. all languages are covered for at least one of the task types)
    2.5 repeat steps 2.2-2.4 until no tasks can be removed (or some other stopping criteria)
3. Save the results to a json file

There is two main assumptions here:

1) Performance on a task can be estimated linearly (additive) from the performance on other tasks and that this approach can be performed individually for each language.
2) The most predictable task is determined by the highest MSE (with z-score normalization) correlation between the estimated performance and the actual performance

The first assumption is a convenient simplification. We might end up selecting too many tasks as the performance on a task might not be predictable from the performance on tasks from another language.
The second assumption we compare changing out the MSE with spearman and pearson correlation. This is explored for danish and english (See files task_selection/figures_task_selection/{lang}_w_{metric}.png). MSE does in general
seem to maintain the correlation with the mean score across all tasks better than the other metrics - though there is no reason to assume that the mean across all tasks is a gold standard.ß
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr, zscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import mteb

MODEL = str
REVISION = str


def results_to_dataframe(
    mteb_results: dict[MODEL, dict[REVISION, list[mteb.MTEBResults]]],
    languages: list[str],
    split_and_hf_subset_as_columns: bool = False,
):
    data = []
    for model_name, revisions in mteb_results.items():
        for rev, tasks_results in revisions.items():
            if split_and_hf_subset_as_columns:
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
            else:
                for task_result in tasks_results:
                    data.append(
                        {
                            "model": model_name,
                            "revision": rev,
                            "task": task_result.task_name,
                            "main_score": task_result.get_score(languages=languages),
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


def _fit_predict(model_i, task, task_df, classifer):
    clf = classifer()
    X_train = task_df.drop([task], axis=1).drop(model_i)
    y_train = task_df[[task]].drop(model_i)
    clf.fit(X_train.values, y_train.values)
    X_test = task_df.drop(columns=[task]).loc[model_i]
    y_pred = clf.predict(X_test.values.reshape(1, -1))
    return float(y_pred.flatten()[0])


def leave_one_task_out(task_df: pd.DataFrame, classifer) -> pd.DataFrame:
    """Predicts the performance of a model on a task by training on all other tasks.

    Args:
        task_df: a DataFrame with one column for each task.
        classifer: a scikit-learn model that has a fit and predict method.

    Returns:
        a matrix of predictions for each model and task.
    """
    predictions = pd.DataFrame(columns=task_df.columns)
    # for task in df.columns:
    columns_tqdm = tqdm(task_df.columns)
    for task in columns_tqdm:
        columns_tqdm.set_description(f"Task: {task}")

        # model_i = task_df.index[0]
        # _fit_predict(model_i, task, task_df, classifer)

        # with Pool() as p:
        #     task_predictions = p.starmap(
        #         _fit_predict,
        #         [(model_i, task, task_df, classifer) for model_i in task_df.index],
        #     )

        # without multiprocessing
        task_predictions = []
        for model_i in task_df.index:
            task_predictions.append(_fit_predict(model_i, task, task_df, classifer))
        predictions[task] = list(task_predictions)

    return predictions


def spearman(x, y):
    return spearmanr(x, y)[0]


def pearson(x, y):
    return pearsonr(x, y)[0]


def mse_with_zscore(x, y):
    return mean_squared_error(zscore(x), zscore(y))


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


def calculate_task_scores(
    predictions: pd.DataFrame, observed_results: pd.DataFrame, metric=mse_with_zscore
) -> pd.DataFrame:
    """Calculate how well the predictions match the observed results.

    Args:
        predictions: a DataFrame with columns: Model, Overall, and one column for each task.
        observed_results: a DataFrame with columns: Model, Overall, and one column for each task.
        metric: a function that takes two lists of numbers and returns a single number.

    Returns:
        a DataFrame with a single row that contains the score for each task.
    """
    scores = {}
    for task in predictions.columns:
        if task not in ["Model", "Overall"]:
            scores[task] = metric(predictions[task], observed_results[task])
    return pd.DataFrame(scores, index=[0])


def save_results(
    selected_tasks: list[str],
    n_tasks: list[int],
    spearman_correalation_with_overall: list[float],
    pearson_correlation_with_overall: list[float],
    predictability_of_task_removed_spearman: list[float],
    predictability_of_task_removed_mse_with_zscore: list[float],
    task_removal_order: list[str],
    language: str,
    path: Path,
):
    json_obj = {
        "selected_tasks": selected_tasks,
        "n_tasks": n_tasks,
        "spearman_correalation_with_overall": spearman_correalation_with_overall,
        "pearson_correlation_with_overall": pearson_correlation_with_overall,
        "predictability_of_task_removed_spearman": predictability_of_task_removed_spearman,
        "predictability_of_task_removed_mse_with_zscore": predictability_of_task_removed_mse_with_zscore,
        "task_removal_order": task_removal_order,
        "language": language,
    }

    path.parent.mkdir(exist_ok=True, parents=True)

    with path.open("w") as f:
        json.dump(json_obj, f)

    create_task_selection_plot(
        n_tasks,
        spearman_correalation_with_overall,
        pearson_correlation_with_overall,
        task_removal_order,
        language,
    )

    create_predictability_plot(
        task_removal_order,
        predictability_of_task_removed_spearman,
        predictability_of_task_removed_mse_with_zscore,
        language,
    )


def create_predictability_plot(
    task_removal_order: list[str],
    predictability_of_task_removed_spearman: list[float],
    predictability_of_task_removed_mse_with_zscore: list[float],
    language: str,
):
    """Creates a plot showing the predictability of the task removed."""
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 7)
    y = predictability_of_task_removed_spearman
    x = task_removal_order[1:]
    ax.plot(x, y, label="spearman")
    y = predictability_of_task_removed_mse_with_zscore
    ax.plot(x, y, label="mse with zscore")
    ax.set_xlabel("Task Removed")
    ax.set_ylabel("Predictability of task removed")
    ax.legend()
    ax.set_title(f"Predictability of task removed for language={language}")
    plt.xticks(rotation=90)

    # save the plot
    path = Path(__file__).parent / "figures_task_selection"
    path.mkdir(exist_ok=True)
    save_path = path / f"task_selection_{language}_predictability.png"
    plt.tight_layout()
    # set y ticks to 0 1 with 0.1 steps
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    plt.savefig(save_path, dpi=500)


def create_task_selection_plot(
    n_tasks: list[int],
    spearman_correalation_with_overall: list[float],
    pearson_correlation_with_overall: list[float],
    task_removal_order: list[str],
    language: str,
):
    """Create a line plot showing the correlation with the mean score as tasks are removed."""
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 7)
    x_label = [
        f"{task_name} ({n_tasks} tasks)"
        for task_name, n_tasks in zip(task_removal_order, n_tasks)
    ]
    x_label[0] = f"All tasks ({n_tasks[0]})"

    # plot line for pearson and spearman
    ax.plot(x_label, spearman_correalation_with_overall, label="spearman")
    ax.plot(x_label, pearson_correlation_with_overall, label="pearson")

    ax.set_xlabel("Task Removed (number of tasks)")
    ax.set_ylabel("Correlation with mean score across all tasks")
    ax.legend()
    ax.set_title(f"Task selection for language={language}")
    plt.xticks(rotation=90)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    # save the plot
    path = Path(__file__).parent / "figures_task_selection"
    path.mkdir(exist_ok=True)
    save_path = path / f"task_selection_{language}.png"
    plt.tight_layout()

    plt.savefig(save_path, dpi=500)


if __name__ == "__main__":
    tasks = get_eu_tasks()
    models = get_models()
    mteb_results = mteb.load_results()
    mteb_results = filter_results(mteb_results, tasks=tasks, models=models)
    mteb_results = normalize_results(mteb_results)

    assert [
        len(revisions.keys()) == 1 for model, revisions in mteb_results.items()
    ], "Some models have more than one revision"

    locked_tasks = []  # tasks which are already included as a part of another language

    save_dir = Path(__file__).parent / "selected_tasks"

    for language in eu_languages:
        save_path = save_dir / f"{language}.json"
        if save_path.exists():
            print(f"Language {language} already processed, skipping")
            with save_path.open("r") as f:
                json_obj = json.load(f)
                locked_tasks = json_obj["selected_tasks"]
            continue

        lang_tasks = [t for t in tasks if language in t.metadata.languages]
        lang_tasks_names = [t.metadata.name for t in lang_tasks]
        lang_results = filter_results(mteb_results, tasks=lang_tasks)
        results_df = results_to_dataframe(mteb_results, languages=[language])
        lang_table = results_df.pivot_table(
            index=["model", "revision"],
            columns=["task"],
            values="main_score",
        )
        lang_table = lang_table.dropna(axis=1)
        n_tasks = len(lang_table.columns)
        if n_tasks != len(lang_table.columns):
            lang_tasks = [
                t for t in lang_tasks if t.metadata.name in lang_table.columns
            ]
            missing_tasks = [
                t for t in lang_tasks if t.metadata.name not in lang_table.columns
            ]
            print(f"Missing tasks for {language}: {missing_tasks}")

        has_tasks_to_remove = True
        tasks_to_select_from = [
            t.metadata.name for t in lang_tasks if t.metadata.name in lang_table.columns
        ]
        print(f"Starting task selection for {language}")
        n_tasks = [len(tasks_to_select_from)]
        spearman_correalation_with_overall = [1.0]
        pearson_correlation_with_overall = [1.0]
        predictability_of_task_removed_spearman = []
        predictability_of_task_removed_mse_with_zscore = []

        task_removal_order = [""]
        mean_scores = lang_table.mean(axis=1)

        while has_tasks_to_remove:
            performance_predictions = leave_one_task_out(lang_table, LinearRegression)

            # find the most predictable task
            task_scores = calculate_task_scores(
                performance_predictions,
                lang_table,
                spearman,
            )
            task_scores_mse = calculate_task_scores(
                performance_predictions,
                lang_table,
                mse_with_zscore,
            )
            most_pred_tasks = list(
                task_scores.T.sort_values(by=0, ascending=False).index  # type: ignore
            )
            print(
                f"Most predictable tasks: {task_scores.T.sort_values(by=0, ascending=False).head()}"
            )

            while most_pred_tasks:
                task_to_remove = most_pred_tasks.pop(0)
                is_cand_removal = (
                    is_candidate_valid_removal(
                        current_tasks=tasks_to_select_from,
                        task_to_remove=task_to_remove,
                    )
                    and task_to_remove not in locked_tasks
                )
                if is_cand_removal:
                    tasks_to_select_from.remove(task_to_remove)
                    lang_table = lang_table[tasks_to_select_from]
                    print(f"Removed {task_to_remove}")

                    n_tasks.append(len(tasks_to_select_from))
                    mean_new_scores = lang_table.mean(axis=1)
                    spearman_correalation_with_overall.append(
                        spearman(mean_scores, mean_new_scores)
                    )

                    pearson_correlation_with_overall.append(
                        pearson(mean_scores, mean_new_scores)
                    )

                    predictability_of_task_removed_spearman.append(
                        task_scores.T.sort_values(by=0, ascending=False).values[0][0]  # type: ignore
                    )
                    predictability_of_task_removed_mse_with_zscore.append(
                        task_scores_mse.T.sort_values(by=0, ascending=True).values[0][0]  # type: ignore
                    )

                    task_removal_order.append(task_to_remove)
                    break

            if not most_pred_tasks:
                has_tasks_to_remove = False

        save_results(
            tasks_to_select_from,
            n_tasks,
            spearman_correalation_with_overall,
            pearson_correlation_with_overall,
            predictability_of_task_removed_spearman,
            predictability_of_task_removed_mse_with_zscore,
            task_removal_order,
            language,
            save_path,
        )

        print(f"Finished task selection for {language}")
