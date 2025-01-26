from __future__ import annotations

from typing import Any, Callable

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from mteb.load_results.benchmark_results import BenchmarkResults

MODEL_NAME = str
REVISION = str
METRIC = Callable[[list[float], list[float]], float]


def spearman(x: list[float], y: list[float]) -> float:
    return spearmanr(x, y)[0]


def pearson(x: list[float], y: list[float]) -> float:
    return pearsonr(x, y)[0]


def mse_with_zscore(x: list[float], y: list[float]) -> float:
    # using StandardScaler
    # fit on x and transform x and y
    scaler = StandardScaler()
    x_z = scaler.fit_transform(pd.DataFrame(x))
    y_z = scaler.transform(pd.DataFrame(y))
    return float(mean_squared_error(x_z, y_z))


def results_to_dataframe(
    mteb_results: BenchmarkResults,
    drop_na: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Convert the results of the MTEB evaluation to a pandas DataFrame.

    Args:
        mteb_results: The results of the MTEB evaluation.
        drop_na: Whether to drop missing values from the DataFrame.
        **kwargs: Additional keyword arguments to be passed to the `get_score` method of the `MTEBResults` class.
    """
    data = []
    for model_res in mteb_results:
        for task_result in model_res.task_results:
            data.append(
                {
                    "Model": model_res.model_name,
                    "Revision": model_res.model_revision,
                    "task": task_result.task_name,
                    "main_score": task_result.get_score(**kwargs),
                }
            )
    df = pd.DataFrame(data)

    if drop_na:
        df = df.dropna(axis=1)
    return df.pivot_table(
        index=["Model", "Revision"],
        columns=["task"],
        values="main_score",
    )


def leave_one_task_out(
    results_df: pd.DataFrame, sklearn_estimator: BaseEstimator
) -> pd.DataFrame:
    """Predicts the performance of a model on a task by training on all other tasks.

    Args:
        results_df: a DataFrame with one column for each task.
        sklearn_estimator: a scikit-learn base estimator for predicting the score. The estimator must have a `fit` and `predict` method.

    Returns:
        a matrix of predictions for each model and task.
    """

    def _fit_predict(model_i, task, task_df, sklearn_estimator):
        X_train = task_df.drop([task], axis=1).drop(model_i)
        y_train = task_df[[task]].drop(model_i)
        sklearn_estimator.fit(X_train.values, y_train.values)
        X_test = task_df.drop(columns=[task]).loc[model_i]
        y_pred = sklearn_estimator.predict(X_test.values.reshape(1, -1))
        return float(y_pred.flatten()[0])

    predictions = pd.DataFrame(columns=results_df.columns)

    columns_tqdm = tqdm(results_df.columns)
    for task in columns_tqdm:
        columns_tqdm.set_description(f"Task: {task}")
        task_predictions = []
        for model_i in results_df.index:
            task_predictions.append(
                _fit_predict(model_i, task, results_df, sklearn_estimator)
            )
        predictions[task] = list(task_predictions)

    return predictions


def most_predictable_task(
    results_df: pd.DataFrame,
    sklearn_estimator: BaseEstimator = LinearRegression(),
    metrics: list[METRIC] = [spearman, mse_with_zscore, pearson],
) -> list[dict[str, dict[str, float]]]:
    """Calculates the most predictable task

    Args:
        results_df: a DataFrame with one column for each task.
        sklearn_estimator: a scikit-learn base estimator for predicting the score. The estimator must have a `fit` and `predict` method.
        metrics: a list of functions to evaluate the performance of the model.

    Returns:
        a list of dictionaries on the form [{"task": {"metric": value}},...], sorted by how predictable the task is using the first metric in the list.
    """
    predictions = leave_one_task_out(results_df, sklearn_estimator)

    most_pred_tasks = []
    for task in results_df.columns:
        task_predictions = predictions[task]
        task_scores = results_df[task]

        task_results = {}
        for metric in metrics:
            task_results[metric.__name__] = metric(task_scores, task_predictions)  # type: ignore

        most_pred_tasks.append({task: task_results})

    # sort according to the first metric
    metric_name = metrics[0].__name__

    most_pred_tasks.sort(key=lambda x: list(x.values())[0][metric_name], reverse=True)

    return most_pred_tasks
