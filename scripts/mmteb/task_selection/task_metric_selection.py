import json
import os

# supress warnings
import warnings
from functools import reduce

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.repocard import metadata_load
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from yaml import safe_load

warnings.filterwarnings("ignore")

HF_TOKEN = "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"

MODEL_INFOS = {}


def get_leaderboard_df(results_path: str = "results.csv"):
    download_dir = snapshot_download(
        repo_id="mteb/leaderboard",
        repo_type="space",
    )

    MODEL_META_PATH = os.path.join(download_dir, "model_meta.yaml")
    with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
        MODEL_META = safe_load(f)

    LEADERBOARD_CONFIG_PATH = "config.yaml"
    with open(
        os.path.join(download_dir, LEADERBOARD_CONFIG_PATH), "r", encoding="utf-8"
    ) as f:
        LEADERBOARD_CONFIG = safe_load(f)

    with open(os.path.join(download_dir, "EXTERNAL_MODEL_RESULTS.json")) as f:
        EXTERNAL_MODEL_RESULTS = json.load(f)

    TASKS_CONFIG = LEADERBOARD_CONFIG["tasks"]
    BOARDS_CONFIG = LEADERBOARD_CONFIG["boards"]

    TASKS = list(TASKS_CONFIG.keys())

    TASK_TO_METRIC = {k: v["metric"] for k, v in TASKS_CONFIG.items()}

    TASK_DESCRIPTIONS = {k: v["task_description"] for k, v in TASKS_CONFIG.items()}
    TASK_DESCRIPTIONS["Overall"] = "Overall performance across MTEB tasks."
    MODELS_TO_SKIP = MODEL_META["models_to_skip"]

    TASK_TO_TASK_TYPE = {task_category: [] for task_category in TASKS}
    for board_config in BOARDS_CONFIG.values():
        for task_category, task_list in board_config["tasks"].items():
            TASK_TO_TASK_TYPE[task_category].extend(task_list)

    # model_list = os.listdir(os.path.join(download_dir, "results"))

    task_dict = BOARDS_CONFIG["en"]["tasks"]
    all_tasks = reduce(lambda x, y: x + y, task_dict.values())

    tasks = list(task_dict.keys())
    datasets = all_tasks

    # print(model_list)
    api = HfApi(token=HF_TOKEN)
    models = api.list_models(filter="mteb")

    i = 0
    df_list = []
    for model in EXTERNAL_MODEL_RESULTS:
        results_list = []
        for task in tasks:
            # Not all models have InstructionRetrieval, other new tasks
            if task not in EXTERNAL_MODEL_RESULTS[model]:
                continue
            results_list += EXTERNAL_MODEL_RESULTS[model][task][TASK_TO_METRIC[task]]

        res = {
            k: v
            for d in results_list
            for k, v in d.items()
            if (k == "Model") or any([x in k for x in datasets])
        }

        # <a target="_blank" style="text-decoration: underline" href="https://huggingface.co/in...al-7b-instruct">e5-mistral-7b-instruct</a>
        # extract the model name e5-mistral-7b-instruct from the value
        res["Model"] = res["Model"].split('">')[1].split("</a>")[0]
        if len(res) == len(datasets) + 1 and res.keys() == set(datasets) | {"Model"}:
            df_list.append(res)

    for model in models:
        if model.modelId in MODELS_TO_SKIP:
            continue
        i += 1
        print("MODEL", model.modelId)
        if model.modelId not in MODEL_INFOS:
            readme_path = hf_hub_download(model.modelId, filename="README.md")
            meta = metadata_load(readme_path)
            MODEL_INFOS[model.modelId] = {"metadata": meta}
        meta = MODEL_INFOS[model.modelId]["metadata"]
        if "model-index" not in meta:
            continue
        if len(datasets) > 0:
            task_results = [
                sub_res
                for sub_res in meta["model-index"][0]["results"]
                if (sub_res.get("task", {}).get("type", "") in tasks)
                and any(
                    [x in sub_res.get("dataset", {}).get("name", "") for x in datasets]
                )
            ]
        else:
            task_results = [
                sub_res
                for sub_res in meta["model-index"][0]["results"]
                if (sub_res.get("task", {}).get("type", "") in tasks)
            ]
        out = [
            {
                res["dataset"]["name"].replace("MTEB ", ""): [
                    round(score["value"], 2)
                    for score in res["metrics"]
                    if score["type"] == TASK_TO_METRIC.get(res["task"]["type"])
                ][0]
            }
            for res in task_results
        ]
        out = {k: v for d in out for k, v in d.items()}
        out["Model"] = model.modelId
        # only keep the models that have results for all tasks
        if len(out) == len(datasets) + 1 and out.keys() == set(datasets) | {"Model"}:
            df_list.append(out)

    df = pd.DataFrame(df_list)
    # add Overall column with average of all tasks, excluding the Model column
    df["Overall"] = df.iloc[:, 1:].mean(axis=1)
    # sort by Overall
    df = df.sort_values("Overall", ascending=False)
    # make Model column first, Overall column second
    cols = df.columns.tolist()
    cols = ["Model", "Overall"] + [
        col for col in cols if col not in ["Model", "Overall"]
    ]
    df = df[cols]

    df.to_csv(results_path, index=False)


def get_mse_scores(df):
    # remove overall column
    df_train = df.drop(columns=["Overall", "Model"])

    # make a new df for mse scores, same columns as df_train, with "Model" as index
    df_mse_models = pd.DataFrame(columns=df_train.columns)

    # apply linear regression, training data will leave one row out, y is on of the columns, X is the rest
    for i in tqdm(range(len(df_train))):
        for col in tqdm(df_train.columns):
            X_train, y_train = (
                df_train.drop(index=i).drop(columns=[col]),
                df_train.drop(index=i)[col],
            )
            X_test, y_test = df_train.drop(columns=[col]).iloc[i], df_train[col].iloc[i]

            reg = LinearRegression().fit(X_train, y_train)
            mse_score = mean_squared_error(
                reg.predict(X_test.values.reshape(1, -1)), [y_test]
            )

            df_mse_models.loc[i, col] = mse_score

    # take transpose of df_mse
    df_mse_tasks = df_mse_models.T

    df_mse_models["Avg. MSE"] = df_mse_models.mean(axis=1)
    df_mse_tasks["Avg. MSE"] = df_mse_tasks.mean(axis=1)

    # add Model column
    df_mse_models["Model"] = df["Model"]

    # make Model column first, Overall column second
    cols = df_mse_models.columns.tolist()
    cols = ["Model", "Avg. MSE"] + [
        col for col in cols if col not in ["Model", "Avg. MSE"]
    ]
    df_mse_models = df_mse_models[cols]
    # sort by Overall
    df_mse_models = df_mse_models.sort_values("Avg. MSE", ascending=True)

    df_mse_models.to_csv("mse_models_results.csv", index=False)

    # name index column to Task
    df_mse_tasks.index.name = "Task"

    # make Task column first, Overall column second
    cols = df_mse_tasks.columns.tolist()
    cols = ["Avg. MSE"] + [col for col in cols if col not in ["Task", "Avg. MSE"]]
    df_mse_tasks = df_mse_tasks[cols]

    # sort by Overall
    df_mse_tasks = df_mse_tasks.sort_values("Avg. MSE", ascending=True)

    df_mse_tasks.to_csv("mse_tasks_results.csv")


if __name__ == "__main__":
    if not os.path.exists("results.csv"):
        get_leaderboard_df()
    df = pd.read_csv("results.csv")

    if not os.path.exists("mse_models_results.csv") or not os.path.exists(
        "mse_tasks_results.csv"
    ):
        get_mse_scores(df)

    df_mse_models = pd.read_csv("mse_models_results.csv")
    df_mse_tasks = pd.read_csv("mse_tasks_results.csv")

    # remove index column
    df_mse_tasks = df_mse_tasks.set_index("Task")
    df_mse_models = df_mse_models.set_index("Model")

    print(df_mse_tasks.iloc[:10, [0]].to_markdown())
    print(df_mse_models.iloc[:10, [0]].to_markdown())
