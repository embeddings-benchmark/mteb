"""
Usage: python mteb_to_tex.py results_folder_path
results_folder_path contains results of multiple models whose folders should be named after them
"""
import json
import os
import sys

from mteb import MTEB
import numpy as np

### GLOBAL VARIABLES ###
TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
]

TASK_LIST_SUMMARIZATION = [
    "SummEval",
]

TASK_LIST = (
    TASK_LIST_BITEXT
    + TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
    + TASK_LIST_SUMMARIZATION
)

TASK_LIST_EN = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
    + TASK_LIST_SUMMARIZATION
)

# Optionally custom select models; else will add all models in the results folder
MODELS = []

# Optionally map results folder name to a different name in the table
MODEL_TO_NAME = {
    "gritlm-7b": "GritLM-7B",
}

### LOGIC ###
results_folder = sys.argv[1].rstrip("/")

all_results = {}

mteb_task_names = [t.metadata.name for t in MTEB().tasks] + ["CQADupstackRetrieval"]

for model_name in os.listdir(results_folder):
    model_res_folder = os.path.join(results_folder, model_name)
    if os.path.isdir(model_res_folder):
        all_results.setdefault(model_name, {})
        for file_name in os.listdir(model_res_folder):
            if not file_name.split(".")[0].split("/")[-1] in mteb_task_names:
                print(f"Skipping non-MTEB file: {file_name}")
                continue
            print(f"Parsing MTEB file: {model_name}/{file_name}")
            with open(os.path.join(model_res_folder, file_name), "r", encoding="utf-8") as f:
                results = json.load(f)
                all_results[model_name] = {**all_results[model_name], **{file_name.replace(".json", ""): results}}

if len(MODELS) == 0:
  MODELS = list(all_results.keys())

def get_rows(dataset, model_name, limit_langs=[], skip_langs=[]):
    rows = []
    # CQADupstackRetrieval uses the same metric as its subsets
    tasks = MTEB(tasks=[dataset.replace("CQADupstackRetrieval", "CQADupstackTexRetrieval")]).tasks
    assert len(tasks) == 1, f"Found {len(tasks)} for {dataset}. Expected 1."
    main_metric = tasks[0].metadata.main_score
    test_result = all_results.get(model_name, {}). get(dataset, {})

    # Dev / Val set is used for MSMARCO (See BEIR paper)
    if "MSMARCO" in dataset:
        test_result = (
            test_result.get("dev") if "dev" in test_result else test_result.get("validation")
        )
    else:
        test_result = test_result.get("test")

    for lang in tasks[0].metadata.eval_langs:
        if (limit_langs and lang not in limit_langs) or (skip_langs and lang in skip_langs):
            continue
        elif test_result is None:
            rows.append([lang, main_metric, None])
            continue

        test_result_lang = test_result.get(lang, test_result)
        if main_metric == "cosine_spearman":
            test_result_lang = test_result_lang.get("cos_sim", {}).get("spearman")
        elif main_metric == "ap":
            test_result_lang = test_result_lang.get("cos_sim", {}).get("ap")
        else:
            test_result_lang = test_result_lang.get(main_metric)

        if test_result_lang is None:
            rows.append([lang, main_metric, None])
            continue

        rows.append([lang, main_metric, test_result_lang])
    return rows


def get_table(models, task_list, limit_langs=[], skip_langs=[], name="table", no_lang_col=False):
    TABLE = "Dataset & Language & " + " & ".join([MODEL_TO_NAME.get(model, model) for model in models]) + " \\\\" + "\n"
    if no_lang_col:
        TABLE = TABLE.replace("Language & ", "")
    scores_all = []
    for ds in task_list:
        try:
            results =  [get_rows(dataset=ds, model_name=model, limit_langs=limit_langs, skip_langs=skip_langs) for model in models]
            assert all(len(sub) == len(results[0]) for sub in results)
            for lang_idx in range(len(results[0])):
                scores = [x[lang_idx][-1] for x in results]
                scores_all.append(scores)
                lang = results[0][lang_idx][0]
                beginning = [ds, lang] if not(no_lang_col) else [ds]
                one_line = " & ".join(beginning + [str(round(x*100, 2)) if x is not None else "" for x in scores])
                TABLE += one_line + " \\\\" + "\n"
        except Exception as e:
            print(f"Skipping {ds} due to {e}")


    arr = np.array(scores_all, dtype=np.float32)
    # Get an index of columns which has any NaN value
    index = np.isnan(arr).any(axis=0)
    # Delete columns (models) with any NaN value from 2D NumPy Array
    arr = np.delete(arr, index, axis=1)
    # Average
    scores_avg = list(np.mean(arr, axis=0))
    # Insert empty string for NaN columns
    for i, val in enumerate(index):
        if val == True:
            scores_avg.insert(i, "")
    lang = "mix" if not(limit_langs) else limit_langs[0]
    beginning = ["Average", lang] if not(no_lang_col) else ["Average"]
    TABLE += " & ".join(beginning + [str(round(x*100, 2)) if x else "" for x in scores_avg]) + " \\\\" + "\n"

    with open(f"{name}.txt", "w") as f:
        f.write(TABLE)

get_table(MODELS, TASK_LIST_CLASSIFICATION, limit_langs=["en", "en-en",], name="mteb_clf", no_lang_col=True)
get_table(MODELS, TASK_LIST_CLUSTERING, limit_langs=["en", "en-en",], name="mteb_clu", no_lang_col=True)
get_table(MODELS, TASK_LIST_PAIR_CLASSIFICATION, limit_langs=["en", "en-en",], name="mteb_pclf", no_lang_col=True)
get_table(MODELS, TASK_LIST_RERANKING, limit_langs=["en", "en-en",], name="mteb_rrk", no_lang_col=True)
get_table(MODELS, TASK_LIST_RETRIEVAL, limit_langs=["en", "en-en",], name="mteb_rtr", no_lang_col=True)
get_table(MODELS, TASK_LIST_STS, limit_langs=["en", "en-en",], name="mteb_sts", no_lang_col=True)
get_table(MODELS, TASK_LIST_EN, limit_langs=["en", "en-en",], name="mteb_en", no_lang_col=True)
