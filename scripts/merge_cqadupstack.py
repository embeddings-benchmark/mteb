"""
Merges CQADupstack subset results
Usage: python merge_cqadupstack.py path_to_results_folder
"""
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK_LIST_CQA = [
    "CQADupstackAndroid",
    "CQADupstackEnglish",
    "CQADupstackGaming",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
]

NOAVG_KEYS = [
    "evaluation_time",
    "mteb_version",
    "mteb_dataset_name",
    "dataset_revision",
]

import os
import sys
import json
import glob

results_folder = sys.argv[1]
files = glob.glob(f'{results_folder.strip("/")}/CQADupstack*.json')

logger.info(f"Found CQADupstack files: {files}")

if len(files) == len(TASK_LIST_CQA):
    all_results = {}
    for file_name in files:
        with open(file_name, 'r', encoding='utf-8') as f:
            results = json.load(f)
            for split, split_results in results.items():
                if split not in ("train", "validation", "dev", "test"):
                    all_results[split] = split_results
                    continue
                all_results.setdefault(split, {})
                for metric, score in split_results.items():
                    all_results[split].setdefault(metric, 0)
                    if metric == "evaluation_time":
                        score = all_results[split][metric] + score
                    elif metric not in NOAVG_KEYS:
                        score = all_results[split][metric] + score * 1/len(TASK_LIST_CQA)
                    all_results[split][metric] = score

    logger.info("Saving ", all_results)
    with open(os.path.join(results_folder, "CQADupstackRetrieval.json"), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
else:
    logger.warning(f"Missing files {set(TASK_LIST_CQA) - set(files)} or got too many files.")
