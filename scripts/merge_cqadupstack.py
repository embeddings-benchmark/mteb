"""Merges CQADupstack subset results
Usage: python merge_cqadupstack.py path_to_results_folder
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK_LIST_CQA = [
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
]

NOAVG_KEYS = [
    "evaluation_time",
    "mteb_version",
    "mteb_dataset_name",
    "dataset_revision",
]


results_folder = sys.argv[1]
# Ensure at least 1 character btw CQADupstack & Retrieval
files = glob.glob(f'{results_folder.rstrip("/")}/CQADupstack*?*Retrieval.json')

logger.info(f"Found CQADupstack files: {files}")

if len(files) == len(TASK_LIST_CQA):
    all_results = {}
    for file_name in files:
        with open(file_name, "r", encoding="utf-8") as f:
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
                        score = all_results[split][metric] + score * 1 / len(
                            TASK_LIST_CQA
                        )
                    all_results[split][metric] = score
    all_results["mteb_dataset_name"] = "CQADupstackRetrieval"

    logger.info("Saving ", all_results)
    with open(
        os.path.join(results_folder, "CQADupstackRetrieval.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(all_results, f, indent=4)
else:
    logger.warning(
        f"Got {len(files)}, but expected {len(TASK_LIST_CQA)} files. Missing: {set(TASK_LIST_CQA) - set([x.split('/')[-1].split('.')[0] for x in files])}; Too much: {set([x.split('/')[-1].split('.')[0] for x in files]) - set(TASK_LIST_CQA)}"
    )
