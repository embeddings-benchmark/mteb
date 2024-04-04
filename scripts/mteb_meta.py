"""
Usage: python mteb_meta.py path_to_results_folder

Creates evaluation results metadata for the model card.
E.g.
---
tags:
- mteb
model-index:
- name: SGPT-5.8B-weightedmean-msmarco-specb-bitfit
  results:
  - task:
      type: classification
    dataset:
      type: mteb/banking77
      name: MTEB Banking77
      config: default
      split: test
      revision: 44fa15921b4c889113cc5df03dd4901b49161ab7
    metrics:
    - type: accuracy
      value: 84.49350649350649
---
"""

from __future__ import annotations

import json
import logging
import os
import sys

from mteb import MTEB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


results_folder = sys.argv[1].rstrip("/")
model_name = results_folder.split("/")[-1]

all_results = {}

for file_name in os.listdir(results_folder):
    if not file_name.endswith(".json"):
        logger.info(f"Skipping non-json {file_name}")
        continue
    with open(os.path.join(results_folder, file_name), "r", encoding="utf-8") as f:
        results = json.load(f)
        all_results = {**all_results, **{file_name.replace(".json", ""): results}}

# Use "train" split instead
TRAIN_SPLIT = ["DanishPoliticalCommentsClassification"]
# Use "validation" split instead
VALIDATION_SPLIT = [
    "AFQMC",
    "Cmnli",
    "IFlyTek",
    "TNews",
    "MSMARCO",
    "MultilingualSentiment",
    "Ocnli",
]
# Use "dev" split instead
DEV_SPLIT = [
    "CmedqaRetrieval",
    "CovidRetrieval",
    "DuRetrieval",
    "EcomRetrieval",
    "MedicalRetrieval",
    "MMarcoReranking",
    "MMarcoRetrieval",
    "MSMARCO",
    "T2Reranking",
    "T2Retrieval",
    "VideoRetrieval",
]

MARKER = "---"
TAGS = "tags:"
MTEB_TAG = "- mteb"
HEADER = "model-index:"
MODEL = f"- name: {model_name}"
RES = "  results:"

META_STRING = "\n".join([MARKER, TAGS, MTEB_TAG, HEADER, MODEL, RES])


ONE_TASK = "  - task:\n      type: {}\n    dataset:\n      type: {}\n      name: {}\n      config: {}\n      split: {}\n      revision: {}\n    metrics:"
ONE_METRIC = "    - type: {}\n      value: {}"
SKIP_KEYS = ["std", "evaluation_time", "main_score", "threshold"]

for ds_name, res_dict in sorted(all_results.items()):
    mteb_desc = (
        MTEB(
            tasks=[
                ds_name.replace("CQADupstackRetrieval", "CQADupstackAndroidRetrieval")
            ]
        )
        .tasks[0]
        .metadata_dict
    )
    hf_hub_name = (
        mteb_desc["dataset"]["path"]
        if "dataset" in mteb_desc
        else mteb_desc.get("beir_name")
    )
    if "beir_name" in mteb_desc:
        logger.warning(
            "`beir_name` is deprecated and will be removed in the future. New result files contain `hf_hub_name` instead."
        )
    if ds_name == "CQADupstackRetrieval" in ds_name:
        hf_hub_name = "mteb/cqadupstack"
    mteb_type = mteb_desc["type"]
    revision = res_dict.get("dataset_revision")  # Okay if it's None
    split = "test"
    if (ds_name in TRAIN_SPLIT) and ("train" in res_dict):
        split = "train"
    elif (ds_name in VALIDATION_SPLIT) and ("validation" in res_dict):
        split = "validation"
    elif (ds_name in DEV_SPLIT) and ("dev" in res_dict):
        split = "dev"
    elif "test" not in res_dict:
        logger.info(f"Skipping {ds_name} as split {split} not present.")
        continue
    res_dict = res_dict.get(split)

    has_langs = any([x in res_dict for x in mteb_desc["eval_langs"]])
    langs = mteb_desc["eval_langs"] if has_langs else ["default"]
    for lang in langs:
        mteb_name = f"MTEB {ds_name}"
        mteb_name += f" ({lang})" if has_langs else ""
        # For English there is no language key if it's the only language
        test_result_lang = res_dict.get(lang) if has_langs else res_dict
        # Skip if the language was not found but it has other languages
        if test_result_lang is None:
            logger.info(f"Skipping {ds_name} as {lang} not present.")
            continue
        META_STRING += "\n" + ONE_TASK.format(
            mteb_type, hf_hub_name, mteb_name, lang, split, revision
        )
        for metric, score in test_result_lang.items():
            if not isinstance(score, dict):
                score = {metric: score}
            for sub_metric, sub_score in score.items():
                if any([x in sub_metric for x in SKIP_KEYS]):
                    continue
                META_STRING += "\n" + ONE_METRIC.format(
                    f"{metric}_{sub_metric}" if metric != sub_metric else metric,
                    # All MTEB scores are 0-1, multiply them by 100 for 3 reasons:
                    # 1) It's easier to visually digest (You need two chars less: "0.1" -> "1")
                    # 2) Others may multiply them by 100, when building on MTEB making it confusing what the range is
                    # This happend with Text and Code Embeddings paper (OpenAI) vs original BEIR paper
                    # 3) It's accepted practice (SuperGLUE, GLUE are 0-100)
                    sub_score * 100,
                )

META_STRING += "\n" + MARKER
if os.path.exists("./mteb_metadata.md"):
    logger.warning("Overwriting mteb_metadata.md")
with open("./mteb_metadata.md", "w") as f:
    f.write(META_STRING)
