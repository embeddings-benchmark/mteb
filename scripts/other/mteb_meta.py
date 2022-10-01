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
    metrics:
    - type: accuracy
      value: 84.49350649350649
---
"""

import io
import json
import os
import sys

from mteb import MTEB

results_folder = sys.argv[1].strip("/")
model_name = results_folder.split("/")[-1]

all_results = {}

for file_name in os.listdir(results_folder):
    if not file_name.endswith(".json"):
        print(f"Skipping non-json {file_name}")
        continue
    with io.open(os.path.join(results_folder, file_name), "r", encoding="utf-8") as f:
        results = json.load(f)
        all_results = {**all_results, **{file_name.replace(".json", ""): results}}

MARKER = "---"
TAGS = "tags:"
MTEB_TAG = "- mteb"
HEADER = "model-index:"
MODEL = f"- name: {model_name}"
RES = "  results:"

META_STRING = "\n".join([MARKER, TAGS, MTEB_TAG, HEADER, MODEL, RES])


ONE_TASK = "  - task:\n      type: {}\n    dataset:\n      type: {}\n      name: {}\n      config: {}\n      split: {}\n    metrics:"
ONE_METRIC = "    - type: {}\n      value: {}"
SKIP_KEYS = ["std", "evaluation_time", "main_score", "threshold"]

for ds_name, res_dict in sorted(all_results.items()):
    mteb_desc = (
        MTEB(tasks=[ds_name.replace("CQADupstackRetrieval", "CQADupstackAndroidRetrieval")])
        .tasks[0]
        .description
    )
    hf_hub_name = mteb_desc.get("hf_hub_name", mteb_desc.get("beir_name"))
    if "CQADupstack" in ds_name:
        hf_hub_name = "BeIR/cqadupstack"
    mteb_type = mteb_desc.get("type")
    split = "test"
    if ds_name == "MSMARCO":
        split = "dev" if "dev" in res_dict else "validation"
    if split not in res_dict:
        print(f"Skipping {ds_name} as split {split} not present.")
        continue
    res_dict = res_dict.get(split)
    for lang in mteb_desc["eval_langs"]:
        mteb_name = f"MTEB {ds_name}"
        mteb_name += f" ({lang})" if len(mteb_desc["eval_langs"]) > 1 else ""
        # For English there is no language key if it's the only language
        test_result_lang = res_dict.get(lang) if len(mteb_desc["eval_langs"]) > 1 else res_dict
        # Skip if the language was not found but it has other languages
        if test_result_lang is None:
            continue
        META_STRING += "\n" + ONE_TASK.format(
            mteb_type,
            hf_hub_name,
            mteb_name,
            lang if len(mteb_desc["eval_langs"]) > 1 else "default",
            split
        )
        for (metric, score) in test_result_lang.items():
            if not isinstance(score, dict):
                score = {metric: score}
            for sub_metric, sub_score in score.items():
                if any([x in sub_metric for x in SKIP_KEYS]):
                    continue
                META_STRING += "\n" + ONE_METRIC.format(
                    f"{metric}_{sub_metric}" if metric != sub_metric else metric,
                    # All MTEB scores are 0-1, multiply them by 100 for reasons:
                    # 1) It's easier to visually digest (You need two chars less: "0.1" -> "1")
                    # 2) Others may multiply them by 100, when building on MTEB making it confusing what the range is
                    # This happend with Text and Code Embeddings paper (OpenAI) vs original BEIR paper
                    # 3) It's accepted practice (SuperGLUE, GLUE are 0-100)
                    sub_score * 100,
                )

META_STRING += "\n" + MARKER
with open("./mteb_metadata.md", "w") as f:
    f.write(META_STRING)
