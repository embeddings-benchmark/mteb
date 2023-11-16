import logging
import os
import json
from mteb.evaluation.MTEB import MTEB
import argparse

def main(results_folder):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model_name = results_folder.split("/")[-1]

    all_results = {}

    for file_name in os.listdir(results_folder):
        if not file_name.endswith(".json"):
            logger.info(f"Skipping non-json {file_name}")
            raise ValueError("This is not the proper folder. It does not contain the corresponding Json files.")
            continue
        with open(os.path.join(results_folder, file_name), "r", encoding="utf-8") as f:
            results = json.load(f)
            all_results = {**all_results, **{file_name.replace(".json", ""): results}}

    # Use "train" split instead
    TRAIN_SPLIT = ["DanishPoliticalCommentsClassification"]
    # Use "validation" split instead
    VALIDATION_SPLIT = ["AFQMC", "Cmnli", "IFlyTek", "TNews", "MSMARCO", "MultilingualSentiment", "Ocnli"]
    # Use "dev" split instead
    DEV_SPLIT = ["CmedqaRetrieval", "CovidRetrieval", "DuRetrieval", "EcomRetrieval", "MedicalRetrieval", "MMarcoReranking", "MMarcoRetrieval", "MSMARCO", "T2Reranking", "T2Retrieval", "VideoRetrieval"]

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
            MTEB(tasks=[ds_name.replace("CQADupstackRetrieval", "CQADupstackAndroidRetrieval")]).tasks[0].description
        )
        hf_hub_name = mteb_desc.get("hf_hub_name", mteb_desc.get("beir_name"))
        if "CQADupstack" in ds_name:
            hf_hub_name = "BeIR/cqadupstack"
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
        for lang in mteb_desc["eval_langs"]:
            mteb_name = f"MTEB {ds_name}"
            mteb_name += f" ({lang})" if len(mteb_desc["eval_langs"]) > 1 else ""
            # For English there is no language key if it's the only language
            test_result_lang = res_dict.get(lang) if len(mteb_desc["eval_langs"]) > 1 else res_dict
            # Skip if the language was not found but it has other languages
            if test_result_lang is None:
                continue
            META_STRING += "\n" + ONE_TASK.format(
                mteb_type, hf_hub_name, mteb_name, lang if len(mteb_desc["eval_langs"]) > 1 else "default", split, revision
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
    if os.path.exists(f"./mteb_metadata.yaml"):
        logger.warning("Overwriting mteb_metadata.md")
    with open(f"./mteb_metadata_{model_name}.yaml", "w") as f:
        f.write(META_STRING)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Select the model that you want to add to the Leaderboard.")
    
    parser.add_argument("--output_folder", type=str, required=True, help = "Select the folder in which the results are stored.")
    args = parser.parse_args()
    

    main(args.output_folder)