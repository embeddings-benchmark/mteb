from __future__ import annotations

import argparse
import mteb
from mteb import MTEB
import logging
import json
import os
from collections import defaultdict
from run_mteb_no_sub import c1Reranker
from prompts import get_prompt, PROMPT_DICT, validate_json, BEIR_DATASETS
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def get_safe_folder_name(model_name):
    return model_name.replace("/", "_").replace("\\", "_")

def run_evaluation(dataset_name: str, subtask: str, checkpoint, num_gpus: int, skip_prompt: bool) -> None:
    """Run MTEB evaluation for specified model and dataset

    Args:
        dataset_name: Name of MTEB dataset to evaluate
    """
    # Load model

    # Initialize MTEB task and evaluation
    tasks = mteb.get_tasks(tasks=[dataset_name])
    evaluation = MTEB(tasks=tasks)

    if dataset_name == "BrightRetrieval":
        previous_results = f"/home/oweller2/my_scratch/BRIGHT/outputs/{subtask}_bm25_long_False/score.json"
        eval_splits = ["standard"]
    elif dataset_name == "mFollowIRCrossLingual":
        previous_results = f"/home/oweller2/my_scratch/mteb/results/promptriever/mFollowIRCrossLingual_{subtask}_predictions.json"
        eval_splits = None
    elif dataset_name == "mFollowIR":
        previous_results = f"/home/oweller2/my_scratch/mteb/results/promptriever/mFollowIR_{subtask}_predictions.json"
        eval_splits = None
    else:
        print(f"Running with no subtask or eval splits for dataset: {dataset_name}")
        raise ValueError(f"Dataset {dataset_name} not supported")
        
    
    encode_kwargs = {
        "batch_size": 999999,
    }

    prompt = get_prompt(dataset_name, subtask)
    if prompt is not None and not skip_prompt:
        is_prompted = True
        # as string prompt hash
        prompt_hash = "_" + str(hashlib.sha256(prompt.encode()).hexdigest())
    else:
        is_prompted = False
        prompt = None
        prompt_hash = ""
    print(f"Prompt: {prompt}")

    if previous_results is not None:
        assert validate_json(previous_results), f"Previous results are not valid json: {previous_results}"
    print(f"Previous results: {previous_results}")

    model = c1Reranker(model_name_or_path=checkpoint.strip(), num_gpus=num_gpus, dataset_prompt=prompt)
    output_dir = f"results/{checkpoint}/{dataset_name}_{subtask}{prompt_hash}"
    print(f"Output directory: {output_dir}")

    # Run evaluation
    evaluation.run(
        model,
        save_predictions=True,
        encode_kwargs=encode_kwargs,
        output_folder=output_dir,
        previous_results=previous_results,
        eval_subsets=[subtask],
        eval_splits=eval_splits,
        top_k=100
    )


def main():
    parser = argparse.ArgumentParser(description="Run MTEB evaluation")
    parser.add_argument("-d", "--dataset", required=True, help="MTEB dataset name")
    parser.add_argument("-s", "--subtask", required=True, help="MTEB subtask name")
    parser.add_argument("-c", "--checkpoint", required=True, help="Checkpoint number")
    parser.add_argument("-n", "--num_gpus", required=False, help="Number of GPUs", default=1)
    parser.add_argument("-p", "--skip_prompt", action="store_true", help="Skip prompt")
    args = parser.parse_args()
    run_evaluation(args.dataset.strip(), args.subtask.strip(), args.checkpoint.strip(), args.num_gpus.strip(), args.skip_prompt)


if __name__ == "__main__":
    # add VLLM_ALLOW_LONG_MAX_MODEL_LEN
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    main()
