from __future__ import annotations

import argparse
import mteb
from mteb import MTEB
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_safe_folder_name(model_name):
    return model_name.replace("/", "_").replace("\\", "_")

def run_evaluation(dataset_name: str, subtask: str) -> None:
    """Run MTEB evaluation for specified model and dataset

    Args:
        dataset_name: Name of MTEB dataset to evaluate
    """
    # Load model
    model = mteb.get_model("bm25s")
    # model = mteb.get_model("intfloat/multilingual-e5-base")

    # Initialize MTEB task and evaluation
    tasks = mteb.get_tasks(tasks=[dataset_name])
    evaluation = MTEB(tasks=tasks)
    
    encode_kwargs = {
        "batch_size": 9999999999,
    }

    # Run evaluation
    evaluation.run(
        model,
        save_predictions=True,
        encode_kwargs=encode_kwargs,
        output_folder=f"results/bm25",
        # eval_subsets=[subtask] if subtask else None
    )


def main():
    parser = argparse.ArgumentParser(description="Run MTEB evaluation")
    parser.add_argument("-d", "--dataset", required=True, help="MTEB dataset name")
    parser.add_argument("-s", "--subtask", required=False, help="MTEB subtask name")

    args = parser.parse_args()
    run_evaluation(args.dataset, args.subtask)


if __name__ == "__main__":
    main()

    # for InstructIR
    # python run_mteb_bm25.py -d InstructIR
    # python run_mteb_bm25.py -d mFollowIRCrossLingual
    # python run_mteb_bm25.py -d mFollowIR

    # all BEIR ones
    # "ArguAna default"
    # "ClimateFEVER default"
    # "DBPedia default"
    # "FiQA2018 default"
    # "NFCorpus default"
    # "NQ default"
    # "SCIDOCS default"
    # "SciFact default"
    # "TRECCOVID default"
    # "Touche2020 default"

    