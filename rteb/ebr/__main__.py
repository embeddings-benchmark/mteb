import argparse
from collections import defaultdict
import json
import logging
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

from ebr.retrieve import run_retrieve_task
from ebr.datasets import DatasetMeta, DATASET_REGISTRY
from ebr.models import ModelMeta, MODEL_REGISTRY
from ebr.core import Encoder, Retriever


logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Evaluation
    parser.add_argument(
        "--gpus", type=int, default=0, help="Number of gpus used for encoding.")
    parser.add_argument(
        "--cpus", type=int, default=1, help="Number of cpus used for computation (this is only for models that are not using gpus).")
    parser.add_argument(
        "--bf16", action="store_true", help="`Use bf16 precision.")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for encoding.")
    parser.add_argument(
        "--embd_batch_size", type=int, default=1024, help="Batch size for computing similarity of embeddings.")
    parser.add_argument(
        "--embd_in_memory_threshold", type=int, default=200000,
        help="Embeddings will be stored in memory if the amount is below this threshold.")

    # Model
    #parser.add_argument(
    #    "--model_name", type=str, default=None, help="Model name or path.")
    #parser.add_argument(
    #    "--embd_dtype", type=str, default="float", help="Embedding type. Options: float32, int8, binary.")
    #parser.add_argument(
    #    "--embd_dim", type=int, default=None, help="Embedding dimension.")
    #parser.add_argument(
    #    "--max_length", type=int, default=None, help="Maximum length of model input.")

    # Data
    parser.add_argument(
        "--data_path", type=str, default="data/", help="Path of the dataset, must be specified for custom tasks.")
    parser.add_argument(
        "--task_name", type=str, default=None, help="Name of the task. Can be multiple tasks splitted by `,`.")
    parser.add_argument(
        "--data_type", default="eval", choices=["eval", "train", "chunk", "merge"], help="Dataset type.")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloader.")
    
    # Output
    parser.add_argument(
        "--save_path", type=str, default="output/", help="Path to save the output.")
    parser.add_argument(
        "--save_embds", action="store_true", help="Whether to save the embeddings.")
    parser.add_argument(
        "--load_embds", action="store_true", help="Whether to load the computed embeddings.")
    parser.add_argument(
        "--save_prediction", action="store_true", help="Whether to save the predictions.")
    parser.add_argument(
        "--topk", type=int, default=100, help="Number of top documents per query.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Whether to overwrite the results.")
    
    args = parser.parse_args()
    return args


def _dump_model_meta(
    results_dir: str = "results",
    model_registry: dict[str, ModelMeta] = MODEL_REGISTRY,
):
    models = [meta.model_dump() for meta in model_registry.values()]
    with open(Path(results_dir) / "models.json", "w") as f:
        f.write(json.dumps(models, indent=4))

def _dump_dataset_info(
    results_dir: str = "results",
    dataset_registry: dict[str, DatasetMeta] = DATASET_REGISTRY,
):
    group_data = defaultdict(list)
    for dataset_meta in dataset_registry.values():
        for group_name in dataset_meta.groups.keys():
            leaderboard = dataset_meta.loader.LEADERBOARD
            group_data[(leaderboard, group_name)].append(dataset_meta.dataset_name)

    groups = []
    for (leaderboard, group_name), datasets in group_data.items():
        groups.append({"name": group_name, "datasets": datasets, "leaderboard": leaderboard})
    with open(Path(results_dir) / "datasets.json", "w") as f:
        f.write(json.dumps(groups, indent=4))


def _compile_results(
    results_dir: str = "results",
    output_dir: str = "output"
):
    results = []
    for dataset_output_dir in Path(output_dir).iterdir():

        dataset_results = []
        for one_result in dataset_output_dir.iterdir():

            eval_file = one_result / "retrieve_eval.json"
            if eval_file.exists():
                with open(eval_file) as f:
                    dataset_results.append(json.load(f))

        results.append({
            **DATASET_REGISTRY[dataset_output_dir.name].model_dump(),
            "results": dataset_results,
            "is_closed": DATASET_REGISTRY[dataset_output_dir.name].tier != 3
        })

    with open(Path(results_dir) / "results.json", "w") as f:
        f.write(json.dumps(results, indent=4))


def main(args: argparse.Namespace):

    _dump_model_meta()
    _dump_dataset_info()

    if args.gpus:
        trainer = pl.Trainer(
            strategy=DDPStrategy(find_unused_parameters=False),
            accelerator="gpu",
            devices=args.gpus,
            precision="bf16" if args.bf16 else "32",
        )
    else:
        trainer = pl.Trainer(
            strategy=DDPStrategy(),
            accelerator="cpu",
            devices=args.cpus,
        )

    if not trainer.is_global_zero:
        logging.basicConfig(level=logging.ERROR)

    # Evaluate each model on the specified datasets
    for model_meta in MODEL_REGISTRY.values():

        encoder = Encoder(
            model_meta.load_model(),
            save_embds=args.save_embds,
            load_embds=args.load_embds
        )
        retriever = Retriever(
            topk=args.topk,
            similarity=model_meta.similarity,
            save_prediction=args.save_prediction
        )

        eval_results = {}
        for dataset_meta in DATASET_REGISTRY.values():

            #if trainer.is_global_zero:
            #    trainer.print(f"Evaluating {model_meta.model_name} on {dataset_meta.dataset_name}")

            result = run_retrieve_task(dataset_meta, trainer, encoder, retriever, args)
            eval_results[dataset_meta.dataset_name] = result
    
        metric = "ndcg_at_10"

        # Print the results
        if trainer.is_global_zero:
            trainer.print("=" * 40)
            trainer.print(args.save_path)
            trainer.print("=" * 40)
            for task in eval_results.keys():
                if metric in eval_results[task]:
                    trainer.print(f"{task:<32}{eval_results[task][metric]:.4f}")

    _compile_results()

if __name__ == "__main__":
    args = get_args()
    main(args)
