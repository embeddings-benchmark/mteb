import argparse
from pathlib import Path
import os
import json

from beir.retrieval.evaluation import EvaluateRetrieval
import pytorch_lightning as pl
from termcolor import colored

from ebr.core import Encoder
from ebr.core.data import RetrieveDataModule
from ebr.core.meta import DatasetMeta


CORPUS_EMBD_FILENAME = "corpus_embds.jsonl"
QUERIES_EMBD_FILENAME = "queries_embds.jsonl"
RETRIEVE_EVAL_FILENAME = "retrieve_eval.json"
RETRIEVE_PRED_FILENAME = "retrieve_pred.json"


def run_retrieve_evaluation(relevance, prediction):
    if len(relevance) != len(prediction):
        raise RuntimeError("Prediction and ground truth have different sizes.")
    
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        relevance, prediction, k_values=[1,3,5,10,20,50,100], ignore_identical_ids=False
    )
    scores = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
    }
    return scores


def run_retrieve_task(
    dataset_meta: DatasetMeta,
    trainer: pl.Trainer,
    encoder: Encoder,
    retriever: pl.LightningModule,
    args: argparse.Namespace
):
    dataset_name = dataset_meta.dataset_name

    task_save_path = Path(args.save_path) / dataset_name / encoder.model._id
    task_save_path.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        eval_file = task_save_path / RETRIEVE_EVAL_FILENAME
        pred_file = task_save_path / RETRIEVE_PRED_FILENAME
        if eval_file.exists():
            with open(str(eval_file)) as f:
                scores = json.load(f)
            return scores
        else:
            if pred_file.exists():
                return

    # DataModule manages the datasets
    dataset_kwargs = {
        "query_instruct": encoder.model.query_instruct,
        "corpus_instruct": encoder.model.corpus_instruct
    }
    collator_kwargs = {}

    dm = RetrieveDataModule(
        data_path=args.data_path,
        dataset_name=dataset_name,
        batch_size=args.batch_size,
        embd_batch_size=args.embd_batch_size,
        num_workers=args.num_workers,
        dataset_kwargs=dataset_kwargs,
        collator_kwargs=collator_kwargs,
    )
    if trainer.is_global_zero:
        dm.prepare_data()
        trainer.print("Queries size:", len(dm.dataset.queries))
        trainer.print("Corpus size:", len(dm.dataset.corpus))
    
    trainer.strategy.barrier()

    if len(dm.dataset.queries) < trainer.num_devices or len(dm.dataset.corpus) < trainer.num_devices:
        trainer.print(colored("Skipping the task due to too few queries / documents.", "red"))
        return {}

    if len(dm.dataset.queries) >= 1e6:
        trainer.print(colored("Skipping the task due to too many queries.", "red"))
        return {}

    if dataset_name == "bm25":
        # Build the index from corpus
        retriever.build_index(dm.dataset.corpus)
        # Compute the scores for queries
        retriever.save_file = os.path.join(task_save_path, RETRIEVE_PRED_FILENAME)
        trainer.predict(model=retriever, dataloaders=dm.queries_dataloader())
    
    else:
        # Compute the query embeddings
        trainer.print(colored("Encode queries", "yellow"))
        encoder.is_query = True
        encoder.in_memory = (len(dm.dataset.queries) < args.embd_in_memory_threshold)
        encoder.save_file = os.path.join(task_save_path, QUERIES_EMBD_FILENAME)
        if args.load_embds and encoder.embd_files_exist(trainer.num_devices):
            queries_embds_files = encoder.get_embd_files(trainer.num_devices)
            trainer.print(f"Embedding files exist: {queries_embds_files}")
            dm.set_queries_embds(queries_embds_files=queries_embds_files)
        else:
            trainer.print(f"in_memory = {encoder.in_memory}")
            trainer.print(f"save_file = {encoder.save_file}")
            trainer.predict(model=encoder, dataloaders=dm.queries_dataloader())
            # Set the query embeddings
            queries_embds_files = encoder.get_embd_files()
            dm.set_queries_embds(queries_embds=encoder.embds, queries_embds_files=queries_embds_files)
        
        # Compute the corpus embeddings
        trainer.print(colored("Encode corpus", "yellow"))
        encoder.is_query = False
        encoder.save_file = os.path.join(task_save_path, CORPUS_EMBD_FILENAME)
        encoder.in_memory = (len(dm.dataset.corpus) < args.embd_in_memory_threshold)
        if args.load_embds and encoder.embd_files_exist(trainer.num_devices):
            corpus_embds_files = encoder.get_embd_files(trainer.num_devices)
            trainer.print(f"Embedding files exist: {corpus_embds_files}")
            dm.set_corpus_embds(corpus_embds_files=corpus_embds_files)
        else:
            trainer.print(f"in_memory = {encoder.in_memory}")
            trainer.print(f"save_file = {encoder.save_file}")
            trainer.predict(model=encoder, dataloaders=dm.corpus_dataloader())
            # Set the corpus embeddings
            corpus_embds_files = encoder.get_embd_files()
            dm.set_corpus_embds(corpus_embds=encoder.embds, corpus_embds_files=corpus_embds_files)

        # Run retriever
        trainer.print(colored("Retrieve", "yellow"))
        retriever.corpus_embd_dataloader = dm.corpus_embd_dataloader()
        retriever.in_memory = (len(dm.dataset.queries) < args.embd_in_memory_threshold)
        retriever.save_file = os.path.join(task_save_path, RETRIEVE_PRED_FILENAME)
        trainer.predict(model=retriever, dataloaders=dm.queries_embd_dataloader())

        # Remove the embeddings
        if not args.save_embds and not args.load_embds and trainer.is_global_zero:
            for file in queries_embds_files + corpus_embds_files:
                if os.path.exists(file):
                    os.remove(file)

    # Run evaluation
    if trainer.is_global_zero:
        scores = run_retrieve_evaluation(dm.dataset.relevance, retriever.prediction)
        trainer.print("-" * 40)
        trainer.print("Dataset:", colored(f"{dataset_name}", "red"))
        trainer.print("Model:", colored(f"{encoder.model.model_name}", "red"))
        trainer.print("Save path:", colored(task_save_path, "yellow"))
        trainer.print("Retrieval evaluation:")
        trainer.print(scores)
        scores |= {
            "model_name": encoder.model.model_name,
            "embd_dim": encoder.model.embd_dim,
            "embd_dtype": encoder.model.embd_dtype
        }
        with open(os.path.join(task_save_path, RETRIEVE_EVAL_FILENAME), "w") as f:
            json.dump(scores, f)
        trainer.print(os.path.join(task_save_path, RETRIEVE_EVAL_FILENAME))
        return scores

    return
