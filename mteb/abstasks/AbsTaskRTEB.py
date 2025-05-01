from __future__ import annotations

import argparse
import json
import logging
import os
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense.util import cos_sim, dot_score
from datasets import DatasetDict, Value, load_dataset
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Dataset

from mteb.abstasks.TaskMetadata import HFSubset, TaskMetadata
from mteb.encoder_interface import Encoder
from mteb.load_results.task_results import ScoresDict

from .AbsTask import AbsTask

CORPUS_EMBD_FILENAME = "corpus_embds.jsonl"
QUERIES_EMBD_FILENAME = "queries_embds.jsonl"
RETRIEVE_EVAL_FILENAME = "retrieve_eval.json"
RETRIEVE_PRED_FILENAME = "retrieve_pred.json"

logger = logging.getLogger(__name__)


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca9910b1e0d218759d4/beir/datasets/data_loader_hf.py#L10
class HFDataLoader:
    def __init__(
        self,
        hf_repo: str | None = None,
        streaming: bool = False,
        keep_in_memory: bool = False,
        trust_remote_code: bool = False,
        token: str | None = None,
    ):
        self._loaded = False
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.hf_repo = hf_repo
        self.hf_repo_qrels = hf_repo  # Always use same repo

        self.streaming = streaming
        self.keep_in_memory = keep_in_memory
        self.trust_remote_code = trust_remote_code

        self.token = token or os.environ["HF_TOKEN"]

    def load(
        self, split="test"
    ) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
        if not self._loaded:
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            # logger.info("Doc Example: %s", self.corpus[0]) # Removed as self.corpus is now a Dataset

            logger.info("Loading Queries...")
            self._load_queries()

            self._load_qrels(split)
            self._loaded = True

        # filter queries with no qrels
        qrels_dict = defaultdict(dict)

        def qrels_dict_init(row):
            qrels_dict[row["query-id"]][row["corpus-id"]] = int(row["score"])

        # Check if qrels is a Dataset before mapping
        if hasattr(self.qrels, "map"):
            self.qrels.map(qrels_dict_init)
        else:
            # If not a Dataset, assume it's already a dict (e.g., from _load_qrels)
            qrels_dict = self.qrels

        # Check if queries is a Dataset before filtering
        if hasattr(self.queries, "filter"):
            self.queries = self.queries.filter(lambda x: x["id"] in qrels_dict)
        # logger.info("Loaded %d %s Queries.", len(self.queries), split.upper()) # Removed as self.queries is now a Dataset
        # logger.info("Query Example: %s", self.queries[0]) # Removed as self.queries is now a Dataset

        return self.corpus, self.queries, qrels_dict  # Return qrels_dict

    def _load_dataset(self, dataset_type: str):
        """Helper to load and standardize datasets"""
        ds = load_dataset(
            self.hf_repo,
            dataset_type,
            keep_in_memory=self.keep_in_memory,
            streaming=self.streaming,
            trust_remote_code=self.trust_remote_code,
        )
        ds = next(iter(ds.values()))  # get first split
        return ds.cast_column("id", Value("string")).remove_columns(
            [col for col in ds.column_names if col not in ["id", "text"]]
        )

    def _load_corpus(self):
        self.corpus = self._load_dataset("corpus")

    def _load_queries(self):
        self.queries = self._load_dataset("queries")

    def _load_qrels(self, split):
        qrels_ds = load_dataset(
            self.hf_repo,
            "default",
            keep_in_memory=self.keep_in_memory,
            streaming=self.streaming,
            trust_remote_code=self.trust_remote_code,
        )
        self.qrels = qrels_ds


def gather_list(data: list, num_devices: int):
    """Gather list data and merge them into a list."""
    if num_devices == 1:
        return data
    gathered = [None] * num_devices
    dist.all_gather_object(gathered, data)
    return sum(gathered, [])


def run_retrieve_evaluation(relevance, prediction):
    if len(relevance) != len(prediction):
        raise RuntimeError("Prediction and ground truth have different sizes.")

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        relevance,
        prediction,
        k_values=[1, 3, 5, 10, 20, 50, 100],
        ignore_identical_ids=False,
    )
    scores = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
    }
    return scores


class Retriever(LightningModule):
    def __init__(
        self,
        topk: int = 100,
        similarity: str = "cosine",
        save_prediction: bool = False,
    ):
        super().__init__()
        self.topk = topk
        if similarity == "cosine":
            self.similarity_fn = cos_sim
            self.largest = True
        elif similarity == "dot":
            self.similarity_fn = dot_score
            self.largest = True
        elif similarity == "euclidean":
            self.similarity_fn = torch.cdist
            self.largest = False
        else:
            raise ValueError(f"similarity {similarity} is invalid.")
        self.in_memory = True
        self.save_file = None
        self.save_prediction = save_prediction

    @property
    def local_prediction_file_name(self):
        assert self.save_file is not None
        num_shards = self.trainer.num_devices
        return f"{self.save_file}-{self.local_rank}-of-{num_shards}"

    def get_local_prediction_files(self, num_shards=None):
        assert self.save_file is not None
        if num_shards is None:
            num_shards = self.trainer.num_devices
        return [f"{self.save_file}-{i}-of-{num_shards}" for i in range(num_shards)]

    def on_predict_epoch_start(self):
        self.local_prediction = {}

    def predict_step(self, batch, batch_idx):
        query_ids, query_embds = batch["id"], batch["embd"].float()
        if isinstance(query_ids, torch.Tensor):
            # TODO: change dataloader to support int id
            raise NotImplementedError("id must be a string.")
        corpus_ids = []
        batch_scores = []
        # Compute the similarity in batches
        for corpus_batch in self.corpus_embd_dataloader:
            corpus_ids += corpus_batch["id"]
            corpus_embds = corpus_batch["embd"].float().to(query_embds.device)
            scores = self.similarity_fn(query_embds, corpus_embds).cpu()
            batch_scores.append(scores)
        # Concat the scores and compute top-k
        scores = torch.cat(batch_scores, dim=1)
        if not self.largest:
            scores = scores * -1
        topk = min(self.topk, len(corpus_ids))
        topk_scores, topk_ids = torch.topk(scores, topk, dim=1, largest=True)
        topk_scores, topk_ids = topk_scores.tolist(), topk_ids.tolist()
        for i, qid in enumerate(query_ids):
            result = OrderedDict()
            for j in range(topk):
                cid = corpus_ids[topk_ids[i][j]]
                result[cid] = topk_scores[i][j]
            self.local_prediction[qid] = result

    def on_predict_epoch_end(self):
        if self.trainer.num_devices > 1:
            if self.in_memory:
                gathered_prediction = [None] * self.trainer.num_devices
                dist.all_gather_object(gathered_prediction, self.local_prediction)
                self.prediction = {
                    k: v for preds in gathered_prediction for k, v in preds.items()
                }
            else:
                with open(self.local_prediction_file_name, "w") as f:
                    json.dump(self.local_prediction, f)
                self.trainer.strategy.barrier()
                self.prediction = {}
                if self.trainer.is_global_zero:
                    for file in self.get_local_prediction_files():
                        with open(file) as f:
                            self.prediction.update(json.load(f))
        else:
            self.prediction = self.local_prediction

        if self.save_prediction and self.trainer.is_global_zero:
            assert self.save_file is not None
            with open(self.save_file, "w") as f:
                json.dump(self.prediction, f)


class EmbeddingDataCollator:
    def __call__(self, examples):
        assert len(examples) > 0
        batch = {
            key: [example[key] for example in examples] for key in examples[0].keys()
        }
        batch["embd"] = torch.tensor(batch["embd"])
        return batch


class EmptyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Optionally apply any transformations
        if self.transform:
            item = self.transform(item)

        return item


class JSONLDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        self.data = []

        # Always convert to list for uniform processing
        file_paths = [file_path] if isinstance(file_path, str) else file_path

        for path in file_paths:
            with open(path) as f:
                for line in f:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Optionally apply any transformations
        if self.transform:
            item = self.transform(item)

        return item


class MTEBToRTEBEncoderWrapper(LightningModule):
    """Acts as a PyTorch Lightning Module to wrap an MTEB Encoder,
    replicating the necessary functionality of RTEB's Encoder class
    for use with trainer.predict, but overriding __setattr__ to prevent recursion.
    """

    @property
    def load_embds(self) -> bool:
        return self._load_embds

    @property
    def save_embds(self) -> bool:
        # If in_memory=False, we have to save the embeddings
        return self._save_embds or not self.in_memory

    @property
    def local_embd_file_name(self) -> str:
        assert self.save_file is not None
        num_shards = self.trainer.num_devices
        return f"{self.save_file}-{self.local_rank}-of-{num_shards}"

    def get_local_embd_files(self, num_shards=None) -> list[str]:
        # Return local (intermediate) file names, which are jsonl files
        assert self.save_file is not None
        if num_shards is None:
            num_shards = self.trainer.num_devices
        return [f"{self.save_file}-{i}-of-{num_shards}" for i in range(num_shards)]

    def get_embd_files(self, num_shards=None) -> list[str]:
        # Return the final file names, which are arrow files
        local_files = self.get_local_embd_files(num_shards=num_shards)
        return local_files

    def embd_files_exist(self, num_shards=None) -> bool:
        files = self.get_embd_files(num_shards=num_shards)
        return all(os.path.exists(file) for file in files)

    def on_predict_epoch_start(self):
        self.embds = None

        if self.in_memory:
            self.local_embds = []

        if self.load_embds:
            self.local_existing_ids = set()
            if os.path.exists(self.local_embd_file_name):
                logger.warning(f"Load embeddings from {self.local_embd_file_name}")
                ds = JSONLDataset(self.local_embd_file_name)
                for example in ds:
                    self.local_existing_ids.add(example["id"])
                    if self.in_memory:
                        self.local_embds.append(example)
            else:
                logger.warning(
                    f"load_embds is True but {self.local_embd_file_name} doesn't exist. Skipping the loading."
                )

        if self.save_embds:
            if self.load_embds:
                # append to the file
                self.local_embd_file = open(self.local_embd_file_name, "a")
            else:
                # rewrite the file
                self.local_embd_file = open(self.local_embd_file_name, "w")

    def on_predict_epoch_end(self):
        if self.save_embds:
            self.local_embd_file.close()
        if self.in_memory:
            self.embds = gather_list(self.local_embds, self.trainer.num_devices)
        self.trainer.strategy.barrier()

    def __init__(
        self,
        mteb_model: Encoder,
        task_name: str,
        model_name: str = "mteb_wrapped_model",
        save_embds: bool = False,
        load_embds: bool = False,
        batch_size: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._load_embds = load_embds
        self._save_embds = save_embds
        # Keep the embeddings in memory by default. Set it to False for large corpus.
        self.in_memory = True
        self.is_query = False
        self.save_file = None

        self.mteb_model_instance = mteb_model
        self.model_name = model_name
        self.task_name = task_name
        self.batch_size = batch_size
        self.query_instruct = ""  # Add instructions if applicable
        self.corpus_instruct = ""  # Add instructions if applicable
        self.embd_dim = None
        self.embd_dtype = "float32"

        # Internal state
        self.embds = None
        self.local_embds = []
        self.local_existing_ids = set()
        self.local_embd_file = None

    # --- Properties expected by run_retrieve_task ---
    @property
    def model(self):
        return self

    # --- End Properties ---

    def encode(self, sentences: list[str], **kwargs) -> torch.Tensor:
        """Encodes sentences using the wrapped MTEB model and returns torch.Tensor."""
        embeddings = self.mteb_model_instance.encode(
            sentences, batch_size=self.batch_size, **kwargs
        )
        if self.embd_dim is None and hasattr(embeddings, "shape"):
            if len(embeddings.shape) >= 2:
                self.embd_dim = embeddings.shape[1]
            elif len(embeddings.shape) == 1 and embeddings.shape[0] == 0:
                pass
            else:
                logger.warning(
                    f"Unexpected embedding shape: {embeddings.shape}. Cannot determine embd_dim."
                )

        if isinstance(embeddings, np.ndarray):
            return torch.from_numpy(embeddings).to(torch.float32)
        elif isinstance(embeddings, torch.Tensor):
            return embeddings.to(torch.float32)
        elif isinstance(embeddings, list):
            if not embeddings:
                dim = self.embd_dim if self.embd_dim is not None else 768
                return torch.empty((0, dim), dtype=torch.float32)
            if isinstance(embeddings[0], np.ndarray):
                return torch.from_numpy(np.stack(embeddings)).to(torch.float32)
            elif isinstance(embeddings[0], torch.Tensor):
                return torch.stack(embeddings).to(torch.float32)
            else:
                raise TypeError(
                    f"Unsupported embedding list element type: {type(embeddings[0])}"
                )
        else:
            raise TypeError(
                f"Unsupported embedding type from MTEB model: {type(embeddings)}"
            )

    # --- Replicated predict hooks from RtebEncoder ---
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not isinstance(batch, dict) or "id" not in batch or "text" not in batch:
            logger.error(
                f"Unsupported batch type or missing keys in predict_step: {type(batch)}"
            )
            return

        indices = batch["id"]
        sentences = batch["text"]

        if not indices or not sentences:
            return

        if self.load_embds and self.local_existing_ids:
            if all(idx in self.local_existing_ids for idx in indices):
                return
            if any(idx in self.local_existing_ids for idx in indices):
                logger.warning(
                    "Partial loading within batch detected, but not supported. Re-encoding entire batch."
                )

        try:
            embds = self.encode(sentences, task_name=self.task_name)
        except Exception as e:
            logger.error(
                f"Encoding failed for batch_idx {batch_idx}: {e}", exc_info=True
            )
            return

        for idx, embd in zip(indices, embds):
            embd_list = embd.tolist()
            obj = {"id": idx, "embd": embd_list}

            if self.in_memory:
                if not (self.load_embds and idx in self.local_existing_ids):
                    self.local_embds.append(obj)

            if self.save_embds and self.local_embd_file:
                if not (self.load_embds and idx in self.local_existing_ids):
                    try:
                        self.local_embd_file.write(json.dumps(obj) + "\n")
                    except Exception as e:
                        logger.error(
                            f"Failed to write embedding for ID {idx} to file: {e}"
                        )

    def apply(self, fn):
        # Override apply to prevent recursion into the wrapped mteb_model_instance
        super().apply(fn)
        return self

    # --- End Replicated Hooks ---


class AbsTaskRTEB(AbsTask):
    """Abstract class for retrieval experiments."""

    ignore_identical_ids: bool = False
    abstask_prompt = "Retrieve text based on user query."
    corpus: Dataset | None = None
    queries: Dataset | None = None
    relevant_docs: dict[str, dict[str, dict[str, int]]] | None = None

    def __init__(self, **kwargs):  # Require hf_repo
        self.rteb_dataset_name = kwargs.pop("rteb_dataset_name", None)
        # Derive dataset name from task name if not provided
        if self.rteb_dataset_name is None:
            # Remove "RTEB" prefix from task name to get dataset name
            self.rteb_dataset_name = self.metadata.name.replace("RTEB", "")

        self.hf_repo = f"embedding-benchmark/{self.rteb_dataset_name}"
        self._hf_data_loader = HFDataLoader(hf_repo=self.hf_repo)

        super().__init__(**kwargs)

    def _validate_task_config(self):
        """Validate task-specific configuration."""
        if not self.hf_repo:
            raise ValueError(
                f"HuggingFace repo is required for {self.__class__.__name__}"
            )
        if not self.rteb_dataset_name:
            raise ValueError(
                f"RTEB dataset name is required for {self.__class__.__name__}"
            )

    @staticmethod
    def create_rteb_task_metadata(
        task_name: str,
        dataset_name: str | None = None,
        description: str | None = None,
        reference: str | None = None,
        dataset_path: str | None = None,
        dataset_revision: str | None = None,
        eval_langs: list[str] | None = None,
        main_score: str = "ndcg_at_10",
        domains: list[str] | None = None,
        revision: str = "1.0.0",
        date: tuple[str, str] | None = None,
        license: str | None = None,
        annotations_creators: str | None = None,
        text_creation: str | None = None,
        task_subtypes: list[str] | None = None,
        dialect: list[str] | None = None,
        bibtex_citation: str | None = None,
        modalities: list[str] | None = None,
        hf_subsets_to_langscripts: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ) -> TaskMetadata:
        """Factory function to create TaskMetadata for RTEB tasks with sensible defaults.

        This function simplifies the creation of TaskMetadata objects for RTEB tasks
        by providing sensible defaults and deriving values where possible.

        Args:
            task_name: Name of the task (e.g., "RTEBLegalQuAD")
            dataset_name: Name of the dataset. If None, derived from task_name by removing "RTEB" prefix
            description: Task description. If None, generated from dataset_name
            reference: Reference URL for the dataset
            dataset_path: HuggingFace dataset path. If None, defaults to "mteb/{dataset_name}"
            dataset_revision: HuggingFace dataset revision
            eval_langs: List of evaluation languages. Defaults to ["eng-Latn"]
            main_score: Main evaluation metric. Defaults to "ndcg_at_10"
            domains: List of domains the dataset belongs to
            revision: Task revision string
            date: Tuple of (start_date, end_date) for the dataset
            license: Dataset license
            annotations_creators: How annotations were created
            text_creation: How text was created
            task_subtypes: List of task subtypes
            dialect: List of dialects
            bibtex_citation: BibTeX citation for the dataset
            modalities: List of modalities
            hf_subsets_to_langscripts: Mapping of HF subsets to language scripts
            **kwargs: Additional arguments to pass to TaskMetadata

        Returns:
            TaskMetadata object configured for the RTEB task
        """
        # Derive dataset name from task name if not provided
        if dataset_name is None:
            dataset_name = task_name.replace("RTEB", "")

        # Generate description if not provided
        if description is None:
            description = f"RTEB evaluation for {dataset_name} dataset."

        # Set default dataset path if not provided
        if dataset_path is None:
            dataset_path = f"mteb/{dataset_name}"

        # Set default date if not provided
        if date is None:
            date = ("2021-01-01", "2021-01-01")

        # Set default eval_langs if not provided
        if eval_langs is None:
            eval_langs = ["eng-Latn"]

        # Set default domains if not provided
        if domains is None:
            domains = []

        # Set default task_subtypes if not provided
        if task_subtypes is None:
            task_subtypes = []

        # Set default dialect if not provided
        if dialect is None:
            dialect = []

        # Set default modalities if not provided
        if modalities is None:
            modalities = ["text"]

        # Set default hf_subsets_to_langscripts if not provided
        if hf_subsets_to_langscripts is None:
            hf_subsets_to_langscripts = {}

        # Create dataset dictionary
        dataset_dict = {"path": dataset_path}
        if dataset_revision:
            dataset_dict["revision"] = dataset_revision

        # Create and return TaskMetadata
        return TaskMetadata(
            name=task_name,
            description=description,
            reference=reference,
            dataset=dataset_dict,
            type="Retrieval",
            category="s2p",
            eval_splits=["test"],
            eval_langs=eval_langs,
            main_score=main_score,
            revision=revision,
            date=date,
            domains=domains,
            license=license,
            annotations_creators=annotations_creators,
            text_creation=text_creation,
            task_subtypes=task_subtypes,
            dialect=dialect,
            bibtex_citation=bibtex_citation,
            modalities=modalities,
            hf_subsets_to_langscripts=hf_subsets_to_langscripts,
            **kwargs,
        )

    def load_data(self, **kwargs):
        """Load data from HuggingFace."""
        if self.data_loaded:
            return

        # Validate task configuration
        self._validate_task_config()

        logger.info(
            f"Loading data for {self.metadata.name} ({self.rteb_dataset_name}) from HuggingFace repo: {self.hf_repo}."
        )

        self.corpus, self.queries, self.relevant_docs = self._hf_data_loader.load()

        self.data_loaded = True

    def run_rteb_evaluation(
        self,
        model: Encoder,
        hf_subset: HFSubset,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> ScoresDict:
        """Runs the RTEB evaluation pipeline with pl.Trainer."""
        logger.info(
            f"Starting RTEB evaluation via PL Runner: {self.metadata.name} ({self.rteb_dataset_name})..."
        )

        if hasattr(model, "mteb_model_meta"):
            model_name = model.mteb_model_meta.name
        else:
            model_name = getattr(model, "model_name", "mteb_wrapped_model")

        # Configure Trainer
        trainer_kwargs = {
            "accelerator": kwargs.get("accelerator", "auto"),
            "devices": kwargs.get("devices", "auto"),
            "num_nodes": kwargs.get("num_nodes", 1),
            "strategy": kwargs.get("strategy", "auto"),
            "precision": kwargs.get("precision", "32-true"),
            "logger": False,  # Disable default logger
            "enable_checkpointing": False,
            "enable_progress_bar": True,
        }
        trainer = pl.Trainer(**trainer_kwargs)

        save_embds_flag = kwargs.get("save_embeddings", False)
        load_embds_flag = kwargs.get("load_embeddings", False)

        rteb_encoder = MTEBToRTEBEncoderWrapper(
            model,
            task_name=self.metadata.name,
            model_name=model_name,
            save_embds=save_embds_flag,
            load_embds=load_embds_flag,
            batch_size=batch_size,
        )
        rteb_encoder._trainer = trainer

        args = argparse.Namespace(
            save_path=kwargs.get(
                "output_folder", f"results/rteb_output/{self.rteb_dataset_name}"
            ),
            batch_size=kwargs.get("batch_size", batch_size),
            embd_batch_size=kwargs.get("embd_batch_size", 128),
            num_workers=kwargs.get("num_workers", 0),
            embd_in_memory_threshold=kwargs.get("embd_in_memory_threshold", 100000),
            overwrite=kwargs.get("overwrite_results", False),
            load_embds=load_embds_flag,  # Use the flag from kwargs
            save_embds=save_embds_flag,  # Use the flag from kwargs
        )
        task_save_path = Path(args.save_path) / model_name
        task_save_path.mkdir(parents=True, exist_ok=True)
        rteb_cache_path = Path(
            f"{os.path.expanduser('~')}/.cache/rteb/{self.rteb_dataset_name}/{model_name}"
        )
        rteb_cache_path.mkdir(parents=True, exist_ok=True)

        # Check if results already exist
        eval_file = rteb_cache_path / RETRIEVE_EVAL_FILENAME  # Use consistent filename
        if not args.overwrite and eval_file.exists():
            if trainer.is_global_zero:
                logger.info(
                    f"Results already exist for {self.metadata.name} at {eval_file}. Skipping."
                )
                with open(str(eval_file)) as f:
                    scores = json.load(f)
                return scores
            else:
                # Non-global zero ranks should wait for global zero to finish
                trainer.strategy.barrier()
                with open(str(eval_file)) as f:
                    scores = json.load(f)
                return scores

        # 1. Load Data using AbsTaskRTEB (already done by the task instance)
        try:
            query_dataloader = DataLoader(
                self.queries,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                collate_fn=None,
            )

            corpus_dataloader = DataLoader(
                self.corpus,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                collate_fn=None,
            )

            if trainer.is_global_zero:
                logger.info(f"Queries size: {len(self.queries)}")
                logger.info(f"Corpus size: {len(self.corpus)}")

            trainer.strategy.barrier()  # Ensure data is prepared on all ranks

            if (
                len(self.queries) < trainer.num_devices
                or len(self.corpus) < trainer.num_devices
            ):
                logger.warning("Skipping the task due to too few queries / documents.")
                return {}

            if len(self.queries) >= 1e6:
                logger.warning("Skipping the task due to too many queries.")
                return {}
        except Exception as e:
            logger.error(
                f"Failed to load data or create DataLoaders: {e}",
                exc_info=True,
            )
            return {
                "main_score": 0.0,
                self.metadata.main_score: 0.0,
                "hf_subset": "default",
                "languages": self.metadata.eval_langs,
            }

        # 2. Encode Queries and Corpus using pl.Trainer
        queries_embds_file = (
            task_save_path / QUERIES_EMBD_FILENAME
        )  # Use consistent filename
        corpus_embds_file = (
            task_save_path / CORPUS_EMBD_FILENAME
        )  # Use consistent filename

        # Encode Queries
        logger.info("Encoding queries")
        rteb_encoder.is_query = True
        rteb_encoder.in_memory = len(self.queries) < args.embd_in_memory_threshold
        rteb_encoder.save_file = os.path.join(task_save_path, QUERIES_EMBD_FILENAME)
        if args.load_embds and rteb_encoder.embd_files_exist(trainer.num_devices):
            queries_embds_files = rteb_encoder.get_embd_files(trainer.num_devices)
            logger.info(f"Embedding files exist: {queries_embds_files}")
            queries_embd_ds = JSONLDataset(
                queries_embds_files
            )  # Create dataset directly
        else:
            logger.info(f"in_memory = {rteb_encoder.in_memory}")
            logger.info(f"save_file = {rteb_encoder.save_file}")
            trainer.predict(
                model=rteb_encoder, dataloaders=query_dataloader
            )  # Use the new dataloader
            # Set the query embeddings
            queries_embds_files = rteb_encoder.get_embd_files()
            if rteb_encoder.in_memory:
                queries_embd_ds = EmptyDataset(
                    rteb_encoder.embds
                )  # Create dataset directly
            else:
                queries_embd_ds = JSONLDataset(
                    queries_embds_files
                )  # Create dataset directly
        trainer.strategy.barrier()  # Ensure embeddings are ready on all ranks

        # Create queries_embd_dataloader
        queries_embd_dataloader = DataLoader(
            queries_embd_ds,
            batch_size=args.embd_batch_size,
            num_workers=args.num_workers,
            collate_fn=EmbeddingDataCollator(),
        )

        # Encode Corpus
        logger.info("Encoding corpus")
        rteb_encoder.is_query = False
        rteb_encoder.in_memory = len(self.corpus) < args.embd_in_memory_threshold
        rteb_encoder.save_file = str(corpus_embds_file)

        if args.load_embds and corpus_embds_file.exists():
            if trainer.is_global_zero:
                logger.info(f"Loading corpus embeddings from {corpus_embds_file}")
            corpus_embd_ds = JSONLDataset(
                [str(corpus_embds_file)]
            )  # Create dataset directly
        else:
            if trainer.is_global_zero:
                logger.info(f"in_memory = {rteb_encoder.in_memory}")
                logger.info(f"save_file = {rteb_encoder.save_file}")
            trainer.predict(
                model=rteb_encoder, dataloaders=corpus_dataloader
            )  # Use the new dataloader
            if rteb_encoder.in_memory:
                corpus_embd_ds = EmptyDataset(
                    rteb_encoder.embds
                )  # Create dataset directly
            else:
                corpus_embd_ds = JSONLDataset(
                    [str(corpus_embds_file)]
                )  # Create dataset directly

        trainer.strategy.barrier()  # Ensure embeddings are ready on all ranks

        # Create corpus_embd_dataloader
        corpus_embd_dataloader = DataLoader(
            corpus_embd_ds,
            batch_size=args.embd_batch_size,
            num_workers=args.num_workers,
            collate_fn=EmbeddingDataCollator(),
        )

        # 3. Manually Perform Retrieval
        logger.info("Retrieve")
        retriever_instance = Retriever(topk=100)  # Instantiate Retriever
        retriever_instance.corpus_embd_dataloader = (
            corpus_embd_dataloader  # Use the new dataloader
        )
        retriever_instance.in_memory = len(self.queries) < args.embd_in_memory_threshold
        retriever_instance.save_file = str(
            rteb_cache_path / RETRIEVE_PRED_FILENAME
        )  # Use consistent filename
        retriever_instance.save_prediction = True  # Ensure prediction is saved

        trainer.predict(
            model=retriever_instance,
            dataloaders=queries_embd_dataloader,  # Use the new dataloader
        )

        # Remove the embeddings if not saving
        if not args.save_embds and not args.load_embds and trainer.is_global_zero:
            if queries_embds_file.exists():
                os.remove(queries_embds_file)
            if corpus_embds_file.exists():
                os.remove(corpus_embds_file)

        # 4. Run Evaluation
        rteb_scores = {}
        if trainer.is_global_zero:
            try:
                # Load predictions from the file saved by the retriever
                prediction_file = rteb_cache_path / RETRIEVE_PRED_FILENAME
                if not prediction_file.exists():
                    logger.error(f"Prediction file not found at {prediction_file}")
                    raise FileNotFoundError(
                        f"Prediction file not found at {prediction_file}"
                    )

                with open(str(prediction_file)) as f:
                    predictions = json.load(f)

                filtered_predictions = {
                    qid: scores
                    for qid, scores in predictions.items()
                    if qid in self.relevant_docs
                }
                if len(filtered_predictions) != len(self.relevant_docs):
                    logger.warning(
                        f"Number of queries in predictions ({len(filtered_predictions)}) does not match relevance data ({len(self.relevant_docs)}). Evaluating on intersection."
                    )
                    filtered_relevance = {
                        qid: scores
                        for qid, scores in self.relevant_docs.items()
                        if qid in filtered_predictions
                    }
                else:
                    filtered_relevance = self.relevant_docs

                if not filtered_predictions:
                    logger.error(
                        "No overlapping queries between predictions and relevance data."
                    )
                    raise ValueError("No queries to evaluate.")

                rteb_scores = run_retrieve_evaluation(
                    filtered_relevance, filtered_predictions
                )

                logger.info("-" * 40)
                logger.info(f"Dataset: {self.rteb_dataset_name}")
                logger.info(f"Model: {model_name}")
                logger.info(f"Save path: {task_save_path}")
                logger.info("Retrieval evaluation:")
                logger.info(rteb_scores)  # Log the scores dictionary

                # 5. Format and Save Results
                mteb_scores = dict(rteb_scores)
                if self.metadata.main_score not in mteb_scores:
                    logger.warning(
                        f"Main score '{self.metadata.main_score}' not found in RTEB results."
                    )
                    fallback_score = (
                        next(iter(mteb_scores.values()), 0.0) if mteb_scores else 0.0
                    )
                    mteb_scores["main_score"] = fallback_score
                else:
                    mteb_scores["main_score"] = mteb_scores[self.metadata.main_score]

                mteb_scores["model_name"] = model_name
                if rteb_encoder.embd_dim:
                    mteb_scores["embd_dim"] = rteb_encoder.embd_dim
                mteb_scores["embd_dtype"] = rteb_encoder.embd_dtype

                keys_to_remove = ["model_name", "embd_dim", "embd_dtype"]
                final_scores = {}
                for key, value in mteb_scores.items():
                    if key not in keys_to_remove:
                        try:
                            final_scores[key] = float(value)
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Could not convert score '{key}' to float. Skipping."
                            )

                if "main_score" not in final_scores and "main_score" in mteb_scores:
                    try:
                        final_scores["main_score"] = float(mteb_scores["main_score"])
                    except (ValueError, TypeError):
                        final_scores["main_score"] = 0.0

                final_scores["hf_subset"] = (
                    hf_subset if self.is_multilingual else "default"
                )
                final_scores["languages"] = self.metadata.eval_langs

                with open(str(eval_file), "w") as f:
                    json.dump(final_scores, f)
                logger.info(f"Results saved to: {eval_file}")
                rteb_scores = final_scores  # Return the final formatted scores

            except Exception as e:
                logger.error(
                    f"Error during score calculation or saving: {e}", exc_info=True
                )
                rteb_scores = {
                    "main_score": 0.0,
                    self.metadata.main_score: 0.0,
                    "hf_subset": hf_subset if self.is_multilingual else "default",
                    "languages": self.metadata.eval_langs,
                }

        trainer.strategy.barrier()  # Ensure global zero finishes saving before other ranks proceeds

        # If not global zero, wait for global zero to save and then load the results
        if not trainer.is_global_zero:
            if eval_file.exists():
                with open(str(eval_file)) as f:
                    rteb_scores = json.load(f)
            else:
                logger.error(
                    f"Evaluation file not found on non-global zero rank: {eval_file}"
                )
                rteb_scores = {
                    "main_score": 0.0,
                    self.metadata.main_score: 0.0,
                    "hf_subset": hf_subset if self.is_multilingual else "default",
                    "languages": self.metadata.eval_langs,
                }

        logger.info(f"Finished RTEB evaluation for {self.metadata.name}.")
        return rteb_scores

    def evaluate(
        self,
        model,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> dict[HFSubset, ScoresDict]:
        """Evaluate the model using the RTEB task runner."""
        if not self.data_loaded:
            self.load_data()

        # RTEB tasks handle subsets internally based on dataset name
        scores = {}
        hf_subsets = list(self.hf_subsets) if self.is_multilingual else ["default"]
        if subsets_to_run is not None:
            hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

        for hf_subset in hf_subsets:
            logger.info(
                f"Task: {self.metadata.name}, split: {split}, subset: {hf_subset}. Running..."
            )

            scores[hf_subset] = self.run_rteb_evaluation(
                model=model,
                hf_subset=hf_subset,
                encode_kwargs=encode_kwargs,
                batch_size=16,
                **kwargs,
            )

        return scores

    def _evaluate_subset(
        self,
        model: Encoder,
        data_split: DatasetDict | Dataset,
        encode_kwargs: dict[str, Any],
        **kwargs: Any,
    ):
        """Evaluate a subset of the dataset.

        Warning:
            This method is deprecated and will be removed in future versions.
            Use RTEBTaskRunner.run_rteb_evaluation for evaluation logic.

        Delegates to the parent class implementation while issuing a deprecation warning.
        """
        import warnings

        warnings.warn(
            "_evaluate_subset is deprecated for RTEB tasks. Use RTEBTaskRunner.run_rteb_evaluation instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return super()._evaluate_subset(model, data_split, encode_kwargs, **kwargs)

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ):
        """Calculate metrics for a given split.

        Note:
            This method exists only for API compatibility. Actual metric calculation
            happens in RTEBTaskRunner.run_rteb_evaluation. This implementation:
            1. Logs a warning when called
            2. Returns empty ScoresDict to satisfy interface requirements

        Parameters:
            split: Dataset split to evaluate (e.g., 'test')
            hf_subset: Optional Hugging Face dataset subset name
            compute_overall: Whether to compute overall metrics across subsets

        Returns:
            ScoresDict: Empty dictionary to maintain interface compatibility
        """
        logger.warning(
            f"_calculate_metrics_from_split called for split {split}, but metrics are calculated by RTEBTaskRunner."
        )
        return ScoresDict()
