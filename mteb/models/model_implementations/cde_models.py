import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import mteb
from mteb._create_dataloaders import _corpus_to_dict
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.models_protocols import PromptType
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.types import Array, BatchedInput

from .bge_models import bge_full_data

if TYPE_CHECKING:
    from mteb.abstasks import (
        AbsTaskClassification,
        AbsTaskRetrieval,
        AbsTaskSummarization,
    )
logger = logging.getLogger(__name__)

CDE_CITATION = """@misc{morris2024contextualdocumentembeddings,
    title={Contextual Document Embeddings},
    author={John X. Morris and Alexander M. Rush},
    year={2024},
    eprint={2410.02525},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2410.02525},
}"""


class CDEWrapper(SentenceTransformerEncoderWrapper):
    dataset_embeddings: torch.Tensor | None = None
    prev_embeddings_key: tuple[str, str] | None = None
    classification_task_types = (
        "Classification",
        "MultilabelClassification",
    )
    retrieval_task_types = (
        "Retrieval",
        "Reranking",
        "InstructionRetrieval",
        "InstructionReranking",
    )

    def __init__(self, model: str, *args, **kwargs: Any) -> None:
        from transformers import AutoConfig

        super().__init__(model, *args, **kwargs)
        model_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        self.max_sentences = model_config.transductive_corpus_size

    def _create_embeddings_key(
        self,
        task_metadata: TaskMetadata,
        hf_subset: str,
    ) -> tuple[str, str]:
        """Create a key for the embeddings based on task metadata and subset.

        Args:
            task_metadata: Task metadata containing the task name.
            hf_subset: The subset of the task, e.g., "train", "test", etc.

        Returns:
            A tuple containing the task name and the subset.
        """
        return (
            task_metadata.name,
            hf_subset,
        )

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        prompt = self.get_prompt(task_metadata, prompt_type)
        if prompt:
            logger.info(
                f"Using prompt=`{prompt}` for task={task_metadata.name} prompt_type={prompt_type}"
            )
        else:
            logger.info(
                f"No model prompts found for task={task_metadata.name} prompt_type={prompt_type}"
            )
        sentences = [text for batch in inputs for text in batch["text"]]
        self._load_task_sample(
            sentences,
            task_metadata,
            prompt_type,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **kwargs,
        )

        logger.info(f"Encoding {len(sentences)} sentences.")
        if self.dataset_embeddings is None:
            raise ValueError("Dataset embeddings are not loaded")

        embeddings = self.model.encode(
            sentences,
            prompt=prompt,
            dataset_embeddings=self.dataset_embeddings,
            **kwargs,
        )
        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()

        self.prev_embeddings_key = self._create_embeddings_key(task_metadata, hf_subset)
        return embeddings

    def _load_task_sample(
        self,
        sentences: Sequence[str],
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
        hf_split: str,
        hf_subset: str,
        **kwargs: Any,
    ) -> None:
        """Load a sample of the task dataset to create embeddings for the sentences.

        Args:
            sentences: List of sentences to encode.
            task_metadata: Task metadata containing the task name and type.
            prompt_type: The type of prompt to use (query or passage).
            hf_split: Split of the task, e.g., "train", "test", etc.
            hf_subset: Subset of the task, e.g., "train", "validation", etc.
            **kwargs: Encoding arguments to pass to the model.
        """
        if self.prev_embeddings_key == self._create_embeddings_key(
            task_metadata, hf_subset
        ):
            logger.info(
                f"Embeddings for {task_metadata.name} subset: {hf_subset} already loaded"
            )
            return
        prompt = "search_document: "
        logger.info(
            f"Loading dataset embeddings for task {task_metadata.name}, subset: {hf_subset} "
            f"Prompt: `{prompt}`"
        )
        if task_metadata.type in self.retrieval_task_types:
            task: AbsTaskRetrieval = mteb.get_task(task_metadata.name)
            task.load_data()
            task.convert_v1_dataset_format_to_v2()
            cur_ds = task.dataset[hf_subset][hf_split]["corpus"]
            sentences = cur_ds.map(_corpus_to_dict)["text"]
        elif task_metadata.type in self.classification_task_types:
            task: AbsTaskClassification = mteb.get_task(task_metadata.name)
            task.load_data()
            if hf_subset in task.dataset:
                cur_ds = task.dataset[hf_subset][task.train_split]
            else:
                cur_ds = task.dataset[task.train_split]
            sentences = cur_ds[task.input_column_name]
        elif task_metadata.type == "Summarization":
            task: AbsTaskSummarization = mteb.get_task(task_metadata.name)
            task.load_data()
            if hf_subset in task.dataset:
                cur_ds = task.dataset[hf_subset][hf_split]
            else:
                cur_ds = task.dataset[hf_split]
            sentences = cur_ds["text"]

        # We need to sample with replacement if the minicorpus needs to be bigger than the number of sentences
        is_replace = len(sentences) < self.max_sentences
        rng_state = np.random.default_rng(42)
        minicorpus = rng_state.choice(
            sentences, size=self.max_sentences, replace=is_replace
        )
        self.dataset_embeddings = self.model.encode(
            minicorpus,
            prompt=prompt,
            convert_to_tensor=True,
            **kwargs,
        )


cde_model_prompts = {
    # same as for nomic models
    "Classification": "classification: ",
    "MultilabelClassification": "classification: ",
    "Clustering": "clustering: ",
    "PairClassification": "classification: ",
    "Reranking": "classification: ",
    "STS": "classification: ",
    "Summarization": "classification: ",
    PromptType.query.value: "search_query: ",
    PromptType.document.value: "search_document: ",
}

cde_small_v1 = ModelMeta(
    loader=CDEWrapper,
    loader_kwargs=dict(
        model_prompts=cde_model_prompts,
        trust_remote_code=True,
    ),
    name="jxm/cde-small-v1",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="e151df18af0d7f1d1c37b074fee58406ececf19f",
    release_date="2024-09-24",
    n_parameters=int(281 * 1e6),
    memory_usage_mb=1072,  # Though the second-stage model is only 140M
    max_tokens=512,
    embed_dim=768,
    license="mit",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers"],
    reference="https://huggingface.co/jxm/cde-small-v1",
    use_instructions=True,
    adapted_from="nomic-ai/nomic-bert-2048",
    superseded_by="jxm/cde-small-v2",
    training_datasets=bge_full_data,
    public_training_code="https://github.com/jxmorris12/cde",
    public_training_data="https://huggingface.co/datasets/cfli/bge-full-data",
    citation=CDE_CITATION,
)

cde_small_v2 = ModelMeta(
    loader=CDEWrapper,
    loader_kwargs=dict(
        model_prompts=cde_model_prompts,
        trust_remote_code=True,
    ),
    name="jxm/cde-small-v2",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="4e1d021a6c3fd7ce8aa0a7204057eee5ae61d390",
    release_date="2025-01-13",
    n_parameters=int(306 * 1e6),
    memory_usage_mb=1166,  # Though the second-stage model is only 140M
    max_tokens=512,
    embed_dim=768,
    license="mit",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers"],
    reference="https://huggingface.co/jxm/cde-small-v1",
    use_instructions=True,
    adapted_from="answerdotai/ModernBERT-base",
    superseded_by="jxm/cde-small-v2",
    training_datasets=bge_full_data,
    public_training_code="https://github.com/jxmorris12/cde",
    public_training_data="https://huggingface.co/datasets/cfli/bge-full-data",
    citation=CDE_CITATION,
)
