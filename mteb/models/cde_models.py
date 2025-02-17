from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from math import ceil
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig

from mteb.encoder_interface import PromptType
from mteb.evaluation import corpus_to_str
from mteb.model_meta import ModelMeta, ScoringFunction
from mteb.models.bge_models import bge_full_data
from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper

logger = logging.getLogger(__name__)


class CDEWrapper(SentenceTransformerWrapper):
    dataset_embeddings: torch.Tensor = None

    def __init__(self, model: str, *args, **kwargs: Any) -> None:
        super().__init__(model, *args, **kwargs)
        model_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        self.max_sentences = model_config.transductive_corpus_size

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        prompt_name = self.get_prompt_name(self.model_prompts, task_name, prompt_type)
        if prompt_name:
            logger.info(
                f"Using prompt_name={prompt_name} for task={task_name} prompt_type={prompt_type}"
            )
        else:
            logger.info(
                f"No model prompts found for task={task_name} prompt_type={prompt_type}"
            )
        logger.info(f"Encoding {len(sentences)} sentences.")
        if self.dataset_embeddings is None:
            raise ValueError("Dataset embeddings are not loaded")

        embeddings = self.model.encode(
            sentences,
            prompt_name=prompt_name,
            dataset_embeddings=self.dataset_embeddings,
            **kwargs,
        )
        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings

    def load_task_sample(
        self,
        sentences: Sequence[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> None:
        logger.info(
            f"Loading dataset embeddings for task {task_name}. Prompt type: {prompt_type}"
        )
        if isinstance(sentences[0], list):
            sentences = [s for sentences_list in sentences for s in sentences_list]
        # We need to sample with replacement if the minicorpus needs to be bigger than the number of sentences
        is_replace = len(sentences) < self.max_sentences
        minicorpus = np.random.choice(sentences, size=self.max_sentences, replace=is_replace)
        prompt_name= self.get_prompt_name(
            self.model_prompts, task_name, prompt_type
        )

        self.dataset_embeddings = self.model.encode(
            corpus_to_str(minicorpus),
            prompt_name=prompt_name,
            convert_to_tensor=True,
            **kwargs,
        )


cde_model_prompts = {
    PromptType.query.value: "search_query: ",
    PromptType.passage.value: "search_document: ",
}

cde_small_v1 = ModelMeta(
    loader=partial(
        CDEWrapper,
        model="jxm/cde-small-v1",
        # revision="7017cdcb2abeccc8e9abd1c2379eb0e05121eec8",
        model_prompts=cde_model_prompts,
        trust_remote_code=True,
    ),
    name="jxm/cde-small-v1",
    languages=["eng_Latn"],
    open_weights=True,
    revision="7017cdcb2abeccc8e9abd1c2379eb0e05121eec8",
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
)

cde_small_v2 = ModelMeta(
    loader=partial(
        CDEWrapper,
        model="jxm/cde-small-v2",
        # revision="a7e5882ad52c27ea2831fc8258f24379c25cb459",
        model_prompts=cde_model_prompts,
        trust_remote_code=True,
    ),
    name="jxm/cde-small-v2",
    languages=["eng_Latn"],
    open_weights=True,
    revision="a7e5882ad52c27ea2831fc8258f24379c25cb459",
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
)
