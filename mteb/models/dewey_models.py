from __future__ import annotations

import logging
import torch
import mteb
import numpy as np
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable
from sentence_transformers import SentenceTransformer
from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.e5_instruct import E5_MISTRAL_TRAINING_DATA
from mteb.models.wrapper import Wrapper

logger = logging.getLogger(__name__)


class DeweyWrapper(Wrapper):
    def __init__(
            self,
            model_name: str,
            revision: str,
            instruction_template: str | Callable[[str], str] | None = None,
            max_seq_length: int = 1536,
            **kwargs: Any,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, revision=revision, **kwargs)
        self.instruction_template = instruction_template
        self.model.max_seq_length = max_seq_length

    def encode(
            self,
            sentences: Sequence[str],
            *,
            task_name: str,
            prompt_type: PromptType | None = None,
            **kwargs: Any,
    ) -> np.ndarray:
        task_type = mteb.get_task(task_name=task_name).metadata.type
        if task_type == "Retrieval":
            if prompt_type.value == "query":
                prompt_name = "retrieve_query"
            else:
                prompt_name = "retrieve_passage"
        elif task_type in ["STS", "PairClassification", "Summarization", "Reranking"]:
            prompt_name = "sts"
        else:
            prompt_name = task_name

        embeddings = self.model.encode(
            list(sentences),
            normalize_embeddings=True,
            prompt_name=prompt_name,
            **kwargs,
        )

        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings


dewey_en_beta = ModelMeta(
    loader=partial(  # type: ignore
        DeweyWrapper,
        model_name="infgrad/dewey_en_beta",
        revision="286d85cb42af9080a8382eea648a71015ef8879e",
        # short text  (e.g. mteb-eng-v2, beir)  using cls_add_mean
        # long text (e.g. Loco, LongEmbed) using mean
        config_kwargs={"single_vector_type": "mean"},
        model_kwargs={
            "attn_implementation": "flash_attention_2",
            "torch_dtype": torch.bfloat16,
        },
        trust_remote_code=True,
        max_seq_length=64 * 1024,
    ),
    name="infgrad/dewey_en_beta",
    languages=["eng-Latn"],
    open_weights=True,
    revision="d6330ce98f8a0d741e781df845904c9484f00efa",
    release_date="2025-03-24",  # first commit
    n_parameters=400_000_000,
    memory_usage_mb=1522,
    max_tokens=131072,
    embed_dim=2048,
    license="mit",
    reference="https://huggingface.co/infgrad/dewey_en_beta",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
    # this model used Linq-Embed-Mistral as teacher model whose training dataset is E5_MISTRAL_TRAINING_DATA
    training_datasets={
        **E5_MISTRAL_TRAINING_DATA
    },
    public_training_code=None,
    public_training_data=None,
)
