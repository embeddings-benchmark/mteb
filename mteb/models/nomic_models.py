from __future__ import annotations

import logging
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import (
    get_prompt_name,
    validate_task_to_prompt_name,
)

from .wrapper import Wrapper

logger = logging.getLogger(__name__)


class NomicWrapper(Wrapper):
    """following the hf model card documentation."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        model_prompts: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, revision=revision, **kwargs)
        self.model_prompts = (
            validate_task_to_prompt_name(model_prompts) if model_prompts else None
        )

    def to(self, device: torch.device) -> None:
        self.model.to(device)

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        input_type = get_prompt_name(self.model_prompts, task_name, prompt_type)

        # default to search_document if input_type and prompt_name are not provided
        if input_type is None:
            input_type = "search_document"

        sentences = [f"{input_type}: {sentence}" for sentence in sentences]

        emb = self.model.encode(sentences, batch_size=batch_size, **kwargs)
        # v1.5 has a non-trainable layer norm to unit normalize the embeddings for binary quantization
        # the outputs are similar to if we just normalized but keeping the same for consistency
        if self.model_name == "nomic-ai/nomic-embed-text-v1.5":
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb)
            emb = F.layer_norm(emb, normalized_shape=(emb.shape[1],))
            emb = F.normalize(emb, p=2, dim=1)
            if kwargs.get("convert_to_tensor", False):
                emb = emb.cpu().detach().numpy()

        return emb


model_prompts = {
    "Classification": "classification: ",
    "MultilabelClassification": "classification: ",
    "Clustering": "clustering: ",
    PromptType.query.value: "search_query: ",
    PromptType.passage.value: "search_document: ",
}

nomic_embed_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        NomicWrapper,
        trust_remote_code=True,
        model_name="nomic-ai/nomic-embed-text-v1.5",
        revision="b0753ae76394dd36bcfb912a46018088bca48be0",
        model_prompts=model_prompts,
    ),
    name="nomic-ai/nomic-embed-text-v1.5",
    languages=["eng-Latn"],
    open_weights=True,
    revision="b0753ae76394dd36bcfb912a46018088bca48be0",
    release_date="2024-02-10",  # first commit
)

nomic_embed_v1 = ModelMeta(
    loader=partial(  # type: ignore
        NomicWrapper,
        trust_remote_code=True,
        model_name="nomic-ai/nomic-embed-text-v1",
        revision="0759316f275aa0cb93a5b830973843ca66babcf5",
        model_prompts=model_prompts,
    ),
    name="nomic-ai/nomic-embed-text-v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="0759316f275aa0cb93a5b830973843ca66babcf5",
    release_date="2024-01-31",  # first commit
    n_parameters=None,
    memory_usage=None,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/nomic-ai/nomic-embed-text-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instuctions=True,
)
