from __future__ import annotations

from functools import partial
from typing import Any, Optional

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

import mteb
from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts


class NomicWrapper:
    """following the hf model card documentation."""

    def __init__(self, model_name: str, revision: str, **kwargs: Any):
        self.model_name = model_name
        self.mdl = SentenceTransformer(model_name, revision=revision, **kwargs)

    def to(self, device: torch.device) -> None:
        self.mdl.to(device)

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        prompt_name: str | None = None,
        batch_size: int = 32,
        input_type: Optional[str] = None,
        **kwargs: Any,
    ):
        if prompt_name:
            task = mteb.get_task(prompt_name)
            task_type = task.metadata.type
            if task_type in ["Classification", "MultilabelClassification"]:
                input_type = "classification"
            elif task_type == "Clustering":
                input_type = "clustering"

        # default to search_document if input_type and prompt_name are not provided
        if input_type is None:
            input_type = "search_document"

        sentences = [f"{input_type}: {sentence}" for sentence in sentences]

        emb = self.mdl.encode(sentences, batch_size=batch_size, **kwargs)
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

    def encode_queries(self, queries: list[str], batch_size: int = 32, **kwargs: Any):
        if "prompt_name" in kwargs:
            kwargs.pop("prompt_name")

        emb = self.encode(
            queries, batch_size=batch_size, input_type="search_query", **kwargs
        )

        return emb

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list[str]],
        batch_size: int = 32,
        **kwargs: Any,
    ):
        if "prompt_name" in kwargs:
            kwargs.pop("prompt_name")

        sentences = corpus_to_texts(corpus)
        emb = self.encode(
            sentences, batch_size=batch_size, input_type="search_document", **kwargs
        )
        return emb


nomic_embed_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        NomicWrapper,
        trust_remote_code=True,
        model_name="nomic-ai/nomic-embed-text-v1.5",
        revision="b0753ae76394dd36bcfb912a46018088bca48be0",
    ),
    name="nomic-ai/nomic-embed-text-v1.5",
    languages=["eng-Latn"],
    open_source=True,
    revision="b0753ae76394dd36bcfb912a46018088bca48be0",
    release_date="2024-02-10",  # first commit
)

nomic_embed_v1 = ModelMeta(
    loader=partial(  # type: ignore
        NomicWrapper,
        trust_remote_code=True,
        model_name="nomic-ai/nomic-embed-text-v1",
        revision="0759316f275aa0cb93a5b830973843ca66babcf5",
    ),
    name="nomic-ai/nomic-embed-text-v1",
    languages=["eng-Latn"],
    open_source=True,
    revision="0759316f275aa0cb93a5b830973843ca66babcf5",
    release_date="2024-01-31",  # first commit
)

if __name__ == "__main__":
    mdl = mteb.get_model(nomic_embed_v1_5.name, nomic_embed_v1_5.revision)
    emb = mdl.encode(["test"], convert_to_tensor=True)
    print(emb.shape)
    emb = mdl.encode_queries(["test"], convert_to_tensor=True)
    print(emb.shape)
    emb = mdl.encode(
        ["test"],
        convert_to_tensor=True,
        prompt_name="AmazonCounterfactualClassification",
    )
    print(emb.shape)

    mdl = mteb.get_model(nomic_embed_v1.name, nomic_embed_v1.revision)
    emb = mdl.encode(["test"], convert_to_tensor=True)
    print(emb.shape)
